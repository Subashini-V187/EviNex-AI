"""EviNex AI: ask questions about document contents."""

from __future__ import annotations

import hashlib
import io
import os

import streamlit as st
from dotenv import load_dotenv

from pdf_processor import (
    FALLBACK_ANSWER,
    chunk_document_pages,
    create_gemini_client,
    embed_document_chunks,
    extract_document,
    extract_pdf_pages,
    generate_grounded_answer,
    generate_image_answer,
    generate_scanned_pdf_answer,
    render_pdf_page,
    retrieve_relevant_content,
)

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024
PREVIEW_CHARACTER_LIMIT = 5000

# Documents with less text than this are sent to Gemini in full.
FULL_CONTEXT_CHAR_LIMIT = 200_000

# PDFs with at most this many pages are sent to Gemini as the original file,
# so it can read charts, diagrams and scanned pages.
PDF_DIRECT_MAX_PAGES = 40

SUPPORTED_TYPES = [
    "pdf",
    "doc",
    "docx",
    "xls",
    "xlsx",
    "pptx",
    "txt",
    "md",
    "csv",
    "json",
    "png",
    "jpg",
    "jpeg",
]


# ============================================================
# HELPERS
# ============================================================

def get_client():
    """Create the Gemini client, using Streamlit Secrets if available."""
    api_key = None

    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        api_key = None

    return create_gemini_client(api_key)


def make_result(
    question,
    answer,
    evidence,
    error,
    document_hash,
    chart=None,
    show_pages=None,
):
    return {
        "question": question,
        "answer": answer,
        "evidence": evidence,
        "error": error,
        "document_hash": document_hash,
        "chart": chart,
        "show_pages": show_pages or [],
    }


def result_from_answer(question, answer, document_hash):
    return make_result(
        question,
        answer.get("answer"),
        answer.get("evidence", []),
        None,
        document_hash,
        chart=answer.get("chart"),
        show_pages=answer.get("show_pages", []),
    )


def friendly_error(exc):
    text = str(exc)

    if "GEMINI_API_KEY is not available" in text:
        return (
            "The Gemini API key is not available to the app. "
            "Check that GEMINI_API_KEY is saved in Streamlit Secrets."
        )

    return f"Gemini could not complete the request: {text}"


# ============================================================
# CHART AND IMAGE OUTPUT
# ============================================================

def render_chart(chart):
    """Draw a chart returned by Gemini and offer a PNG download."""
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np

    categories = chart["categories"]
    series = chart["series"]
    chart_type = chart["chart_type"]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    if chart_type == "pie":
        ax.pie(
            series[0]["values"],
            labels=categories,
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.axis("equal")

    elif chart_type == "line":
        for item in series:
            ax.plot(categories, item["values"], marker="o", label=item["name"])

            for x_value, y_value in zip(categories, item["values"]):
                ax.annotate(
                    f"{y_value:g}",
                    (x_value, y_value),
                    textcoords="offset points",
                    xytext=(0, 6),
                    ha="center",
                    fontsize=8,
                )

    else:
        positions = np.arange(len(categories))
        count = len(series)
        width = 0.8 / count

        for index, item in enumerate(series):
            offset = (index - (count - 1) / 2) * width

            bars = ax.bar(
                positions + offset,
                item["values"],
                width,
                label=item["name"],
            )

            ax.bar_label(bars, fmt="%g", fontsize=8, padding=2)

        ax.set_xticks(positions)
        ax.set_xticklabels(categories)

    if chart_type != "pie":
        ax.set_xlabel(chart.get("x_label", ""))
        ax.set_ylabel(chart.get("y_label", ""))
        ax.spines[["top", "right"]].set_visible(False)

        if len(series) > 1:
            ax.legend()

    if chart.get("title"):
        ax.set_title(chart["title"], fontweight="bold")

    fig.tight_layout()

    st.pyplot(fig)

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    st.download_button(
        "Download chart (PNG)",
        data=buffer.getvalue(),
        file_name="evinex_chart.png",
        mime="image/png",
    )


def display_result(result, filename, file_bytes, extension, pages=None):
    """Show the answer, optional chart, optional page images and evidence."""
    st.divider()
    st.subheader("AI Answer")

    if result.get("error"):
        st.error(result["error"])
        return

    st.write(result["answer"])

    # ---------------- chart ----------------
    chart = result.get("chart")

    if chart:
        st.subheader("Chart")

        try:
            render_chart(chart)
        except Exception as exc:
            st.warning(f"The chart could not be drawn: {exc}")

    # ---------------- page images (PDF only) ----------------
    show_pages = result.get("show_pages") or []

    if show_pages and extension == ".pdf":
        st.subheader("Referenced pages")

        for page_number in show_pages:
            image = render_pdf_page(file_bytes, page_number)

            if image:
                st.image(
                    image,
                    caption=f"{filename} · Page {page_number}",
                    use_container_width=True,
                )

    # ---------------- evidence ----------------
    evidence = result.get("evidence", [])

    if evidence:
        st.subheader("Evidence")

        for item in evidence:
            st.write(f'"{item["quote"]}"')

            page_number = item["page_number"]
            source_label = item.get("source_label") or f"Page {page_number}"

            st.caption(f"Source: {filename} · {source_label}")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EviNex AI",
    page_icon="📄",
    layout="centered",
)

st.title("EviNex AI")

st.write(
    "Upload a document and ask a question. "
    "Answers are based only on the document. "
    "Ask for a chart or to see a figure and the answer will include it."
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Choose a document",
    type=SUPPORTED_TYPES,
)

if uploaded_file is None:
    st.info("Upload a document to see its details and ask a question.")
    st.stop()


# ============================================================
# READ FILE
# ============================================================

file_bytes = uploaded_file.getvalue()
filename = uploaded_file.name
extension = os.path.splitext(filename)[1].lower()

if len(file_bytes) > MAX_FILE_SIZE_BYTES:
    st.error("This file is larger than the 20 MB limit.")
    st.stop()

document_hash = hashlib.sha256(file_bytes).hexdigest()


# ============================================================
# IMAGE FILES
# ============================================================

if extension in (".png", ".jpg", ".jpeg"):

    st.subheader("Document information")
    st.write(f"**Filename:** {filename}")
    st.write("**Type:** Image")

    with st.expander("Image preview", expanded=True):
        st.image(file_bytes, caption=filename, use_container_width=True)

    with st.form("image_question_form"):
        question = st.text_input(
            "Ask a question about this image",
            placeholder="For example: What information is shown in this image?",
            max_chars=500,
        )
        submitted = st.form_submit_button("Ask")

    if submitted:
        if not question.strip():
            st.warning("Enter a question to continue.")
        else:
            try:
                with st.spinner("Analyzing the image..."):
                    client = get_client()

                    mime_type = (
                        "image/png" if extension == ".png" else "image/jpeg"
                    )

                    answer = generate_image_answer(
                        client,
                        file_bytes,
                        mime_type,
                        question.strip(),
                    )

                st.session_state["answer_result"] = result_from_answer(
                    question.strip(), answer, document_hash
                )

            except Exception as exc:
                st.session_state["answer_result"] = make_result(
                    question.strip(), None, [], friendly_error(exc), document_hash
                )

    result = st.session_state.get("answer_result")

    if result and result.get("document_hash") == document_hash:
        display_result(result, filename, file_bytes, extension)

    st.stop()


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

if st.session_state.get("document_hash") != document_hash:

    try:
        if extension == ".pdf":
            pages = extract_pdf_pages(file_bytes)
        else:
            pages = extract_document(file_bytes, filename)

    except Exception as exc:
        st.error(f"Document processing failed: {exc}")
        st.stop()

    st.session_state["document_hash"] = document_hash
    st.session_state["document_pages"] = pages
    st.session_state["document_chunks"] = chunk_document_pages(pages)
    st.session_state["embeddings_ready"] = False
    st.session_state.pop("answer_result", None)


pages = st.session_state["document_pages"]


# ============================================================
# EXTRACTED TEXT
# ============================================================

all_text = "\n\n".join(
    f"Page {page['page_number']}\n{page.get('text', '')}"
    for page in pages
    if page.get("text")
)


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

st.subheader("Document information")
st.write(f"**Filename:** {filename}")
st.write(f"**File type:** {extension.upper()}")
st.write(f"**Sections/pages detected:** {len(pages)}")


# ============================================================
# TEXT PREVIEW
# ============================================================

with st.expander("Extracted text preview", expanded=False):

    if all_text:
        st.text_area(
            "Text extracted from the document",
            value=all_text[:PREVIEW_CHARACTER_LIMIT],
            height=220,
            disabled=True,
        )

        if len(all_text) > PREVIEW_CHARACTER_LIMIT:
            st.caption(
                f"Preview limited to the first "
                f"{PREVIEW_CHARACTER_LIMIT:,} characters."
            )
    else:
        st.warning("No selectable text was found.")


# ============================================================
# SCANNED / DIRECT PDF DETECTION
# ============================================================

is_scanned_pdf = extension == ".pdf" and not bool(all_text)

use_direct_pdf = extension == ".pdf" and (
    is_scanned_pdf or len(pages) <= PDF_DIRECT_MAX_PAGES
)

if is_scanned_pdf:
    st.info(
        "This appears to be a scanned PDF. "
        "EviNex AI will use Gemini's multimodal "
        "document understanding to analyze the pages."
    )


# ============================================================
# QUESTION FORM
# ============================================================

with st.form("document_question_form"):
    question = st.text_input(
        "Ask a question about this document",
        placeholder=(
            "For example: Show a bar chart of quarterly units produced"
        ),
        max_chars=500,
    )
    submitted = st.form_submit_button("Ask")


# ============================================================
# QUESTION PROCESSING
# ============================================================

if submitted:

    if not question.strip():
        st.warning("Enter a question to continue.")

    else:
        try:
            with st.spinner("Analyzing the document and asking Gemini..."):

                client = get_client()

                # ---------------- PDF sent as the original file ----------------
                if use_direct_pdf:

                    answer = generate_scanned_pdf_answer(
                        client,
                        file_bytes,
                        question.strip(),
                    )

                    answer_result = result_from_answer(
                        question.strip(), answer, document_hash
                    )

                # ---------------- TEXT-BASED DOCUMENT ----------------
                else:

                    full_pages = []

                    for page in pages:
                        parts = [page.get("text", "")]

                        for table in page.get("tables", []):
                            for row in table.get("data", []):
                                parts.append(" | ".join(str(v) for v in row))

                        page_text = "\n".join(p for p in parts if p)

                        if page_text.strip():
                            full_pages.append(
                                {
                                    "page_number": page.get("page_number", 1),
                                    "text": page_text,
                                    "source_label": page.get(
                                        "source_label", "Document"
                                    ),
                                    "score": 1.0,
                                }
                            )

                    total_chars = sum(len(p["text"]) for p in full_pages)

                    if total_chars <= FULL_CONTEXT_CHAR_LIMIT:
                        # Small document: give Gemini everything
                        relevant_chunks = full_pages

                    else:
                        # Large document: embedding retrieval
                        document_chunks = st.session_state.get("document_chunks")

                        if document_chunks is None:
                            document_chunks = chunk_document_pages(pages)

                        if not st.session_state.get("embeddings_ready"):
                            document_chunks = embed_document_chunks(
                                client,
                                document_chunks,
                            )
                            st.session_state["document_chunks"] = document_chunks
                            st.session_state["embeddings_ready"] = True

                        relevant_chunks = retrieve_relevant_content(
                            client,
                            document_chunks,
                            question.strip(),
                            top_k=8,
                        )

                    if not relevant_chunks:
                        answer_result = make_result(
                            question.strip(),
                            FALLBACK_ANSWER,
                            [],
                            None,
                            document_hash,
                        )

                    else:
                        answer = generate_grounded_answer(
                            client,
                            question.strip(),
                            relevant_chunks,
                        )

                        answer_result = result_from_answer(
                            question.strip(), answer, document_hash
                        )

                st.session_state["answer_result"] = answer_result

        except Exception as exc:
            st.session_state["answer_result"] = make_result(
                question.strip(), None, [], friendly_error(exc), document_hash
            )


# ============================================================
# DISPLAY ANSWER
# ============================================================

result = st.session_state.get("answer_result")

if result and result.get("document_hash") == document_hash:
    display_result(result, filename, file_bytes, extension, pages)
