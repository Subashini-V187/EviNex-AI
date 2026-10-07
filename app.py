"""EviNex AI: ask questions about document contents."""

from __future__ import annotations

import hashlib
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
    retrieve_relevant_content,
)

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024
PREVIEW_CHARACTER_LIMIT = 5000

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


def make_result(question, answer, evidence, error, document_hash):
    return {
        "question": question,
        "answer": answer,
        "evidence": evidence,
        "error": error,
        "document_hash": document_hash,
    }


def friendly_error(exc):
    text = str(exc)

    if "GEMINI_API_KEY is not available" in text:
        return (
            "The Gemini API key is not available to the app. "
            "Check that GEMINI_API_KEY is saved in Streamlit Secrets."
        )

    return f"Gemini could not complete the request: {text}"


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
    "Answers are based only on the document."
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

                    # Order: client, image_bytes, mime_type, question
                    answer = generate_image_answer(
                        client,
                        file_bytes,
                        mime_type,
                        question.strip(),
                    )

                st.session_state["answer_result"] = make_result(
                    question.strip(),
                    answer["answer"],
                    answer["evidence"],
                    None,
                    document_hash,
                )

            except Exception as exc:
                st.session_state["answer_result"] = make_result(
                    question.strip(),
                    None,
                    [],
                    friendly_error(exc),
                    document_hash,
                )

    result = st.session_state.get("answer_result")

    if result and result.get("document_hash") == document_hash:
        st.divider()
        st.subheader("AI Answer")

        if result.get("error"):
            st.error(result["error"])
        else:
            st.write(result["answer"])

            evidence = result.get("evidence", [])

            if evidence:
                st.subheader("Evidence")
                for item in evidence:
                    st.write(f'"{item["quote"]}"')
                    st.caption(
                        f"Source: {filename} · Page {item['page_number']}"
                    )

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
# SCANNED PDF DETECTION
# ============================================================

is_scanned_pdf = extension == ".pdf" and not bool(all_text)

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
        placeholder="For example: What conclusion does the report reach?",
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

                # ---------------- SCANNED PDF ----------------
                if is_scanned_pdf:

                    # Order: client, pdf_bytes, question
                    answer = generate_scanned_pdf_answer(
                        client,
                        file_bytes,
                        question.strip(),
                    )

                    answer_result = make_result(
                        question.strip(),
                        answer["answer"],
                        answer["evidence"],
                        None,
                        document_hash,
                    )

                # ---------------- NORMAL DOCUMENT ----------------
                else:

                    document_chunks = st.session_state.get("document_chunks")

                    if document_chunks is None:
                        document_chunks = chunk_document_pages(pages)

                    # Embed once per document
                    if not st.session_state.get("embeddings_ready"):

                        # Order: client, chunks
                        document_chunks = embed_document_chunks(
                            client,
                            document_chunks,
                        )

                        st.session_state["document_chunks"] = document_chunks
                        st.session_state["embeddings_ready"] = True

                    # Order: client, chunks, question
                    relevant_chunks = retrieve_relevant_content(
                        client,
                        document_chunks,
                        question.strip(),
                        top_k=5,
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
                        # Order: client, question, retrieved_content
                        answer = generate_grounded_answer(
                            client,
                            question.strip(),
                            relevant_chunks,
                        )

                        answer_result = make_result(
                            question.strip(),
                            answer["answer"],
                            answer["evidence"],
                            None,
                            document_hash,
                        )

                st.session_state["answer_result"] = answer_result

        except Exception as exc:
            st.session_state["answer_result"] = make_result(
                question.strip(),
                None,
                [],
                friendly_error(exc),
                document_hash,
            )


# ============================================================
# DISPLAY ANSWER
# ============================================================

result = st.session_state.get("answer_result")

if result and result.get("document_hash") == document_hash:

    st.divider()
    st.subheader("AI Answer")

    if result.get("error"):
        st.error(result["error"])

    else:
        st.write(result["answer"])

        st.subheader("Evidence")

        evidence = result.get("evidence", [])

        if evidence:
            for item in evidence:
                st.write(f'"{item["quote"]}"')

                page_number = item["page_number"]
                source_label = item.get("source_label") or f"Page {page_number}"

                st.caption(f"Source: {filename} · {source_label}")

        else:
            st.caption("No supporting excerpt was found in the document.")
