"""EviNex AI: ask questions about the contents of a PDF."""

from __future__ import annotations

import hashlib

import streamlit as st
from dotenv import load_dotenv

from pdf_processor import (
    FALLBACK_ANSWER,
    extract_pdf_pages,
    generate_grounded_answer,
    retrieve_relevant_content,
)


load_dotenv()

MAX_PDF_SIZE_BYTES = 20 * 1024 * 1024
PREVIEW_CHARACTER_LIMIT = 5000

st.set_page_config(page_title="EviNex AI", page_icon="📄", layout="centered")
st.title("EviNex AI")
st.write("Upload a PDF and ask a question. Answers are based only on the document.")

uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

if uploaded_file is None:
    st.info("Upload a PDF to see its details and ask a question.")
    st.stop()

pdf_bytes = uploaded_file.getvalue()
if len(pdf_bytes) > MAX_PDF_SIZE_BYTES:
    st.error("This PDF is larger than the 20 MB limit.")
    st.stop()

document_hash = hashlib.sha256(pdf_bytes).hexdigest()
if st.session_state.get("document_hash") != document_hash:
    try:
        pages = extract_pdf_pages(pdf_bytes)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.session_state["document_hash"] = document_hash
    st.session_state["document_pages"] = pages
    st.session_state.pop("answer_result", None)

pages = st.session_state["document_pages"]
all_text = "\n\n".join(
    f"Page {page['page_number']}\n{page['text']}"
    for page in pages
    if page["text"]
)

st.subheader("Document information")
st.write(f"**Filename:** {uploaded_file.name}")
st.write(f"**Number of pages:** {len(pages)}")

with st.expander("Extracted text preview", expanded=False):
    if all_text:
        preview = all_text[:PREVIEW_CHARACTER_LIMIT]
        st.text_area(
            "Text extracted from the PDF",
            value=preview,
            height=220,
            disabled=True,
        )
        if len(all_text) > PREVIEW_CHARACTER_LIMIT:
            st.caption(
                f"Preview limited to the first {PREVIEW_CHARACTER_LIMIT:,} "
                "characters."
            )
    else:
        st.warning(
            "No selectable text was found. This may be a scanned PDF; "
            "OCR is not included in this version."
        )

if not all_text:
    st.stop()

with st.form("document_question_form"):
    question = st.text_input(
        "Ask a question about this document",
        placeholder="For example: What conclusion does the report reach?",
        max_chars=500,
    )
    submitted = st.form_submit_button("Ask")

if submitted:
    if not question.strip():
        st.warning("Enter a question to continue.")
    else:
        relevant_chunks = retrieve_relevant_content(pages, question.strip())
        if not relevant_chunks:
            st.session_state["answer_result"] = {
                "question": question.strip(),
                "answer": FALLBACK_ANSWER,
                "evidence": [],
                "error": None,
                "document_hash": document_hash,
            }
        else:
            try:
                answer = generate_grounded_answer(
                    question.strip(), relevant_chunks
                )
                st.session_state["answer_result"] = {
                    "question": question.strip(),
                    "answer": answer["answer"],
                    "evidence": answer["evidence"],
                    "error": None,
                    "document_hash": document_hash,
                }
            except RuntimeError:
                st.session_state["answer_result"] = {
                    "question": question.strip(),
                    "answer": None,
                    "evidence": [],
                    "error": (
                        "The Gemini API key is not available to the app. "
                        "Check that `GEMINI_API_KEY` is saved in Replit Secrets, "
                        "then restart the app."
                    ),
                    "document_hash": document_hash,
                }
            except Exception:
                st.session_state["answer_result"] = {
                    "question": question.strip(),
                    "answer": None,
                    "evidence": [],
                    "error": (
                        "Gemini could not complete the request. Check the "
                        "Gemini API key and model access."
                    ),
                    "document_hash": document_hash,
                }

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
                st.caption(
                    f"Source: {uploaded_file.name} · Page {item['page_number']}"
                )
        else:
            st.caption("No supporting excerpt was found in the retrieved text.")
