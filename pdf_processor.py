import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile

import pandas as pd
import pymupdf
from docx import Document
from google import genai
from google.genai import types
from pptx import Presentation


# ============================================================
# CONSTANTS
# ============================================================

FALLBACK_ANSWER = "Cannot determine from the document."
GENERATION_MODEL = "gemini-3.8-flash"
EMBEDDING_MODEL = "gemini-embedding-001"


# ============================================================
# BASIC HELPERS
# ============================================================

def _normalized_text(text):
    if text is None:
        return ""

    text = str(text)
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def create_gemini_client(api_key=None):
    api_key = api_key or os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not available. "
            "Add GEMINI_API_KEY to Streamlit Secrets."
        )

    return genai.Client(api_key=api_key)


def _busy_error(exc):
    """Return a friendly RuntimeError for 503 errors, else None."""
    if "503" in str(exc):
        return RuntimeError(
            "Gemini is temporarily busy. "
            "Please try again in a few seconds."
        )
    return None


# ============================================================
# PDF EXTRACTION
# ============================================================

def _extract_page_tables(page):
    tables = []

    try:
        finder = page.find_tables(strategy="lines")
        found_tables = getattr(finder, "tables", [])

        if not found_tables:
            finder = page.find_tables(strategy="text")
            found_tables = getattr(finder, "tables", [])

        for table_index, table in enumerate(found_tables, start=1):
            try:
                data = table.extract()

                if not data:
                    continue

                cleaned_rows = []

                for row in data:
                    cleaned_row = [_normalized_text(cell) for cell in row]

                    if any(cleaned_row):
                        cleaned_rows.append(cleaned_row)

                if cleaned_rows:
                    tables.append(
                        {
                            "table_index": table_index,
                            "data": cleaned_rows,
                        }
                    )

            except Exception:
                continue

    except Exception:
        pass

    return tables


def extract_pdf_pages(file_bytes):
    pages = []

    document = pymupdf.open(stream=file_bytes, filetype="pdf")

    try:
        for page_number, page in enumerate(document, start=1):
            text = _normalized_text(page.get_text("text"))
            tables = _extract_page_tables(page)

            pages.append(
                {
                    "page_number": page_number,
                    "text": text,
                    "tables": tables,
                    "content_type": "pdf",
                    "source_label": "PDF",
                }
            )

    finally:
        document.close()

    return pages


# ============================================================
# DOCX EXTRACTION
# ============================================================

def extract_docx_document(file_bytes):
    document = Document(io.BytesIO(file_bytes))

    parts = []

    for paragraph in document.paragraphs:
        text = _normalized_text(paragraph.text)

        if text:
            parts.append(text)

    for table_index, table in enumerate(document.tables, start=1):
        parts.append(f"Table {table_index}:")

        for row in table.rows:
            values = [_normalized_text(cell.text) for cell in row.cells]

            if any(values):
                parts.append(" | ".join(values))

    return [
        {
            "page_number": 1,
            "text": "\n".join(parts),
            "tables": [],
            "content_type": "docx",
            "source_label": "DOCX",
        }
    ]


# ============================================================
# LEGACY DOC EXTRACTION
# ============================================================

def extract_doc_document(file_bytes):
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        result = subprocess.run(
            ["antiword", temp_path],
            capture_output=True,
            text=True,
            timeout=60,
        )

        text = _normalized_text(result.stdout)

        if not text:
            raise RuntimeError("Could not extract text from the DOC file.")

        return [
            {
                "page_number": 1,
                "text": text,
                "tables": [],
                "content_type": "doc",
                "source_label": "DOC",
            }
        ]

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# TXT / MD
# ============================================================

def extract_text_document(file_bytes, extension):
    text = file_bytes.decode("utf-8", errors="replace")

    return [
        {
            "page_number": 1,
            "text": _normalized_text(text),
            "tables": [],
            "content_type": extension,
            "source_label": extension.upper(),
        }
    ]


# ============================================================
# CSV
# ============================================================

def extract_csv_document(file_bytes):
    dataframe = pd.read_csv(io.BytesIO(file_bytes))

    lines = []

    lines.append(
        "Columns: " + ", ".join(str(column) for column in dataframe.columns)
    )

    for row_number, row in dataframe.iterrows():
        values = [f"{column}={row[column]}" for column in dataframe.columns]
        lines.append(f"Row {row_number + 1}: " + " | ".join(values))

    return [
        {
            "page_number": 1,
            "text": "\n".join(lines),
            "tables": [],
            "content_type": "csv",
            "source_label": "CSV",
        }
    ]


# ============================================================
# EXCEL
# ============================================================

def _extract_excel_workbook(file_bytes, engine):
    excel = pd.ExcelFile(io.BytesIO(file_bytes), engine=engine)

    pages = []

    for sheet_number, sheet_name in enumerate(excel.sheet_names, start=1):
        dataframe = pd.read_excel(excel, sheet_name=sheet_name)

        lines = [f"Sheet: {sheet_name}"]

        if len(dataframe.columns) > 0:
            lines.append(
                "Columns: "
                + ", ".join(str(column) for column in dataframe.columns)
            )

        for row_number, row in dataframe.iterrows():
            values = [f"{column}={row[column]}" for column in dataframe.columns]
            lines.append(f"Row {row_number + 1}: " + " | ".join(values))

        pages.append(
            {
                "page_number": sheet_number,
                "text": "\n".join(lines),
                "tables": [],
                "content_type": "excel",
                "source_label": f"Excel sheet: {sheet_name}",
            }
        )

    return pages


def extract_xlsx_document(file_bytes):
    return _extract_excel_workbook(file_bytes, "openpyxl")


def extract_xls_document(file_bytes):
    return _extract_excel_workbook(file_bytes, "xlrd")


# ============================================================
# PPTX EXTRACTION
# ============================================================

def extract_pptx_document(file_bytes):
    presentation = Presentation(io.BytesIO(file_bytes))

    pages = []

    for slide_number, slide in enumerate(presentation.slides, start=1):
        texts = []

        for shape in slide.shapes:

            if hasattr(shape, "text"):
                text = _normalized_text(shape.text)

                if text:
                    texts.append(text)

            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    values = [_normalized_text(cell.text) for cell in row.cells]

                    if any(values):
                        texts.append(" | ".join(values))

        pages.append(
            {
                "page_number": slide_number,
                "text": "\n".join(texts),
                "tables": [],
                "content_type": "pptx",
                "source_label": f"PowerPoint slide {slide_number}",
            }
        )

    return pages


# ============================================================
# LEGACY PPT EXTRACTION
# ============================================================

def extract_ppt_document(file_bytes):
    temp_dir = tempfile.mkdtemp()

    ppt_path = os.path.join(temp_dir, "input.ppt")
    converted_path = os.path.join(temp_dir, "input.pptx")

    try:
        with open(ppt_path, "wb") as file:
            file.write(file_bytes)

        result = subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to",
                "pptx",
                "--outdir",
                temp_dir,
                ppt_path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if not os.path.exists(converted_path):
            raise RuntimeError(
                "Could not convert PPT to PPTX. "
                f"LibreOffice output: {result.stdout} {result.stderr}"
            )

        with open(converted_path, "rb") as file:
            pptx_bytes = file.read()

        return extract_pptx_document(pptx_bytes)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================
# JSON
# ============================================================

def extract_json_document(file_bytes):
    text = file_bytes.decode("utf-8", errors="replace")

    data = json.loads(text)

    formatted = json.dumps(data, indent=2, ensure_ascii=False)

    return [
        {
            "page_number": 1,
            "text": formatted,
            "tables": [],
            "content_type": "json",
            "source_label": "JSON",
        }
    ]


# ============================================================
# UNIVERSAL DOCUMENT EXTRACTION
# ============================================================

def extract_document(file_bytes, extension):
    """
    Extract content from supported file formats.

    extension can be "xlsx", ".xlsx" or a full filename like "data.xlsx".
    """

    extension = str(extension).lower().strip()
    extension = extension.lstrip(".")

    if "." in extension:
        extension = extension.rsplit(".", 1)[-1]

    if extension == "pdf":
        return extract_pdf_pages(file_bytes)

    if extension == "docx":
        return extract_docx_document(file_bytes)

    if extension == "doc":
        return extract_doc_document(file_bytes)

    if extension == "txt":
        return extract_text_document(file_bytes, "txt")

    if extension == "md":
        return extract_text_document(file_bytes, "md")

    if extension == "csv":
        return extract_csv_document(file_bytes)

    if extension == "xlsx":
        return extract_xlsx_document(file_bytes)

    if extension == "xls":
        return extract_xls_document(file_bytes)

    if extension == "pptx":
        return extract_pptx_document(file_bytes)

    if extension == "ppt":
        return extract_ppt_document(file_bytes)

    if extension == "json":
        return extract_json_document(file_bytes)

    raise ValueError(f"Unsupported document format: .{extension}")


# ============================================================
# CHUNKING
# ============================================================

def _make_chunks(
    text,
    page_number,
    content_type="text",
    source_label="Document",
    chunk_size=1200,
    overlap=200,
):
    text = _normalized_text(text)

    if not text:
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)

        chunks.append(
            {
                "page_number": page_number,
                "text": text[start:end],
                "content_type": content_type,
                "source_label": source_label,
            }
        )

        if end >= text_length:
            break

        start = max(end - overlap, start + 1)

    return chunks


def chunk_document_pages(pages):
    chunks = []

    for page in pages:
        page_number = page.get("page_number", 1)
        source_label = page.get("source_label", "Document")
        text = page.get("text", "")
        content_type = page.get("content_type", "text")

        chunks.extend(
            _make_chunks(
                text=text,
                page_number=page_number,
                content_type=content_type,
                source_label=source_label,
            )
        )

        for table in page.get("tables", []):
            table_index = table.get("table_index", 1)
            rows = table.get("data", [])

            for row_index, row in enumerate(rows, start=1):
                row_text = (
                    f"Table {table_index}, data row {row_index}: "
                    + " | ".join(_normalized_text(value) for value in row)
                )

                chunks.append(
                    {
                        "page_number": page_number,
                        "text": row_text,
                        "content_type": "table",
                        "source_label": source_label,
                        "table_index": table_index,
                        "row_index": row_index,
                        "table_data": row,
                    }
                )

    return chunks


# ============================================================
# GEMINI EMBEDDINGS
# ============================================================

def _embed_texts(client, texts, task_type):
    if not texts:
        return []

    embeddings = []

    for text in texts:
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type),
        )

        embeddings.append(result.embeddings[0].values)

    return embeddings


def embed_document_chunks(client, chunks):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = _embed_texts(client, texts, "RETRIEVAL_DOCUMENT")

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    return chunks


# ============================================================
# COSINE SIMILARITY
# ============================================================

def _cosine_similarity(vector_a, vector_b):
    if not vector_a or not vector_b:
        return 0.0

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_relevant_content(client, chunks, question, top_k=6):
    if not chunks:
        return []

    query_result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )

    query_embedding = query_result.embeddings[0].values

    scored_chunks = []

    for chunk in chunks:
        embedding = chunk.get("embedding")

        if not embedding:
            continue

        item = dict(chunk)
        item["score"] = _cosine_similarity(query_embedding, embedding)
        scored_chunks.append(item)

    scored_chunks.sort(key=lambda item: item["score"], reverse=True)

    return scored_chunks[:top_k]


# ============================================================
# GROUNDED ANSWER
# ============================================================

def generate_grounded_answer(client, question, retrieved_content):
    if not retrieved_content:
        return {"answer": FALLBACK_ANSWER, "evidence": []}

    excerpts = []

    for index, item in enumerate(retrieved_content, start=1):
        excerpts.append(
            f"""
SOURCE {index}
Page/Section: {item.get("page_number", 1)}
Source: {item.get("source_label", "Document")}

Content:
{item.get("text", "")}
"""
        )

    context = "\n".join(excerpts)

    prompt = f"""
You are EviNex AI,
an evidence-grounded multimodal
document intelligence assistant.

Answer the user's question ONLY using
the supplied document evidence.

Do NOT use outside knowledge.

If the answer cannot be determined
from the supplied evidence, return:

Cannot determine from the document.

USER QUESTION:
{question}

DOCUMENT EVIDENCE:
{context}

RULES:
1. Give a concise answer.
2. Do not invent information.
3. Every factual claim must be supported.
4. Evidence quotes must appear EXACTLY
   in the supplied content.
5. Include the page or section number.
"""

    schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "answer": types.Schema(type=types.Type.STRING),
            "evidence": types.Schema(
                type=types.Type.ARRAY,
                items=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "quote": types.Schema(type=types.Type.STRING),
                        "page_number": types.Schema(type=types.Type.INTEGER),
                    },
                    required=["quote", "page_number"],
                ),
            ),
        },
        required=["answer", "evidence"],
    )

    try:
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )

        result = json.loads(response.text)

        answer = result.get("answer", FALLBACK_ANSWER)
        evidence = result.get("evidence", [])

        verified_evidence = []

        for evidence_item in evidence:
            quote = _normalized_text(evidence_item.get("quote", ""))
            page_number = evidence_item.get("page_number")

            if not quote:
                continue

            for source in retrieved_content:
                if source.get("page_number") != page_number:
                    continue

                source_text = _normalized_text(source.get("text", ""))

                if quote in source_text:
                    verified_evidence.append(
                        {
                            "quote": quote,
                            "page_number": page_number,
                            "source_label": source.get("source_label", "Document"),
                            "score": source.get("score", 0),
                        }
                    )
                    break

        return {"answer": answer, "evidence": verified_evidence}

    except Exception as exc:
        busy = _busy_error(exc)

        if busy:
            raise busy

        raise


# ============================================================
# SCANNED PDF
# ============================================================

def generate_scanned_pdf_answer(client, pdf_bytes, question):
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(pdf_bytes)
            temp_path = temp_file.name

        uploaded_file = client.files.upload(file=temp_path)

        prompt = f"""
You are EviNex AI.

Analyze the uploaded PDF.

The PDF may contain:
- scanned pages
- images
- tables
- charts
- diagrams
- text

Answer the question using ONLY information
contained in the uploaded PDF.

QUESTION:
{question}

RULES:
1. Do not use outside knowledge.
2. If the answer is not present, say:
   Cannot determine from the document.
3. Give a concise answer.
4. Mention the relevant page number
   when possible.
"""

        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=[uploaded_file, prompt],
            config=types.GenerateContentConfig(temperature=0),
        )

        return {"answer": response.text, "evidence": []}

    except Exception as exc:
        busy = _busy_error(exc)

        if busy:
            raise busy

        raise

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# IMAGE QUESTION ANSWERING
# ============================================================

def generate_image_answer(client, image_bytes, mime_type, question):
    prompt = f"""
You are EviNex AI,
an evidence-grounded multimodal
document intelligence assistant.

Analyze the supplied image.

Answer ONLY using information visible
in the image.

QUESTION:
{question}

RULES:
1. Do not use outside knowledge.
2. If the answer cannot be determined,
   say:
   Cannot determine from the document.
3. Be concise.
4. Carefully inspect:
   - text
   - tables
   - charts
   - labels
   - numbers
"""

    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    try:
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=[image_part, prompt],
            config=types.GenerateContentConfig(temperature=0),
        )

        return {"answer": response.text, "evidence": []}

    except Exception as exc:
        busy = _busy_error(exc)

        if busy:
            raise busy

        raise
