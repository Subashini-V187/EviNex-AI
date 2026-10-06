"""Small tests for PDF table extraction and table-aware retrieval."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

import pymupdf

from pdf_processor import (
    chunk_document_pages,
    extract_pdf_pages,
    generate_grounded_answer,
    retrieve_relevant_content,
)


class FakeGeminiModels:
    def __init__(self) -> None:
        self.last_prompt = ""

    def embed_content(self, **kwargs):
        return SimpleNamespace(
            embeddings=[SimpleNamespace(values=[-1.0, 0.0])]
        )

    def generate_content(self, **kwargs):
        self.last_prompt = kwargs["contents"]
        return SimpleNamespace(
            text=json.dumps(
                {
                    "answer": "₹19 Crore",
                    "evidence": [
                        {
                            "page_number": 2,
                            "quote": "Revenue: ₹19 Crore",
                        }
                    ],
                }
            )
        )


class TableUnderstandingTests(unittest.TestCase):
    def test_extracts_table_with_its_original_page_number(self) -> None:
        rows = [
            ["Year", "Revenue"],
            ["2023", "INR 10 Crore"],
            ["2024", "INR 14 Crore"],
            ["2025", "INR 19 Crore"],
        ]
        document = pymupdf.open()
        document.new_page()
        page = document.new_page()
        x0, y0 = 72, 72
        column_widths = [100, 180]
        row_height = 30
        x1 = x0 + sum(column_widths)
        y1 = y0 + len(rows) * row_height

        for x in (x0, x0 + column_widths[0], x1):
            page.draw_line((x, y0), (x, y1), color=(0, 0, 0), width=0.7)
        for row_index in range(len(rows) + 1):
            y = y0 + row_index * row_height
            page.draw_line((x0, y), (x1, y), color=(0, 0, 0), width=0.7)
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row):
                x = x0 + sum(column_widths[:column_index]) + 8
                y = y0 + row_index * row_height + 19
                page.insert_text((x, y), value)

        pdf_bytes = document.tobytes()
        document.close()

        extracted_pages = extract_pdf_pages(pdf_bytes)
        table = extracted_pages[1]["tables"][0]
        self.assertEqual(table["page_number"], 2)
        self.assertEqual(table["columns"], ["Year", "Revenue"])
        self.assertEqual(table["rows"][-1], ["2025", "INR 19 Crore"])

    def test_retrieves_and_verifies_the_2025_table_value(self) -> None:
        table = {
            "page_number": 2,
            "table_index": 1,
            "columns": ["Year", "Revenue"],
            "rows": [
                ["2023", "₹10 Crore"],
                ["2024", "₹14 Crore"],
                ["2025", "₹19 Crore"],
            ],
        }
        pages = [{"page_number": 2, "text": "", "tables": [table]}]
        chunks = chunk_document_pages(pages)
        table_chunks = [chunk for chunk in chunks if chunk["content_type"] == "table"]
        embeddings = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        fake_models = FakeGeminiModels()
        fake_client = SimpleNamespace(models=fake_models)

        retrieved = retrieve_relevant_content(
            pages,
            "What was the revenue in 2025?",
            top_k=1,
            chunks=table_chunks,
            document_embeddings=embeddings,
            client=fake_client,
        )
        self.assertEqual(retrieved[0]["table_data"]["Year"], "2025")
        self.assertEqual(retrieved[0]["page_number"], 2)

        result = generate_grounded_answer(
            "What was the revenue in 2025?",
            retrieved,
            client=fake_client,
        )
        self.assertEqual(result["answer"], "₹19 Crore")
        self.assertEqual(
            result["evidence"],
            [{"page_number": 2, "quote": "Revenue: ₹19 Crore"}],
        )
        self.assertIn("₹19 Crore", fake_models.last_prompt)
        self.assertNotIn("₹10 Crore", fake_models.last_prompt)

        bad_quote_client = SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **kwargs: SimpleNamespace(
                    text=json.dumps(
                        {
                            "answer": "₹99 Crore",
                            "evidence": [
                                {
                                    "page_number": 2,
                                    "quote": "Revenue: ₹99 Crore",
                                }
                            ],
                        }
                    )
                )
            )
        )
        rejected = generate_grounded_answer(
            "What was the revenue in 2025?",
            retrieved,
            client=bad_quote_client,
        )
        self.assertEqual(rejected["answer"], "Cannot determine from the document.")


if __name__ == "__main__":
    unittest.main()
