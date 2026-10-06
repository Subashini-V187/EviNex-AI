---
name: EviNex AI first-version scope
description: User-defined boundaries for the initial EviNex AI release.
---

For EviNex AI's first working version, support PDF upload, PyMuPDF text and table extraction, page-aware chunking, Gemini embeddings and vector retrieval, document metadata and preview, question input, document-grounded answers, verified evidence, page/source references, and the exact unsupported-answer fallback. Do not add login, signup, payments, an admin dashboard, a user database, mobile support, or complex authentication.

**Why:** the user explicitly limited the first working version to document Q&A, then expanded its scope to include table understanding.

**How to apply:** keep unrelated account, payment, admin, database, mobile, and authentication features out of the initial app.

Use the official Google Gemini Python SDK (`google-genai`) for document answers and read `GEMINI_API_KEY` from Replit Secrets at runtime. Never ask the user for this key again or hardcode its value into a source file.

**Why:** the user explicitly selected Gemini, named the existing Replit Secret, and set this credential-handling rule.

**How to apply:** keep Gemini as the provider unless the user changes it; use the secret by environment variable only.
