"""Utility for extracting text from PDF or TXT files.

Uses PyMuPDF (imported as `fitz`) for PDF extraction. The import is performed lazily so that the rest of the application works even if PyMuPDF cannot be installed – TXT uploads and pasted text remain functional.
"""
import os
from typing import Optional

# Try to import PyMuPDF. If unavailable, keep a placeholder and handle gracefully.
try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except Exception:  # noqa: BLE001
    _HAS_FITZ = False

def extract_text_from_file(filepath: str) -> str:
    """Extract plain text from a `.txt` or `.pdf` file.

    Args:
        filepath: Absolute path to the uploaded file.
    Returns:
        The extracted raw text.
    Raises:
        ValueError: If the file extension is unsupported.
        RuntimeError: If PDF extraction fails (e.g., PyMuPDF not installed).
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".txt":
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    if ext == ".pdf":
        if not _HAS_FITZ:
            raise RuntimeError(
                "PDF extraction requires PyMuPDF (package 'PyMuPDF'). "
                "It is not available in the current environment."
            )
        try:
            doc = fitz.open(filepath)
            texts = [page.get_text() for page in doc]
            return "\n".join(texts)
        except Exception as exc:
            raise RuntimeError(f"Failed to extract text from PDF: {exc}") from exc
    raise ValueError(f"Unsupported file extension: {ext}")
