from pathlib import Path
from typing import Iterable
from pypdf import PdfReader

def read_file(path: Path) -> list[dict]:
    """Return page-aware text records from PDF, TXT or Markdown files."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [
            {"text": page.extract_text() or "", "page": i + 1}
            for i, page in enumerate(reader.pages)
        ]
    if suffix in {".txt", ".md"}:
        return [{"text": path.read_text(encoding="utf-8", errors="ignore"), "page": 1}]
    raise ValueError("Supported formats are PDF, TXT and Markdown.")

def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # Prefer a sentence/word boundary without creating tiny chunks.
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks

def documents_from_file(path: Path, chunk_size: int = 900, overlap: int = 150) -> list[dict]:
    records = []
    for page_record in read_file(path):
        for chunk_id, chunk in enumerate(split_text(page_record["text"], chunk_size, overlap), start=1):
            records.append({
                "text": chunk,
                "source": path.name,
                "page": page_record["page"],
                "chunk": chunk_id
            })
    return records

def documents_from_paths(paths: Iterable[Path], chunk_size: int = 900, overlap: int = 150) -> list[dict]:
    result = []
    for path in paths:
        result.extend(documents_from_file(Path(path), chunk_size, overlap))
    return result
