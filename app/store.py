import json
from pathlib import Path
from app.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from app.ingestion import documents_from_file

INDEX_PATH = DATA_DIR / "documents.json"

def load_documents() -> list[dict]:
    if not INDEX_PATH.exists():
        return []
    try:
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

def save_documents(documents: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_PATH.write_text(json.dumps(documents, ensure_ascii=False, indent=2), encoding="utf-8")

def ingest_path(path: Path) -> int:
    new_docs = documents_from_file(path, CHUNK_SIZE, CHUNK_OVERLAP)
    current = load_documents()
    # Re-ingesting a filename replaces its previous chunks.
    current = [d for d in current if d["source"] != path.name]
    current.extend(new_docs)
    save_documents(current)
    return len(new_docs)

def clear_documents() -> None:
    save_documents([])
