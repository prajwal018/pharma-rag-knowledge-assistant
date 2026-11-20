from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from app.store import load_documents, ingest_path, clear_documents
from app.retrieval import Retriever
from app.answering import answer_question
from app.config import DATA_DIR, TOP_K

app = FastAPI(title="Pharma Knowledge Assistant API", version="1.0.0",
              description="Document-grounded RAG prototype using public or synthetic documents.")

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1500)
    top_k: int = Field(default=TOP_K, ge=1, le=10)

@app.get("/health")
def health():
    return {"status": "ok", "documents_indexed": len(load_documents())}

@app.post("/documents")
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Only PDF, TXT and MD files are supported.")
    safe_name = Path(file.filename).name
    target = DATA_DIR / safe_name
    content = await file.read()
    if len(content) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File exceeds 15 MB limit.")
    target.write_bytes(content)
    try:
        count = ingest_path(target)
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not process file: {exc}")
    return {"filename": safe_name, "chunks_indexed": count}

@app.post("/ask")
def ask(request: AskRequest):
    docs = load_documents()
    if not docs:
        raise HTTPException(status_code=400, detail="Upload a document before asking questions.")
    retriever = Retriever(docs)
    sources = retriever.search(request.question, request.top_k)
    return {"answer": answer_question(request.question, sources), "sources": sources}

@app.delete("/documents")
def delete_documents():
    clear_documents()
    return {"message": "Document index cleared."}
