import streamlit as st
from pathlib import Path
from app.store import load_documents, ingest_path, clear_documents
from app.retrieval import Retriever
from app.answering import answer_question
from app.config import DATA_DIR, TOP_K

st.set_page_config(page_title="Pharma Knowledge Assistant", page_icon="🧬", layout="wide")
st.title("🧬 Pharma Knowledge Assistant")
st.caption("A document-grounded RAG prototype for public and synthetic pharmaceutical reference material.")
st.warning("Research/demo only. Not for clinical decisions, diagnosis, treatment, dosage, or real patient data.")

with st.sidebar:
    st.header("Knowledge base")
    uploaded = st.file_uploader("Upload PDF, TXT or Markdown", type=["pdf", "txt", "md"])
    if st.button("Index document", type="primary", disabled=uploaded is None):
        target = DATA_DIR / Path(uploaded.name).name
        target.write_bytes(uploaded.getvalue())
        try:
            chunks = ingest_path(target)
            st.success(f"Indexed {chunks} chunks from {uploaded.name}.")
        except Exception as exc:
            st.error(f"Could not index file: {exc}")
    docs = load_documents()
    st.metric("Indexed chunks", len(docs))
    names = sorted({d["source"] for d in docs})
    st.write("**Documents**")
    for name in names:
        st.write(f"• {name}")
    if st.button("Clear knowledge base"):
        clear_documents()
        st.rerun()
    st.divider()
    st.caption("Tip: add the sample SOP from data/sample_docs to try the app.")

st.subheader("Ask your documents")
question = st.text_area("Enter a question", placeholder="What are the document's storage conditions?", height=100)
top_k = st.slider("Number of evidence passages", min_value=1, max_value=8, value=TOP_K)
if st.button("Generate answer", type="primary"):
    if not question.strip():
        st.info("Enter a question first.")
    elif not docs:
        st.info("Upload and index a document first.")
    else:
        with st.spinner("Searching documents..."):
            retriever = Retriever(docs)
            sources = retriever.search(question, top_k)
            response = answer_question(question, sources)
        st.markdown("### Answer")
        st.write(response)
        
        st.markdown("### Retrieved evidence")
        if not sources:
            st.info("No relevant passages found.")
        for i, item in enumerate(sources, start=1):
            with st.expander(
                f"[S{i}] {item['source']} | "
                f"Page {item['page']} | "
                f"Relevance: {item['score']:.3f}"
            ):
                st.write(item["text"])
                st.caption(
                    f"Evidence ID: S{i} | "
                    f"Original PDF page: {item['page']}"
                )
