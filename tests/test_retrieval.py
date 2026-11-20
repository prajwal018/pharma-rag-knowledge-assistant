from app.retrieval import Retriever

def test_retriever_finds_relevant_document():
    docs = [
        {"text": "Store the material between 2 and 8 degrees Celsius.", "source": "sop.pdf", "page": 1},
        {"text": "The report must be reviewed by a quality manager.", "source": "quality.pdf", "page": 2},
    ]
    results = Retriever(docs).search("material degrees Celsius", top_k=1)
    assert results
    assert results[0]["source"] == "sop.pdf"
