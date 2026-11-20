from app.ingestion import split_text

def test_split_text_returns_chunks():
    text = "This is a sentence. " * 150
    chunks = split_text(text, chunk_size=120, overlap=20)
    assert len(chunks) > 1
    assert all(chunks)

def test_empty_text():
    assert split_text("   ") == []

def test_invalid_overlap():
    try:
        split_text("hello", chunk_size=10, overlap=10)
    except ValueError:
        assert True
    else:
        assert False
