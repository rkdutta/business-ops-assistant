"""Exercises the real RAG pipeline: real Ollama embeddings
(qwen3-embedding) against the actual persisted Chroma index at
data/chroma_correspondence — not a fixture, since this is specifically
testing whether semantic search over the real correspondence data works,
which a fake store can't stand in for. Read-only, so no DB isolation
fixture is needed."""

from agents.rag_tool import search_correspondence


def test_semantic_search_finds_relevant_correspondence(require_ollama):
    result = search_correspondence.invoke(
        {"query": "payment plan for an overdue balance at a cafe"}
    )
    assert "Blue Fern Cafe" in result
    assert "installment" in result.lower()


def test_semantic_search_reports_no_match_gracefully(require_ollama):
    result = search_correspondence.invoke({"query": "xyz completely unrelated nonsense query"})
    # A vector store always returns its k-nearest neighbors regardless of
    # relevance, so this can't assert "no results" — just that it doesn't
    # error and returns the expected shape either way.
    assert isinstance(result, str) and result
