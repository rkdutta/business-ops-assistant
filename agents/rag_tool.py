"""RAG tool for searching fake customer/supplier correspondence.

Separate from the MCP sqlite tools since this is a local vector search
concern, not a database-server one. Import search_correspondence and add it
to whichever agent(s)/subagent(s) need it.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings

CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_correspondence"
EMBEDDING_MODEL = "qwen3-embedding"

_store = Chroma(
    persist_directory=str(CHROMA_DIR),
    embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
)


@tool
def search_correspondence(query: str) -> str:
    """Search past correspondence, contracts, and notes about customers and
    suppliers (e.g. agreed payment terms, delivery arrangements, discounts).
    Not for structured data like invoice amounts or order status — use the
    SQL tools for that."""
    results = _store.similarity_search(query, k=3)
    if not results:
        return f"No correspondence found matching '{query}'."
    return "\n\n".join(
        f"[{r.metadata['entity_type']}: {r.metadata['entity_name']}] "
        f"{r.metadata['title']} ({r.metadata['date']})\n{r.page_content}"
        for r in results
    )
