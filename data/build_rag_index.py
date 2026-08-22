"""Builds a Chroma vector index over data/correspondence/*.txt for RAG.

Each correspondence file starts with a small header:

    Entity: <name> (customer|supplier)
    Date: <date>
    Title: <title>

    <body text>

The entity name is resolved against the customers/suppliers tables in
business_ops.db to attach entity_id as searchable metadata.

Run with: .venv/bin/python data/build_rag_index.py
Re-running rebuilds the index from scratch, so it's safe to repeat after
editing/adding correspondence files.
"""

import re
import sqlite3
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

DATA_DIR = Path(__file__).parent
DB_PATH = DATA_DIR / "business_ops.db"
CORRESPONDENCE_DIR = DATA_DIR / "correspondence"
CHROMA_DIR = DATA_DIR / "chroma_correspondence"
EMBEDDING_MODEL = "qwen3-embedding"

HEADER_RE = re.compile(
    r"Entity:\s*(?P<name>.+?)\s*\((?P<type>customer|supplier)\)\s*\n"
    r"Date:\s*(?P<date>.+?)\s*\n"
    r"Title:\s*(?P<title>.+?)\s*\n\n"
    r"(?P<body>.+)",
    re.DOTALL,
)


def resolve_entity_id(conn: sqlite3.Connection, entity_type: str, entity_name: str) -> int | None:
    table = "customers" if entity_type == "customer" else "suppliers"
    row = conn.execute(f"SELECT id FROM {table} WHERE name = ?", (entity_name,)).fetchone()
    return row[0] if row else None


def load_documents(conn: sqlite3.Connection) -> list[Document]:
    documents = []
    for path in sorted(CORRESPONDENCE_DIR.glob("*.txt")):
        match = HEADER_RE.match(path.read_text())
        if not match:
            print(f"Skipping {path.name}: doesn't match the expected header format")
            continue
        entity_type = match["type"]
        entity_name = match["name"]
        entity_id = resolve_entity_id(conn, entity_type, entity_name)
        if entity_id is None:
            print(f"Skipping {path.name}: no {entity_type} named '{entity_name}' in the DB")
            continue
        documents.append(
            Document(
                page_content=match["body"].strip(),
                metadata={
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "entity_name": entity_name,
                    "title": match["title"],
                    "date": match["date"],
                    "source_file": path.name,
                },
            )
        )
    return documents


def build_index():
    conn = sqlite3.connect(DB_PATH)
    documents = load_documents(conn)
    conn.close()

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"Indexed {len(documents)} documents into {CHROMA_DIR}")


if __name__ == "__main__":
    build_index()
