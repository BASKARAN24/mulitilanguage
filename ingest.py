"""
Ingest documents into the RAG system.

Usage:
    python ingest.py

This walks through every .txt file in sample_docs/, guesses the language
from the filename (e.g. "history_en.txt" -> "en"), chunks the text,
embeds each chunk with the multilingual model, and stores everything
in MySQL.

You can also call ingest_document() directly from app.py for file uploads.
"""

import os
import json
from db import get_conn, model

SAMPLE_DOCS_DIR = os.path.join(os.path.dirname(__file__), "sample_docs")


def chunk_text(text, chunk_size=150, overlap=30):
    """Split text into overlapping word chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def ingest_document(title, language, file_path, conn=None):
    """Read a text file, chunk it, embed each chunk, and store in MySQL."""
    own_conn = conn is None
    if own_conn:
        conn = get_conn()

    with open(file_path, encoding="utf-8") as f:
        text = f.read()

    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO documents (title, language, source_file) VALUES (%s, %s, %s)",
        (title, language, file_path),
    )
    doc_id = cursor.lastrowid

    chunks = chunk_text(text)
    for chunk in chunks:
        embedding = model.encode(chunk).tolist()
        cursor.execute(
            "INSERT INTO chunks (doc_id, chunk_text, language, embedding) VALUES (%s, %s, %s, %s)",
            (doc_id, chunk, language, json.dumps(embedding)),
        )

    conn.commit()
    cursor.close()
    if own_conn:
        conn.close()

    print(f"Ingested '{title}' ({language}): {len(chunks)} chunks")
    return doc_id, len(chunks)


def guess_language_from_filename(filename):
    """Expects filenames like 'topic_en.txt', 'topic_hi.txt', etc."""
    name = filename.rsplit(".", 1)[0]
    parts = name.split("_")
    lang_code = parts[-1].lower()
    valid = {"en", "hi", "ta", "fr", "de"}
    return lang_code if lang_code in valid else "en"


def ingest_all_sample_docs():
    if not os.path.isdir(SAMPLE_DOCS_DIR):
        print(f"No sample_docs folder found at {SAMPLE_DOCS_DIR}")
        return

    conn = get_conn()
    files = [f for f in os.listdir(SAMPLE_DOCS_DIR) if f.endswith(".txt")]

    if not files:
        print("No .txt files found in sample_docs/. Add some and re-run.")
        return

    for filename in files:
        file_path = os.path.join(SAMPLE_DOCS_DIR, filename)
        language = guess_language_from_filename(filename)
        title = filename.rsplit(".", 1)[0]
        ingest_document(title, language, file_path, conn=conn)

    conn.close()
    print("\nAll sample documents ingested.")


if __name__ == "__main__":
    ingest_all_sample_docs()
