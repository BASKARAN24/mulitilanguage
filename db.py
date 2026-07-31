"""
Shared helpers: MySQL connection + the multilingual embedding model.
Both ingest.py and retrieve.py import from here so the model is
defined in exactly one place.
"""

import os
import mysql.connector
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()  # reads the .env file into environment variables

# Loaded once and reused everywhere (loading it repeatedly is slow).
print("Loading multilingual embedding model... (first run downloads ~1GB)")
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
print("Model loaded.")


def get_conn():
    """Open a new MySQL connection using credentials from .env"""
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DATABASE", "rag_chatbot"),
    )


def init_schema():
    """Create the tables if they don't exist yet. Safe to run multiple times."""
    conn = get_conn()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id INT AUTO_INCREMENT PRIMARY KEY,
            title VARCHAR(255),
            language VARCHAR(10),
            source_file VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id INT AUTO_INCREMENT PRIMARY KEY,
            doc_id INT,
            chunk_text TEXT,
            language VARCHAR(10),
            embedding JSON,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()
    print("Schema ready (documents + chunks tables).")


if __name__ == "__main__":
    # Running "python db.py" directly sets up your database tables.
    init_schema()
