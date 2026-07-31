"""
Flask API for the multilingual RAG chatbot.

Routes:
    POST /upload  - upload a document (multipart form: file, title, language)
    POST /ask     - ask a question (json body: {"question": "..."})

Run with:
    python app.py
Then it's available at http://127.0.0.1:5000
"""

import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from db import init_schema
from ingest import ingest_document
from generate import answer_question

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploaded_docs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

VALID_LANGUAGES = {"en", "hi", "ta", "fr", "de"}


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Multilingual RAG chatbot API is running.",
        "endpoints": ["/upload (POST)", "/ask (POST)"],
    })


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    title = request.form.get("title", file.filename)
    language = request.form.get("language", "en").lower()

    if language not in VALID_LANGUAGES:
        return jsonify({"error": f"language must be one of {sorted(VALID_LANGUAGES)}"}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    doc_id, num_chunks = ingest_document(title, language, save_path)

    return jsonify({
        "message": "Document ingested successfully",
        "doc_id": doc_id,
        "chunks_created": num_chunks,
    })


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    result = answer_question(question)
    return jsonify(result)


if __name__ == "__main__":
    init_schema()  # make sure tables exist before serving requests
    app.run(debug=True)
