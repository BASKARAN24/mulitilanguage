"""
Takes retrieved chunks + the user's question and asks Grok to produce
a final answer in the same language the user asked in.
"""

import os
from openai import OpenAI
from langdetect import detect, DetectorFactory
from dotenv import load_dotenv
from retrieve import retrieve_top_k

load_dotenv()
DetectorFactory.seed = 0  # makes langdetect deterministic

client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1",
)

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "fr": "French",
    "de": "German",
}


def answer_question(query, k=5, conn=None):
    try:
        detected_code = detect(query)
    except Exception:
        detected_code = "en"

    language_name = LANGUAGE_NAMES.get(detected_code, "English")

    top_chunks = retrieve_top_k(query, k=k, conn=conn)

    if not top_chunks:
        context = "(no documents have been ingested yet)"
    else:
        context = "\n\n".join(
            f"[{lang}] {text}" for _, text, lang in top_chunks
        )

    prompt = f"""You are a multilingual assistant. Use the context below, which may
contain passages in different languages, to answer the user's question.
Respond ONLY in {language_name}, even if the context is in other languages.
If the context does not contain the answer, say so honestly in {language_name}.

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="grok-4",
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "answer": response.choices[0].message.content,
        "detected_language": detected_code,
        "sources": [{"language": lang, "text": text[:150]} for _, text, lang in top_chunks],
    }


if __name__ == "__main__":
    q = input("Ask a question (any language): ")
    result = answer_question(q)
    print("\nDetected language:", result["detected_language"])
    print("\nAnswer:\n", result["answer"])
