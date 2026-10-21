
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = ROOT / "data" / "policies"
EVAL_PATH = ROOT / "data" / "raw" / "policy_eval.csv"
OUTPUT_DIR = ROOT / "outputs"

CHUNK_SIZE = 40
CHUNK_OVERLAP = 10
TOP_K = 3


def tokenize(text):
    """Return whitespace-separated tokens."""
    return text.split()


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into fixed-size word chunks with overlap."""
    words = tokenize(text)
    if not words:
        return []

    step = chunk_size - overlap
    return [
        " ".join(words[start:start + chunk_size])
        for start in range(0, len(words), step)
    ]


def load_policy_chunks():
    chunks = []

    for path in sorted(POLICY_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for index, chunk in enumerate(chunk_text(text)):
            chunks.append({
                "source": path.name,
                "chunk_id": index,
                "text": chunk,
            })

    if not chunks:
        raise RuntimeError(f"No policy text files found in {POLICY_DIR}")

    return chunks


def retrieve(question, chunks, vectorizer, matrix, top_k=TOP_K):
    """Retrieve the most relevant policy chunks using TF-IDF."""
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, matrix).ravel()
    best_indices = scores.argsort()[::-1][:top_k]

    return [
        {
            **chunks[i],
            "score": float(scores[i]),
        }
        for i in best_indices
    ]


def load_evaluation():
    df = pd.read_csv(EVAL_PATH, keep_default_na=False)

    required = {"question"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Evaluation CSV is missing required column(s): {sorted(missing)}. "
            f"Available columns: {list(df.columns)}"
        )

    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_policy_chunks()
    evaluation = load_evaluation()

    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        stop_words="english",
    )
    matrix = vectorizer.fit_transform(
        [chunk["text"] for chunk in chunks]
    )

    print(f"Policy files: {len(list(POLICY_DIR.glob('*.txt')))}")
    print(f"Policy chunks: {len(chunks)}")
    print(f"Evaluation questions: {len(evaluation)}")
    print(f"Chunk size: {CHUNK_SIZE} words")
    print(f"Chunk overlap: {CHUNK_OVERLAP} words")
    print(f"Top-k retrieval: {TOP_K}")

    results = []

    for row_number, row in evaluation.iterrows():
        question = str(row["question"]).strip()
        retrieved = retrieve(question, chunks, vectorizer, matrix)

        results.append({
            "question_id": row.get("question_id", row_number + 1),
            "question": question,
            "retrieved_sources": json.dumps(
                [item["source"] for item in retrieved]
            ),
            "retrieved_chunks": json.dumps(
                [
                    {
                        "source": item["source"],
                        "chunk_id": item["chunk_id"],
                        "score": round(item["score"], 4),
                        "text": item["text"],
                    }
                    for item in retrieved
                ]
            ),
        })

    output_path = OUTPUT_DIR / "rag_retrieval_results.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)

    print(f"Saved retrieval results: {output_path}")
    print("\nSample retrievals:")
    for result in results[:3]:
        print(f"\nQ: {result['question']}")
        for item in json.loads(result["retrieved_chunks"]):
            print(
                f"  {item['source']} "
                f"(chunk {item['chunk_id']}, score {item['score']})"
            )
            print(f"  {item['text']}")


if __name__ == "__main__":
    main()
