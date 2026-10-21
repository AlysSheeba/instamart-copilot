import json
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import mock_llm

POLICY_DIR = ROOT / "data" / "policies"
EVAL_PATH = ROOT / "data" / "raw" / "policy_eval.csv"
OUTPUT_DIR = ROOT / "outputs"

CHUNK_SIZE = 40
TOP_K = 3


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=10):
    words = text.split()
    if not words:
        return []
    step = chunk_size - overlap
    return [
        " ".join(words[start:start + chunk_size])
        for start in range(0, len(words), step)
    ]


def load_policy_chunks(overlap):
    chunks = []
    for path in sorted(POLICY_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for index, chunk in enumerate(chunk_text(text, overlap=overlap)):
            chunks.append({
                "source": path.name,
                "chunk_id": index,
                "text": chunk,
            })
    if not chunks:
        raise RuntimeError(f"No policy files found in {POLICY_DIR}")
    return chunks


def retrieve(question, chunks, vectorizer, matrix):
    query_vector = vectorizer.transform([question])
    scores = cosine_similarity(query_vector, matrix).ravel()
    best_indices = scores.argsort()[::-1][:TOP_K]
    return [
        {**chunks[i], "score": float(scores[i])}
        for i in best_indices
    ]


def answer_question(question, retrieved):
    context = "\n".join(
        f"[{item['source']}] {item['text']}"
        for item in retrieved
    )
    messages = [
        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT. Answer using only the supplied policy "
                "context. Do not invent rules. If the answer is not present, "
                "say that the policy documents do not specify it."
            ),
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\nQUESTION: {question}",
        },
    ]
    response = mock_llm.chat(messages)
    return (
        response.choices[0].message.content,
        response.usage.total_tokens,
    )


def evaluate(overlap, evaluation):
    chunks = load_policy_chunks(overlap=overlap)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        stop_words="english",
    )
    matrix = vectorizer.fit_transform([item["text"] for item in chunks])

    results = []
    for row_number, row in evaluation.iterrows():
        question = str(row["question"]).strip()
        expected_doc = str(row["expected_doc"]).strip()
        expected_fact = str(row["expected_fact"]).strip()

        retrieved = retrieve(question, chunks, vectorizer, matrix)
        answer, tokens = answer_question(question, retrieved)

        sources = [item["source"] for item in retrieved]
        source_hit = expected_doc in sources
        fact_hit = expected_fact.lower() in answer.lower()

        results.append({
            "question_id": row_number + 1,
            "question": question,
            "expected_doc": expected_doc,
            "expected_fact": expected_fact,
            "answer": answer,
            "retrieved_sources": json.dumps(sources),
            "retrieved_chunks": json.dumps([
                {
                    "source": item["source"],
                    "chunk_id": item["chunk_id"],
                    "score": round(item["score"], 4),
                    "text": item["text"],
                }
                for item in retrieved
            ]),
            "source_hit": source_hit,
            "fact_hit": fact_hit,
            "tokens_used": tokens,
            "overlap_words": overlap,
        })

    return chunks, results


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    evaluation = pd.read_csv(EVAL_PATH, keep_default_na=False)

    required = {"question", "expected_doc", "expected_fact"}
    missing = required - set(evaluation.columns)
    if missing:
        raise ValueError(f"Missing evaluation columns: {sorted(missing)}")

    mock_llm.reset()
    chunks_overlap, results_overlap = evaluate(10, evaluation)

    mock_llm.reset()
    chunks_no_overlap, results_no_overlap = evaluate(0, evaluation)

    pd.DataFrame(results_overlap).to_csv(
        OUTPUT_DIR / "rag_answers_overlap10.csv", index=False
    )
    pd.DataFrame(results_no_overlap).to_csv(
        OUTPUT_DIR / "rag_answers_overlap0.csv", index=False
    )

    summary = []
    for label, chunks, results in [
        ("overlap_10", chunks_overlap, results_overlap),
        ("overlap_0", chunks_no_overlap, results_no_overlap),
    ]:
        df = pd.DataFrame(results)
        summary.append({
            "configuration": label,
            "chunk_count": len(chunks),
            "questions": len(df),
            "source_hits": int(df["source_hit"].sum()),
            "source_accuracy_pct": round(df["source_hit"].mean() * 100, 2),
            "fact_hits": int(df["fact_hit"].sum()),
            "fact_accuracy_pct": round(df["fact_hit"].mean() * 100, 2),
            "total_tokens": int(df["tokens_used"].sum()),
            "mean_tokens_per_question": round(df["tokens_used"].mean(), 2),
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUTPUT_DIR / "rag_evaluation_summary.csv", index=False)

    print("\nRAG evaluation summary")
    print(summary_df.to_string(index=False))
    print("\nSample answers (10-word overlap):")
    for row in results_overlap[:3]:
        print(f"\nQuestion: {row['question']}")
        print(f"Answer: {row['answer']}")
        print(f"Expected fact: {row['expected_fact']}")
        print(f"Source hit: {row['source_hit']}; fact hit: {row['fact_hit']}")

    print("\nSaved:")
    print("outputs/rag_answers_overlap10.csv")
    print("outputs/rag_answers_overlap0.csv")
    print("outputs/rag_evaluation_summary.csv")


if __name__ == "__main__":
    main()
