"""Retrieve top KB chunks and produce grounded mini-answers for sample questions."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB = Path(__file__).parent / "data" / "kb.jsonl"
OUT = Path(__file__).parent / "outputs"

QUESTIONS = [
    "How do I detect DNS tunneling?",
    "What features indicate DGA domains?",
    "How should SOC analysts prioritize alerts?",
    "What is a sign of C2 beaconing?",
]


def load_kb(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def answer(question: str, docs: list[dict], vec, X) -> dict:
    # retrieve top chunks, then just stitch them — no external LLM needed for the demo
    sims = cosine_similarity(vec.transform([question]), X).ravel()
    idx = sims.argsort()[::-1][:2]
    contexts = [docs[i] for i in idx]
    grounded = " ".join(c["text"] for c in contexts)
    return {
        "question": question,
        "answer": f"Based on the knowledge base: {grounded}",
        "sources": [{"id": c["id"], "title": c["title"], "score": round(float(sims[i]), 4)} for c, i in zip(contexts, idx)],
    }


def main() -> None:
    if not KB.exists():
        raise SystemExit("Run build_kb.py first.")
    OUT.mkdir(parents=True, exist_ok=True)
    docs = load_kb(KB)
    corpus = [f"{d['title']}. {d['text']}" for d in docs]
    vec = TfidfVectorizer(stop_words="english")
    X = vec.fit_transform(corpus)
    qa = [answer(q, docs, vec, X) for q in QUESTIONS]
    (OUT / "sample_qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    stats = {"n_docs": len(docs), "n_questions": len(QUESTIONS), "vocab_size": int(len(vec.vocabulary_))}
    (OUT / "kb_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(json.dumps(qa[0], indent=2))


if __name__ == "__main__":
    main()
