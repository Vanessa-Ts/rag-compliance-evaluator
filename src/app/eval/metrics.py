"""Pure evaluation metrics and LLM faithfulness judge."""

from __future__ import annotations

import json


def precision_at_k(retrieved_doc_ids: list[str], expected: list[str], k: int) -> float:
    """Fraction of top-k retrieved docs that are in the expected set."""
    if k == 0:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    expected_set = set(expected)
    hits = sum(1 for doc_id in top_k if doc_id in expected_set)
    return hits / k


def hit_at_k(retrieved_doc_ids: list[str], expected: list[str], k: int) -> bool:
    """True if at least one of the top-k retrieved docs is in the expected set."""
    expected_set = set(expected)
    return any(doc_id in expected_set for doc_id in retrieved_doc_ids[:k])


def p95(values: list[float]) -> float:
    """95th percentile of a list of floats."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = max(0, int(len(sorted_vals) * 0.95) - 1)
    return sorted_vals[idx]


_FAITHFULNESS_PROMPT = """\
You are an evaluation judge. Given a question, retrieved context, and a generated answer,
determine whether the answer is faithful to the context (every claim can be verified from it).

Respond with ONLY valid JSON in this exact format:
{{"faithful": true/false, "score": 0.0-1.0, "reasoning": "brief explanation"}}

Question: {question}

Context:
{context}

Answer: {answer}"""


async def judge_faithfulness(
    question: str,
    context: str,
    answer: str,
    generator: object,
) -> tuple[bool, float]:
    """Return (faithful, score) using the generator as an LLM judge (temperature=0)."""
    from app.rag.interfaces import Generator

    if not isinstance(generator, Generator):
        return False, 0.0

    prompt = _FAITHFULNESS_PROMPT.format(
        question=question, context=context, answer=answer
    )
    try:
        raw = await generator.generate(prompt)
        # Strip markdown code fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        faithful = bool(parsed.get("faithful", False))
        score = float(parsed.get("score", 0.0))
        return faithful, max(0.0, min(1.0, score))
    except Exception:
        return False, 0.0
