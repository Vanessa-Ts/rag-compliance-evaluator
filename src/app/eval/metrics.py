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

_JUDGE_PROMPT = """\
You are an evaluation judge. Given a question, retrieved context, and a generated answer,
determine whether the answer is faithful to the context (every claim can be verified from it).

Question: {question}

Context:
{context}

Answer: {answer}"""

_CONTEXT_RELEVANCE_TOOL = {
    "name": "context_relevance_verdict",
    "description": "Score how relevant the retrieved context passages are for answering the question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "score":     {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reasoning": {"type": "string", "maxLength": 300},
        },
        "required": ["score"],
    },
}

_FAITHFULNESS_TOOL = {
    "name": "faithfulness_verdict",
    "description": "Record whether the answer is faithful to the retrieved context.",
    "input_schema": {
        "type": "object",
        "properties": {
            "faithful":  {"type": "boolean"},
            "score":     {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reasoning": {"type": "string", "maxLength": 400},
        },
        "required": ["faithful", "score", "reasoning"],
    },
}


async def judge_faithfulness(
    question: str,
    context: str,
    answer: str,
    generator: object,
) -> tuple[bool, float, str | None]:
    """Return (faithful, score, reasoning) using the generator as an LLM judge."""
    from app.rag.llm import AnthropicCachingGenerator

    if isinstance(generator, AnthropicCachingGenerator):
        prompt = _JUDGE_PROMPT.format(question=question, context=context, answer=answer)
        try:
            response = await generator._client.messages.create(  # type: ignore[call-overload]
                model=generator.model,
                max_tokens=256,
                tools=[_FAITHFULNESS_TOOL],
                tool_choice={"type": "any"},
                messages=[{"role": "user", "content": prompt}],
            )
            tool_input = next(b for b in response.content if b.type == "tool_use").input
            faithful = bool(tool_input["faithful"])
            score = max(0.0, min(1.0, float(tool_input["score"])))
            reasoning = tool_input.get("reasoning")
            return faithful, score, reasoning
        except Exception:
            return False, 0.0, None

    from app.rag.interfaces import Generator

    if not isinstance(generator, Generator):
        return False, 0.0, None

    prompt = _FAITHFULNESS_PROMPT.format(question=question, context=context, answer=answer)
    try:
        raw = await generator.generate(prompt)
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        faithful = bool(parsed.get("faithful", False))
        score = float(parsed.get("score", 0.0))
        return faithful, max(0.0, min(1.0, score)), None
    except Exception:
        return False, 0.0, None


_CONTEXT_RELEVANCE_JUDGE_PROMPT = """\
You are an evaluation judge. Given a question and retrieved context passages, \
score how relevant the passages are for answering the question.
0.0 = completely irrelevant, 1.0 = perfectly targeted.

Question: {question}
Context: {context}"""

_CONTEXT_RELEVANCE_FALLBACK_PROMPT = """\
You are an evaluation judge. Given a question and retrieved context passages, \
score how relevant the passages are for answering the question.
0.0 = completely irrelevant, 1.0 = perfectly targeted.

Question: {question}
Context: {context}
Respond ONLY with JSON: {{"score": <0.0-1.0>}}"""


async def judge_context_relevance(
    question: str, context: str, generator: object
) -> float:
    """Returns a 0–1 score: how relevant the retrieved context is for answering question."""
    from app.rag.llm import AnthropicCachingGenerator

    if isinstance(generator, AnthropicCachingGenerator):
        prompt = _CONTEXT_RELEVANCE_JUDGE_PROMPT.format(question=question, context=context)
        try:
            response = await generator._client.messages.create(  # type: ignore[call-overload]
                model=generator.model,
                max_tokens=128,
                tools=[_CONTEXT_RELEVANCE_TOOL],
                tool_choice={"type": "any"},
                messages=[{"role": "user", "content": prompt}],
            )
            tool_input = next(b for b in response.content if b.type == "tool_use").input
            return max(0.0, min(1.0, float(tool_input["score"])))
        except Exception:
            return 0.0

    from app.rag.interfaces import Generator

    if not isinstance(generator, Generator):
        return 0.0

    prompt = _CONTEXT_RELEVANCE_FALLBACK_PROMPT.format(question=question, context=context)
    try:
        raw = await generator.generate(prompt)
        parsed = json.loads(raw.strip())
        score = float(parsed.get("score", 0.0))
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0
