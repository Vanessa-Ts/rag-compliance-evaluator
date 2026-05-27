"""Tests for eval metrics — pure functions, no network."""

from __future__ import annotations

import pytest

from app.eval.metrics import hit_at_k, p95, precision_at_k


def test_precision_at_k_perfect() -> None:
    assert precision_at_k(["a", "b", "c"], ["a", "b"], k=3) == pytest.approx(2 / 3)


def test_precision_at_k_zero() -> None:
    assert precision_at_k(["x", "y"], ["a", "b"], k=2) == 0.0


def test_precision_at_k_all_hit() -> None:
    assert precision_at_k(["a", "b"], ["a", "b", "c"], k=2) == 1.0


def test_precision_at_k_zero_k() -> None:
    assert precision_at_k(["a"], ["a"], k=0) == 0.0


def test_hit_at_k_true() -> None:
    assert hit_at_k(["x", "a", "y"], ["a"], k=3) is True


def test_hit_at_k_false() -> None:
    assert hit_at_k(["x", "y"], ["a"], k=2) is False


def test_hit_at_k_only_top_k() -> None:
    # "a" is at index 2 but k=2, so it's excluded
    assert hit_at_k(["x", "y", "a"], ["a"], k=2) is False


def test_p95_single() -> None:
    assert p95([100.0]) == 100.0


def test_p95_empty() -> None:
    assert p95([]) == 0.0


def test_p95_sorted() -> None:
    values = list(range(1, 101))  # 1..100
    result = p95([float(v) for v in values])
    assert result <= 100.0
    assert result >= 95.0
