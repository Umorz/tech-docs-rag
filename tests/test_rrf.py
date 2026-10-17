"""Reciprocal Rank Fusion: вклад канала = weight / (k + rank)."""
import pytest

from src.index.hybrid import _rrf_component


def test_rrf_formula():
    assert _rrf_component(1, k=60, weight=1.0) == pytest.approx(1 / 61)
    assert _rrf_component(3, k=60, weight=2.0) == pytest.approx(2 / 63)


@pytest.mark.parametrize("rank", [None, 0, -1])
def test_rrf_missing_rank_gives_nothing(rank):
    assert _rrf_component(rank, k=60, weight=1.0) == 0.0


def test_rrf_higher_rank_contributes_more():
    assert _rrf_component(1, k=60, weight=1.0) > _rrf_component(2, k=60, weight=1.0)
