"""Tests for domain pre-filter in embeddings._score_methods."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))

from chat.embeddings import _score_methods


def _make_unit(title, domain=None, embedding=None):
    return {
        "title": title,
        "_domain": domain,
        "embedding": embedding or [1.0, 0.0, 0.0],
    }


def test_domain_filter_narrows_pool():
    """When domain matches some units, only those units should be scored."""
    units = [
        _make_unit("A", "causal_inference", [1.0, 0.0, 0.0]),
        _make_unit("B", "bayesian", [0.9, 0.1, 0.0]),
        _make_unit("C", "causal_inference", [0.8, 0.2, 0.0]),
    ]
    query_emb = [1.0, 0.0, 0.0]
    scored = _score_methods(query_emb, units, domain="causal_inference")
    titles = [u["title"] for _, u in scored]
    assert "A" in titles
    assert "C" in titles
    assert "B" not in titles


def test_domain_filter_fallback_when_no_match():
    """When domain doesn't match any unit, fall back to full pool."""
    units = [
        _make_unit("A", "bayesian", [1.0, 0.0, 0.0]),
        _make_unit("B", "bayesian", [0.9, 0.1, 0.0]),
    ]
    query_emb = [1.0, 0.0, 0.0]
    scored = _score_methods(query_emb, units, domain="nonexistent_domain")
    assert len(scored) == 2  # falls back to full pool


def test_domain_filter_none_uses_all():
    """When domain is None, all units should be scored."""
    units = [
        _make_unit("A", "causal", [1.0, 0.0, 0.0]),
        _make_unit("B", "bayesian", [0.9, 0.1, 0.0]),
    ]
    query_emb = [1.0, 0.0, 0.0]
    scored = _score_methods(query_emb, units, domain=None)
    assert len(scored) == 2


def test_domain_filter_case_insensitive():
    """Domain matching should be case-insensitive."""
    units = [
        _make_unit("A", "Causal_Inference", [1.0, 0.0, 0.0]),
        _make_unit("B", "bayesian", [0.9, 0.1, 0.0]),
    ]
    query_emb = [1.0, 0.0, 0.0]
    scored = _score_methods(query_emb, units, domain="causal_inference")
    titles = [u["title"] for _, u in scored]
    assert "A" in titles
    assert "B" not in titles


def test_domain_filter_empty_string_uses_all():
    """When domain is empty string, all units should be scored."""
    units = [
        _make_unit("A", "causal", [1.0, 0.0, 0.0]),
        _make_unit("B", "bayesian", [0.9, 0.1, 0.0]),
    ]
    query_emb = [1.0, 0.0, 0.0]
    scored = _score_methods(query_emb, units, domain="")
    assert len(scored) == 2
