"""Tests for domain_loader concept_keywords support."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "apps", "api"))

from chat.domain_loader import (
    DomainHint,
    get_all_concept_keywords,
    match_concept_keywords,
)


def _make_hints() -> dict[str, DomainHint]:
    """Build a small set of DomainHint objects for testing."""
    return {
        "Bayesian": DomainHint(
            name="Bayesian",
            description="Bayesian statistics",
            keywords=["bayesian", "prior"],
            concept_keywords=["posterior distribution", "markov chain monte carlo", "credible interval"],
        ),
        "Survival": DomainHint(
            name="Survival",
            description="Survival analysis",
            keywords=["survival", "hazard"],
            concept_keywords=["kaplan-meier", "cox regression", "credible interval"],
        ),
    }


class TestDomainHintConceptKeywords:
    def test_concept_keywords_field_accepted(self):
        hint = DomainHint(
            name="Test",
            description="desc",
            concept_keywords=["term a", "term b"],
        )
        assert hint.concept_keywords == ["term a", "term b"]

    def test_concept_keywords_defaults_to_empty(self):
        hint = DomainHint(name="Test", description="desc")
        assert hint.concept_keywords == []


class TestGetAllConceptKeywords:
    def test_returns_deduplicated_list(self):
        hints = _make_hints()
        all_kw = get_all_concept_keywords(hints)
        # "credible interval" appears in both domains but should appear once
        assert isinstance(all_kw, list)
        assert all_kw.count("credible interval") == 1
        assert "posterior distribution" in all_kw
        assert "kaplan-meier" in all_kw

    def test_empty_hints(self):
        assert get_all_concept_keywords({}) == []


class TestMatchConceptKeywords:
    def test_extracts_matching_keywords(self):
        hints = _make_hints()
        text = "We used posterior distribution and cox regression in this study."
        matched = match_concept_keywords(text, hints)
        assert "posterior distribution" in matched
        assert "cox regression" in matched
        assert "kaplan-meier" not in matched

    def test_case_insensitive(self):
        hints = _make_hints()
        text = "The Posterior Distribution was estimated via Markov Chain Monte Carlo."
        matched = match_concept_keywords(text, hints)
        assert "posterior distribution" in matched
        assert "markov chain monte carlo" in matched

    def test_no_matches(self):
        hints = _make_hints()
        text = "This sentence has nothing relevant."
        matched = match_concept_keywords(text, hints)
        assert matched == []

    def test_filter_by_domain_names(self):
        hints = _make_hints()
        text = "We computed the posterior distribution and cox regression."
        matched = match_concept_keywords(text, hints, domain_names=["Bayesian"])
        assert "posterior distribution" in matched
        assert "cox regression" not in matched

    def test_filter_with_empty_domain_names(self):
        hints = _make_hints()
        text = "posterior distribution"
        matched = match_concept_keywords(text, hints, domain_names=[])
        assert matched == []
