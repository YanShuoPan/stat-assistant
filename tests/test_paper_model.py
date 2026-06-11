"""Tests for Paper model and paper_id FK on KnowledgeUnit."""
from models import Paper, KnowledgeUnit
from conftest import TestSession


def test_paper_creation(setup_db):
    """Paper can be created and persisted."""
    db = TestSession()
    paper = Paper(
        title="Bootstrap Methods for Time Series",
        authors="Efron, B.",
        year=1979,
        doi="10.1214/aos/1176344552",
        arxiv_id=None,
        domain="statistics",
        cluster="resampling",
        filename="efron1979.pdf",
    )
    db.add(paper)
    db.commit()
    db.refresh(paper)

    assert paper.id is not None
    assert paper.title == "Bootstrap Methods for Time Series"
    assert paper.domain == "statistics"
    assert paper.year == 1979
    assert paper.created_at is not None
    db.close()


def test_knowledge_unit_linked_to_paper(setup_db):
    """KnowledgeUnit can be linked to a Paper via paper_id FK."""
    db = TestSession()
    paper = Paper(
        title="Lasso Regression",
        authors="Tibshirani, R.",
        year=1996,
        domain="statistics",
        filename="tibshirani1996.pdf",
    )
    db.add(paper)
    db.commit()
    db.refresh(paper)

    ku = KnowledgeUnit(
        source_type="paper",
        title="Lasso penalty definition",
        knowledge_type="definition",
        content="The Lasso uses an L1 penalty...",
        paper_id=paper.id,
    )
    db.add(ku)
    db.commit()
    db.refresh(ku)

    assert ku.paper_id == paper.id
    assert ku.id is not None
    db.close()


def test_knowledge_unit_without_paper(setup_db):
    """KnowledgeUnit can have paper_id=None for backward compatibility."""
    db = TestSession()
    ku = KnowledgeUnit(
        source_type="manual",
        title="General note",
        knowledge_type="note",
        content="Some manually entered knowledge.",
        paper_id=None,
    )
    db.add(ku)
    db.commit()
    db.refresh(ku)

    assert ku.paper_id is None
    assert ku.id is not None
    db.close()
