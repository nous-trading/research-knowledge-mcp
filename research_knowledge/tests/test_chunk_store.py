"""chunk_store 회귀 — SQLite on-demand 전환(chunks.json 3.9GB RAM 제거)의 정합성."""

import json

import pytest

from research_knowledge.models import Chunk
from research_knowledge.retrieval import chunk_store


def _chunk(i: int, paper: str = "p1") -> Chunk:
    return Chunk(
        chunk_id=f"{paper}_{i:04d}",
        paper_id=paper,
        section_path=["sec", str(i)],
        page_start=i,
        page_end=i + 1,
        raw_text=f"raw text {i}",
        token_count=3,
        contextualized_text=f"ctx: raw text {i}",
    )


@pytest.fixture
def store(tmp_path):
    conn = chunk_store.create(tmp_path / "chunks.db")
    chunk_store.add_chunks(conn, [_chunk(0), _chunk(1), _chunk(2, paper="p2")])
    return conn


def test_roundtrip_preserves_all_fields(store):
    got = chunk_store.get_by_ids(store, ["p1_0000"])["p1_0000"]
    assert got == _chunk(0)


def test_id_order_matches_insertion_order(store):
    assert chunk_store.id_order(store) == ["p1_0000", "p1_0001", "p2_0002"]


def test_texts_by_ids_returns_contextualized_only(store):
    texts = chunk_store.texts_by_ids(store, ["p1_0001", "p2_0002"])
    assert texts == {"p1_0001": "ctx: raw text 1", "p2_0002": "ctx: raw text 2"}


def test_paper_ids_by_ids(store):
    assert chunk_store.paper_ids_by_ids(store, ["p1_0000", "p2_0002"]) == {
        "p1_0000": "p1",
        "p2_0002": "p2",
    }


def test_empty_id_list_returns_empty(store):
    assert chunk_store.get_by_ids(store, []) == {}
    assert chunk_store.texts_by_ids(store, []) == {}
    assert chunk_store.paper_ids_by_ids(store, []) == {}


def test_count(store):
    assert chunk_store.count(store) == 3


def test_transcript_joins_chunks_in_pos_order(store):
    body = chunk_store.transcript_by_paper(store, "p1")
    assert body.startswith("raw text 0")
    assert "raw text 1" in body
    assert "raw text 2" not in body  # other paper
    assert chunk_store.transcript_by_paper(store, "nope") == ""


def test_migrate_from_json(tmp_path):
    chunks = [_chunk(0), _chunk(1)]
    meta_path = tmp_path / "chunks.json"
    meta_path.write_text(
        json.dumps(
            {
                "chunk_id_order": [c.chunk_id for c in chunks],
                "chunks": {c.chunk_id: c.model_dump() for c in chunks},
            }
        ),
        encoding="utf-8",
    )
    db_path = tmp_path / "chunks.db"

    n = chunk_store.migrate_from_json(meta_path, db_path)

    assert n == 2
    assert not meta_path.exists()
    assert (tmp_path / "chunks.json.bak").exists()
    conn = chunk_store.open_store(db_path)
    assert chunk_store.id_order(conn) == ["p1_0000", "p1_0001"]
    assert chunk_store.get_by_ids(conn, ["p1_0001"])["p1_0001"] == _chunk(1)


def test_open_store_missing_path_raises_without_creating_file(tmp_path):
    """리뷰 결함 #1 회귀: 없는 경로 open 이 0-byte 포이즌 파일을 만들면
    migrate-index 완료 판정과 legacy 안내가 둘 다 무력화된다."""
    missing = tmp_path / "chunks.db"
    with pytest.raises(FileNotFoundError, match="migrate-index"):
        chunk_store.open_store(missing)
    assert not missing.exists()


def test_migrate_leaves_no_tmp_file(tmp_path):
    """리뷰 결함 #4 회귀: 마이그레이션은 tmp 빌드 후 원자 rename —
    chunks.db 존재 = 완료 마커."""
    chunks = [_chunk(0)]
    meta_path = tmp_path / "chunks.json"
    meta_path.write_text(
        json.dumps(
            {
                "chunk_id_order": [c.chunk_id for c in chunks],
                "chunks": {c.chunk_id: c.model_dump() for c in chunks},
            }
        ),
        encoding="utf-8",
    )
    chunk_store.migrate_from_json(meta_path, tmp_path / "chunks.db")
    assert not (tmp_path / "chunks.db.tmp").exists()
    assert (tmp_path / "chunks.db").exists()


def test_search_rejects_nonpositive_rerank_candidates(tmp_path):
    """리뷰 결함 #6 회귀: RERANK_CANDIDATES=0 은 조용한 빈 결과가 아니라
    loud ValueError."""
    from research_knowledge.retrieval.index import HybridIndex
    from research_knowledge.retrieval.search import HybridSearcher

    searcher = HybridSearcher(HybridIndex(tmp_path), None, None)
    with pytest.raises(ValueError, match="rerank_candidates"):
        searcher.search("q", rerank_candidates=0)
    with pytest.raises(ValueError, match="rerank_candidates"):
        searcher.search("q", rerank_candidates=-5)


def test_open_store_backfills_paper_index(tmp_path):
    conn = chunk_store.create(tmp_path / "chunks.db")
    chunk_store.add_chunks(conn, [_chunk(0)])
    conn.execute("DROP INDEX idx_chunks_paper")
    conn.commit()
    conn.close()

    conn = chunk_store.open_store(tmp_path / "chunks.db")
    names = {r[1] for r in conn.execute("PRAGMA index_list(chunks)")}
    assert "idx_chunks_paper" in names
