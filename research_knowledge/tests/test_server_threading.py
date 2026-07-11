"""검색 스레드 고정 회귀 — sqlite3 커넥션은 생성 스레드에 묶이므로
동시 검색이 전부 전용 단일 스레드(rk-search)에서 실행되어야 한다.

asyncio.to_thread(공유 풀)로 돌리면 두 번째 검색부터 "SQLite objects
created in a thread can only be used in that same thread"가 난다
(2026-07-11 프로덕션 병렬 호출 실측).
"""

import asyncio
import threading

from research_knowledge import server


def test_concurrent_searches_run_on_single_dedicated_thread(monkeypatch):
    monkeypatch.setattr(server, "_MODEL_TTL_SECONDS", 0)  # 워치독 태스크 미생성
    seen: list[str] = []

    def fake_run_search(query, top_k, paper_ids):
        seen.append(threading.current_thread().name)
        return None  # index-not-ready 경로 → 에러 JSON 반환으로 종료

    monkeypatch.setattr(server, "_run_search", fake_run_search)

    async def go():
        await asyncio.gather(*(server.search_papers(f"q{i}") for i in range(4)))

    asyncio.run(go())

    assert len(seen) == 4
    assert len(set(seen)) == 1, f"searches ran on multiple threads: {set(seen)}"
    assert seen[0].startswith("rk-search")
