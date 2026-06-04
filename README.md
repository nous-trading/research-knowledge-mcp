# research-knowledge-mcp

Turn a folder of academic PDFs into a searchable knowledge base your LLM can
query over [MCP](https://modelcontextprotocol.io/). It ingests papers with
[Anthropic's Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval),
indexes them with hybrid search (BM25 + dense vectors), and reranks results with
a cross-encoder — so a question like *"manipulation-proof Sharpe ratio threshold"*
returns the right passages, not just keyword matches.

Use it as an **MCP server** (Claude Code, Claude Desktop, any MCP client) or as a
**standalone CLI**.

---

## Why this exists

Naive RAG chunks a document and embeds each chunk in isolation — so a chunk that
says *"the threshold is 0.95"* loses the fact that it's talking about the
*probabilistic Sharpe ratio*. Retrieval quality suffers.

This project implements **Contextual Retrieval**: before embedding, each chunk is
prefixed with a short, LLM-generated description of how it fits into the whole
document. Combined with hybrid search and reranking, retrieval is markedly more
accurate. The expensive part (one LLM call per chunk) is done **once at ingest
time** and made cheap with prompt caching; **search is fully local and free**.

```
PDF ─► Docling parse ─► chunk ─► Contextual Retrieval (LLM, cached) ─┐
                                                                     ▼
                                            ┌──────── dense (bge-m3 → FAISS)
        query ─► BM25 + dense ─► RRF fuse ─►│
                                            └──────── BM25 (SQLite FTS5)
                                                     │
                                                     ▼
                                         cross-encoder rerank (bge-reranker-v2-m3)
                                                     │
                                                     ▼
                                                  top-k results
```

---

## Features

- **Contextual Retrieval** ingest pipeline (Anthropic Cookbook pattern) with
  prompt caching — ~90% cheaper on repeated chunks of the same document.
- **Hybrid search**: BM25 (SQLite FTS5) + dense vectors (BAAI/bge-m3, 1024-d)
  fused with Reciprocal Rank Fusion, then reranked with a cross-encoder.
- **Local & free at query time** — embeddings and reranking run on-device.
- **6 MCP tools**: `search_papers`, `get_paper`, `list_papers`, `cite`,
  `summarize_paper`, `get_concept`.
- **Pluggable auth**: standard `ANTHROPIC_API_KEY`, or inject your own provider
  (e.g. Claude Code OAuth token) — see [Authentication](#authentication).
- **PDF → structured markdown** via [Docling](https://github.com/docling-project/docling).

---

## Install

Requires Python 3.11+.

```bash
git clone https://github.com/nous-trading/research-knowledge-mcp.git
cd research-knowledge-mcp
pip install -e .
```

> **First run downloads ~3 GB of models** (one time): bge-m3 embeddings (~2 GB),
> bge-reranker-v2-m3 (~600 MB), and Docling layout models (~500 MB), cached under
> `~/.cache/huggingface/`.

---

## Configuration

Two environment variables control everything:

| Variable | Purpose | Default |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | Auth for the ingest-time LLM calls. Get one at [console.anthropic.com](https://console.anthropic.com/). | — (required unless an OAT provider is registered) |
| `RESEARCH_KNOWLEDGE_DATA_DIR` | Where papers and the index live. | `./research-knowledge-data` |

The data directory is laid out as:

```
<data>/papers/inbox       ← drop PDFs here to ingest
<data>/papers/processed   ← originals moved here after ingest
<data>/papers/markdown    ← Docling markdown output
<data>/chunks             ← contextualized chunks ({paper_id}.jsonl)
<data>/manifest.json      ← ingest catalog
<data>/index              ← FAISS + SQLite FTS5
<data>/concepts           ← optional concept definitions (markdown + frontmatter)
```

---

## Quick start (CLI)

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export RESEARCH_KNOWLEDGE_DATA_DIR=~/research-data

# 1. Drop PDFs into the inbox
mkdir -p ~/research-data/papers/inbox
cp ~/Downloads/*.pdf ~/research-data/papers/inbox/

# 2. Estimate cost first (no network calls)
python -m research_knowledge ingest ~/research-data/papers/inbox --dry-run

# 3. Ingest
python -m research_knowledge ingest ~/research-data/papers/inbox

# 4. Search
python -m research_knowledge search "manipulation-proof Sharpe ratio threshold"
python -m research_knowledge search "order flow imbalance" --top-k 10 --debug
```

Other commands: `list`, `status`, `rebuild-index`, `summarize <paper_id>`.

---

## Run as an MCP server

```bash
# stdio (for Claude Desktop / Claude Code)
python -m research_knowledge.server

# HTTP
MCP_TRANSPORT=streamable-http MCP_PORT=8014 python -m research_knowledge.server
```

Register it with an MCP client, e.g. Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "research-knowledge": {
      "command": "python",
      "args": ["-m", "research_knowledge.server"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "RESEARCH_KNOWLEDGE_DATA_DIR": "/absolute/path/to/research-data"
      }
    }
  }
}
```

### MCP tools

| Tool | Signature | Purpose |
|------|-----------|---------|
| `search_papers` | `(query, top_k=20, paper_ids=None)` | Hybrid search (BM25 + dense + rerank) |
| `get_paper` | `(paper_id, format="markdown"\|"bibtex")` | Full text or citation |
| `list_papers` | `(year_min=None)` | List ingested papers (optionally filtered by year) |
| `cite` | `(paper_id, style="apa"\|"bibtex")` | Build a citation string |
| `summarize_paper` | `(paper_id, focus=None)` | LLM summary (cached per focus) |
| `get_concept` | `(concept_name)` | Look up a concept note, else fall back to search |

---

## Authentication

The ingest pipeline makes LLM calls (contextualization, metadata, summaries).
Credentials are resolved in this order:

1. **`ANTHROPIC_API_KEY`** — the standard path. Set it and you're done.
2. **An injected OAT provider** — for hosts that authenticate via a Claude Code
   OAuth token instead of an API key. Register one at startup:

   ```python
   from research_knowledge import set_oat_provider

   set_oat_provider(lambda: my_oat_provider)  # any object exposing
   # .access_token, .get_anthropic_client(), .get_async_anthropic_client()
   ```

   When no provider is registered, OAT is simply skipped and the API key is used.

Search and reranking require **no credentials** — they run locally.

---

## Costs

| Item | Per paper (~80 chunks) | Per 100 papers |
|------|------------------------|----------------|
| Contextualization (LLM, one-time) | ~$0.03–0.05 | ~$3–5 |
| Embedding (local) | free | free |
| Search + reranking (local) | free | free |

Ongoing monthly cost to search an existing corpus: **$0**.

---

## Troubleshooting

**First ingest hangs for a minute** — it's downloading models (~3 GB) on first
run. Subsequent runs use the cache.

**`faiss`/`torch` SEGFAULT on Apple Silicon** — an OpenMP clash. The CLI and
server set `OMP_NUM_THREADS=1` automatically; if invoking modules directly, set
it yourself.

**FTS5 query error (`syntax error near ":"`)** — wrap queries with special
characters in quotes: `python -m research_knowledge search '"order flow imbalance"'`.

**Sparse results** — rebuild the index with
`python -m research_knowledge rebuild-index`, or pass `--debug` to see where
candidates drop off (BM25 / dense / fusion / rerank).

---

## Built on

- [Anthropic Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [Docling](https://github.com/docling-project/docling) (IBM Research) — PDF → markdown
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) — dense embeddings
- [BAAI/bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) — reranking
- [FastMCP](https://github.com/jlowin/fastmcp) — MCP server framework

## License

MIT — see [LICENSE](LICENSE).
