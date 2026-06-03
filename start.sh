#!/usr/bin/env bash
# Launch the research-knowledge MCP server.
#
# Auth: set ANTHROPIC_API_KEY (https://console.anthropic.com/).
# Data: set RESEARCH_KNOWLEDGE_DATA_DIR to your corpus/index location
#       (defaults to ./research-knowledge-data).
set -euo pipefail

cd "$(dirname "$0")"

# Use a local venv if present, otherwise rely on the active environment.
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

exec python -m research_knowledge.server
