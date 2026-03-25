#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/python ]]; then
  echo "Create venv: python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'"
  exit 1
fi
exec .venv/bin/python -m uvicorn incident_memory_engine.api.app:app --host 127.0.0.1 --port 8000 --reload
