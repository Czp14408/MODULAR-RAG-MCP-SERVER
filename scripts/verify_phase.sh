#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "[error] .venv not found. Run: bash scripts/bootstrap_venv.sh"
  exit 1
fi

PHASE="${1:-}" 

case "$PHASE" in
  A1|a1)
    .venv/bin/python -c "import mcp_server; import core; import ingestion; import libs; import observability; print('imports_ok')"
    PYTHONPYCACHEPREFIX=/tmp/pythoncache .venv/bin/python -m compileall src
    .venv/bin/python main.py
    ;;
  A2|a2)
    .venv/bin/python -m pytest -q tests/unit/test_smoke_imports.py
    .venv/bin/python -m pytest -q
    ;;
  *)
    echo "Usage: bash scripts/verify_phase.sh <A1|A2>"
    exit 2
    ;;
esac

echo "[ok] $PHASE verification completed"
