#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
# 先装项目运行时依赖，确保 RecursiveCharacterTextSplitter 等组件可直接导入。
python -m pip install -r requirements.txt
# 再装测试依赖，便于立即执行验收命令。
python -m pip install pytest

echo "[ok] virtual environment ready: $ROOT_DIR/.venv"
echo "[hint] activate with: source .venv/bin/activate"
