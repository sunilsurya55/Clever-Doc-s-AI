#!/usr/bin/env bash
set -euo pipefail

# Always run from project root (where run.sh lives)
cd "$(dirname "$0")"
ROOT="$(pwd)"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# NOTE: we deliberately do NOT force HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1
# here. The embedding model (sentence-transformers/all-MiniLM-L6-v2) needs to
# reach Hugging Face Hub on its very first run if no local copy is found at
# EMBEDDING_MODEL_LOCAL_PATH (see app/embeddings.py). After that first run the
# model is cached locally and no further network access is required. If you
# want to force fully offline behaviour on a machine that already has the
# model cached, uncomment the next two lines:
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

# Prefer a project-local virtualenv if one exists, otherwise fall back to
# whatever python3/python is on PATH. This works across macOS/Linux/most
# machines instead of a hardcoded personal path.
if [ -x "${ROOT}/venv/bin/python" ]; then
  PYTHON_BIN="${ROOT}/venv/bin/python"
elif [ -x "${ROOT}/.venv/bin/python" ]; then
  PYTHON_BIN="${ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "[run.sh] ERROR: no python3/python interpreter found on PATH." >&2
  exit 1
fi

export PYTHONPATH="${ROOT}"
export FLASK_APP=app.app
export FLASK_ENV=development

mkdir -p "${ROOT}/logs" "${ROOT}/outputs" "${ROOT}/uploads"

echo "[run.sh] Using python: ${PYTHON_BIN}"
echo "[run.sh] Project root: ${ROOT}"
exec "${PYTHON_BIN}" -m flask run --host=127.0.0.1 --port=5000
