#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash run.sh <questions_txt_path> <predictions_out_path>" >&2
  exit 1
fi

if python3 - <<'PY' >/dev/null 2>&1
import importlib.util
mods = ["rank_bm25", "torch", "transformers"]
missing = [name for name in mods if importlib.util.find_spec(name) is None]
raise SystemExit(1 if missing else 0)
PY
then
  python3 predict.py "$1" "$2"
elif command -v conda >/dev/null 2>&1; then
  source ~/.bashrc >/dev/null 2>&1 || true
  conda run -n cs188 python3 predict.py "$1" "$2"
else
  echo "python3 is missing required packages and no conda fallback was found." >&2
  exit 1
fi
