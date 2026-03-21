#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 predict.py "$1" "$2"
