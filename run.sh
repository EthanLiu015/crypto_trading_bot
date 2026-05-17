#!/usr/bin/env bash
# Always use the project venv (avoids conda base NumPy 2.x conflicts)
set -e
cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  python3.12 -m venv .venv 2>/dev/null || python3 -m venv .venv
  .venv/bin/pip install --upgrade pip
  .venv/bin/pip install -e .
fi

exec .venv/bin/python -m crypto_bot "$@"