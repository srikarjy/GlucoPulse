#!/usr/bin/env bash
# Copy ONLY the public MCP function's files into an empty directory and deploy from there, so nothing else in this
# repository (data/ holds participant data) can be uploaded. Usage: scripts/vercel_stage.sh <empty-dir>
set -euo pipefail
dest="${1:?usage: scripts/vercel_stage.sh <empty-dir>}"
mkdir -p "$dest"
if [ -n "$(ls -A "$dest")" ]; then echo "refusing: $dest is not empty" >&2; exit 1; fi
root="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$dest/api" "$dest/agents/public_mcp" "$dest/serving"
cp "$root/api/index.py" "$dest/api/"
cp "$root/agents/__init__.py" "$dest/agents/"
cp "$root"/agents/public_mcp/*.py "$dest/agents/public_mcp/"
cp "$root/serving/__init__.py" "$root/serving/alerts.py" "$dest/serving/"
cp "$root/requirements.txt" "$root/vercel.json" "$root/.python-version" "$root/.vercelignore" "$dest/"
# guard: no data-like files may be staged
if find "$dest" -type f \( -name '*.csv' -o -name '*.pkl' -o -name '*.npy' -o -name '*.xls*' -o -name '*.zip' -o -name '*.json' ! -name vercel.json \) | grep -q .; then
  echo "refusing: data-like files staged" >&2; exit 1
fi
find "$dest" -type f | sed "s#^$dest/##" | sort
