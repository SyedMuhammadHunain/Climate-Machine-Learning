#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
if [ -d "$SCRIPT_DIR/../frontend/climate-resilient-frontend" ]; then
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
    PROJECT_ROOT="$SCRIPT_DIR"
fi
FRONTEND_DIR="$PROJECT_ROOT/frontend/climate-resilient-frontend"

cd "$FRONTEND_DIR"

echo "Starting Frontend standalone server at http://localhost:3000..."
exec python3 -m http.server 3000 --bind 127.0.0.1
