#!/usr/bin/env bash
# The HTML pages call the API on their own address (same origin), so they are served by the backend,
# not by a separate static server. This script starts that one server; see run_app.sh.
echo "The frontend pages are served by the backend at http://127.0.0.1:${PORT:-8000} (no separate frontend server)."
exec "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/run_app.sh" "$@"
