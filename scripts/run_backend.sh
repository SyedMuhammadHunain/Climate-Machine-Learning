#!/usr/bin/env bash
# Development server with auto-reload: the API and the HTML pages on http://127.0.0.1:8000. Same as ./run_app.sh --reload.
exec "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/run_app.sh" --reload "$@"
