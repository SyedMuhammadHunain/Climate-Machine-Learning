#!/usr/bin/env bash
# Runs the Streamlit app locally (the same app as on Streamlit Community Cloud).
#   ./scripts/run/run_streamlit.sh            http://127.0.0.1:8501
# First run creates .venv-streamlit with only streamlit_app/requirements.txt, like the cloud build.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib_env.sh"
ROOT="$(project_root)"
cd "$ROOT"
ensure_venv "$ROOT/.venv-streamlit" "$ROOT/streamlit_app/requirements.txt" streamlit

PORT="${PORT:-8501}"
echo
echo "  Streamlit app: http://127.0.0.1:$PORT   (stop with Ctrl+C)"
echo
exec "$ROOT/.venv-streamlit/bin/streamlit" run streamlit_app/app.py \
    --server.address 127.0.0.1 --server.port "$PORT" --server.headless true --browser.gatherUsageStats false
