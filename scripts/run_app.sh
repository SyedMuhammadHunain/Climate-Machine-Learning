#!/usr/bin/env bash
# Runs the whole app: FastAPI backend and the HTML pages, on one local server.
#   ./run_app.sh              http://127.0.0.1:8000
#   PORT=8080 ./run_app.sh    another port
#   ./run_app.sh --reload     restart on code changes (development)
# First run creates .venv and installs requirements.txt. The trained model is already in models/.
set -euo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib_env.sh"
ROOT="$(project_root)"
cd "$ROOT"
ensure_venv "$ROOT/.venv" "$ROOT/requirements.txt" fastapi

PORT="${PORT:-8000}"
BASE="http://127.0.0.1:$PORT"
cat <<EOF

  Frontend (pages, served by the backend):
    Overview            $BASE/
    Crop predictor      $BASE/predict.html
    Model results       $BASE/metrics.html
  Backend (API):
    API docs (Swagger)  $BASE/docs
    Health              $BASE/health
    Endpoints           $BASE/api/v1/predict  /api/v1/recommend  /api/v1/metrics  /api/v1/metadata

  Stop with Ctrl+C.

EOF
exec "$ROOT/.venv/bin/uvicorn" app.main:app --host 127.0.0.1 --port "$PORT" "$@"
