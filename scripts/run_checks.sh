#!/usr/bin/env bash
# Runs every check in the project and prints a summary. Exit code is non-zero if any check fails.
#   ./run_checks.sh          checks that use the main .venv, plus the page dry run (needs node)
#   ./run_checks.sh --all    also the Streamlit and Hugging Face checks, each in its own venv
#                            (.venv-streamlit, .venv-hf; created on first use) and the Streamlit layout check
# verify_model.py retrains once (about 1 minute); verify_report.py needs google-chrome, pdftotext, pdfinfo;
# verify_responsive.py needs google-chrome and renders every page at 9 screen sizes (about 3 minutes).
set -uo pipefail
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/lib_env.sh"
ROOT="$(project_root)"
cd "$ROOT"
set -e
ensure_venv "$ROOT/.venv" "$ROOT/requirements.txt" fastapi
set +e

LOGS="$(mktemp -d)"
declare -a NAMES RESULTS
FAILED=0

run() {  # run NAME COMMAND...
    local name="$1"; shift
    printf "  %-22s " "$name"
    if "$@" >"$LOGS/$name.log" 2>&1; then
        echo "PASS"; RESULTS+=("PASS")
    else
        echo "FAIL  (log: $LOGS/$name.log)"; RESULTS+=("FAIL"); FAILED=1
    fi
    NAMES+=("$name")
}

echo "Running checks (logs in $LOGS)..."
for s in verify_model verify_backend verify_frontend verify_readme verify_guide verify_report verify_vercel_bundle verify_responsive; do
    run "$s" "$ROOT/.venv/bin/python" "scripts/$s.py"
done

if command -v node >/dev/null 2>&1; then
    PORT=8799
    "$ROOT/.venv/bin/uvicorn" app.main:app --host 127.0.0.1 --port "$PORT" >"$LOGS/server.log" 2>&1 &
    SERVER=$!
    trap 'kill $SERVER 2>/dev/null' EXIT
    for _ in $(seq 60); do curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break; sleep 1; done
    run dryrun_demo node scripts/dryrun_demo.js "http://127.0.0.1:$PORT"
    kill "$SERVER" 2>/dev/null
else
    echo "  dryrun_demo            SKIPPED (node not installed)"
fi

if [ "${1:-}" = "--all" ]; then
    ensure_venv "$ROOT/.venv-streamlit" "$ROOT/streamlit_app/requirements.txt" streamlit
    run verify_streamlit "$ROOT/.venv-streamlit/bin/python" scripts/verify_streamlit.py
    run verify_responsive_st "$ROOT/.venv/bin/python" scripts/verify_responsive.py --streamlit
    ensure_venv "$ROOT/.venv-hf" "$ROOT/hf_space/requirements.txt" gradio
    "$ROOT/.venv-hf/bin/python" -c "import yaml" 2>/dev/null || \
        { command -v uv >/dev/null && uv pip install -q -p "$ROOT/.venv-hf/bin/python" pyyaml || "$ROOT/.venv-hf/bin/python" -m pip install -q pyyaml; }
    run verify_hf_bundle "$ROOT/.venv-hf/bin/python" scripts/verify_hf_bundle.py
fi

echo
if [ "$FAILED" -eq 0 ]; then echo "All ${#NAMES[@]} checks passed."; else echo "Some checks failed; see the logs above."; fi
exit "$FAILED"
