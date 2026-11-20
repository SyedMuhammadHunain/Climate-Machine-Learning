# Shared helpers for the run_*.sh scripts. Source it; do not run it.
#   project_root            prints the repository root (the nearest folder above the calling script that holds app/main.py)
#   ensure_venv DIR REQ MOD  creates virtualenv DIR (Python 3.13) if missing, installs REQ if MOD cannot be imported

project_root() {
    local here dir
    here="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[1]}")")" && pwd)"
    dir="$here"
    while [ "$dir" != "/" ]; do
        if [ -f "$dir/app/main.py" ]; then echo "$dir"; return; fi
        dir="$(dirname "$dir")"
    done
    echo "$here"
}

ensure_venv() {
    local dir="$1" req="$2" module="$3"
    if [ ! -x "$dir/bin/python" ]; then
        echo "Creating virtual environment $dir (Python 3.13)..."
        if command -v uv >/dev/null 2>&1; then
            uv venv -q -p 3.13 "$dir"
        elif command -v python3.13 >/dev/null 2>&1; then
            python3.13 -m venv "$dir"
        else
            echo "Python 3.13 not found. Install it, or install uv (https://docs.astral.sh/uv/), then run this again." >&2
            exit 1
        fi
    fi
    if ! "$dir/bin/python" -c "import $module" >/dev/null 2>&1; then
        echo "Installing $req into $dir (first run only, takes a few minutes)..."
        if command -v uv >/dev/null 2>&1; then
            uv pip install -q -p "$dir/bin/python" -r "$req"
        else
            "$dir/bin/python" -m pip install -q -r "$req"
        fi
    fi
}
