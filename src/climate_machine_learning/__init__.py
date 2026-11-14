import sys
import argparse
from pathlib import Path
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="climate-machine-learning",
        description="CLI for Climate-Resilient Low-Water Crop Selection Machine Learning System",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: serve
    serve_parser = subparsers.add_parser("serve", help="Launch the FastAPI local web application server")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    serve_parser.add_argument("--reload", action="store_true", help="Enable live auto-reload for development")

    # Command: verify
    subparsers.add_parser("verify", help="Run the automated backend test suite")

    args = parser.parse_args()

    if args.command == "serve" or args.command is None:
        host = getattr(args, "host", "127.0.0.1")
        port = getattr(args, "port", 8000)
        reload = getattr(args, "reload", False)
        print(f"Starting Climate-Machine-Learning FastAPI server on http://{host}:{port}...")
        print(f"Interactive Swagger documentation available at: http://{host}:{port}/docs")
        uvicorn.run("app.main:app", host=host, port=port, reload=reload)
    elif args.command == "verify":
        from scripts.verify_backend import run_verification
        run_verification()


if __name__ == "__main__":
    main()
