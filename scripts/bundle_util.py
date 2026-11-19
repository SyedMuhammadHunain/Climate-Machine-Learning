"""Shared by the bundle builders: prepare an output folder without ever deleting something that is not ours."""
import shutil
from pathlib import Path
from typing import Iterable


class UnsafeOutputDir(Exception):
    """The output folder is not one a builder may clear."""


def prepare_out(out: Path, owned: Iterable[str], marker: str) -> Path:
    """Make `out` ready for a fresh build.

    The folder itself is never deleted. Existing content is only touched when the folder is empty or
    carries our marker file, and then only the entries named in `owned` are replaced. Symlinks are
    refused, so a link cannot point a deletion somewhere else.
    """
    out = Path(out).expanduser().resolve()
    if out.exists():
        if not out.is_dir():
            raise UnsafeOutputDir(f"{out} exists and is not a folder.")
        if any(out.iterdir()) and not (out / marker).is_file():
            raise UnsafeOutputDir(
                f"{out} is not empty and was not created by this script (no {marker} file). "
                "Refusing to delete anything there. Choose a new or empty folder."
            )
        for name in owned:
            target = out / name
            if target.is_symlink():
                raise UnsafeOutputDir(f"{target} is a symlink. Refusing to follow it.")
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
    out.mkdir(parents=True, exist_ok=True)
    (out / marker).write_text("This folder is rebuilt by a bundle script; only its own entries are replaced.\n")
    return out
