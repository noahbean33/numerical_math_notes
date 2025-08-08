#!/usr/bin/env python3
"""
remove_matlab_and_empty_dirs.py

Safely identify and delete MATLAB .m files, then remove empty directories.

Defaults to a DRY RUN (no deletions). Use --force to actually delete.

Usage:
  python remove_matlab_and_empty_dirs.py            # dry run
  python remove_matlab_and_empty_dirs.py --force    # actually delete
  python remove_matlab_and_empty_dirs.py --ext .m   # choose extension(s)

Notes:
- Only targets files ending with .m by default (case-insensitive).
- Does NOT delete .mlx by default. You can include via --ext .mlx
- Empty directory cleanup occurs after file deletions when --force is used.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List


def find_files(root: Path, exts: Iterable[str]) -> List[Path]:
    exts_norm = {e.lower() if e.startswith('.') else f'.{e.lower()}' for e in exts}
    out: List[Path] = []
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in exts_norm:
            out.append(p)
    return out


def remove_files(files: Iterable[Path], dry_run: bool) -> int:
    removed = 0
    for fp in files:
        try:
            if dry_run:
                print(f"[DRY-RUN] Would delete file: {fp}")
            else:
                fp.unlink(missing_ok=True)
                print(f"[DELETE] File deleted: {fp}")
                removed += 1
        except Exception as e:
            print(f"[ERROR] Could not delete {fp}: {e}")
    return removed


def _is_under_git(dir_path: Path, root: Path) -> bool:
    """Return True if dir_path is inside a .git directory under root."""
    try:
        rel = dir_path.resolve().relative_to(root.resolve())
    except Exception:
        # If not relative (e.g., different drive), conservatively do not treat as under .git
        return False
    return any(part == '.git' for part in rel.parts)


def remove_empty_dirs(root: Path, dry_run: bool) -> int:
    removed_dirs = 0
    # Walk bottom-up so we can remove parents after children
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        d = Path(dirpath)
        # Skip the repo root itself
        if d == root:
            continue
        # Never touch .git contents
        if _is_under_git(d, root):
            continue
        try:
            # If directory is empty (no files and no subdirs)
            if not any(Path(dirpath).iterdir()):
                if dry_run:
                    print(f"[DRY-RUN] Would remove empty dir: {d}")
                else:
                    d.rmdir()
                    print(f"[DELETE] Empty dir removed: {d}")
                removed_dirs += 1
        except Exception as e:
            print(f"[WARN] Could not inspect/remove dir {d}: {e}")
    return removed_dirs


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove MATLAB .m files and then empty directories.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Root directory to operate on (default: current working directory)")
    parser.add_argument("--ext", dest="exts", action="append", default=[".m"], help="File extension(s) to remove (default: .m). Can be specified multiple times.")
    parser.add_argument("--force", action="store_true", help="Perform actual deletion. Without this flag, runs in dry-run mode.")
    parser.add_argument("--limit", type=int, default=0, help="If >0, only operate on the first N matching files (useful for testing).")
    args = parser.parse_args()

    root: Path = args.root.resolve()
    dry_run = not args.force

    print(f"Root: {root}")
    print(f"Extensions targeted: {args.exts}")
    print(f"Mode: {'DRY-RUN' if dry_run else 'DELETE'}")

    files = find_files(root, args.exts)
    total = len(files)
    print(f"Found {total} matching files.")

    if args.limit and total > args.limit:
        files = files[: args.limit]
        print(f"Limiting to first {args.limit} files for this run.")

    removed_files = remove_files(files, dry_run=dry_run)

    removed_dirs = 0
    if dry_run:
        print("Skipping empty directory removal in DRY-RUN mode (will preview what would be removed)...")
        # Preview empty dirs that would be removed
        for dirpath, dirnames, filenames in os.walk(root, topdown=False):
            d = Path(dirpath)
            if d == root:
                continue
            if _is_under_git(d, root):
                continue
            try:
                if not any(d.iterdir()):
                    print(f"[DRY-RUN] Would remove empty dir: {d}")
            except Exception as e:
                print(f"[WARN] Could not inspect dir {d}: {e}")
    else:
        removed_dirs = remove_empty_dirs(root, dry_run=False)

    print("\nSummary:")
    print(f" - Files matched: {total}")
    print(f" - Files {'to be ' if dry_run else ''}deleted: {len(files)}")
    if not dry_run:
        print(f" - Empty directories removed: {removed_dirs}")

    if dry_run:
        print("\nRun with --force to actually delete the files and remove empty directories.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
