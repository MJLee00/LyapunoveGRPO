import os
import argparse
import subprocess

JFS_BIN = "/mnt/ws-jfs/dsh-jfs/.jfs"

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--base_dir", default="/mnt/ws-jfs/dsh-jfs/lrz/verl/ckpts/CMDP", help="input file")
parser.add_argument(
    "-p",
    "--prefix",
    default="",
    help="Delete files or directories under base_dir whose names start with this prefix.",
)
parser.add_argument(
    "--pattern",
    default="",
    help="Delete files or directories under base_dir whose names contain this substring.",
)
parser.add_argument("-d", "--dry-run", action="store_true", help="Only print matching paths without deleting them.")
args = parser.parse_args()

if not os.path.isdir(args.base_dir):
    raise FileNotFoundError(f"base_dir does not exist: {args.base_dir}")
if not args.prefix and not args.pattern:
    raise ValueError("At least one of --prefix or --pattern must be provided.")

print(f"base_dir: {args.base_dir}")
print(f"prefix: {args.prefix}")
print(f"pattern: {args.pattern}")
print(f"dry_run: {args.dry_run}")
print(f"jfs_bin: {JFS_BIN}")

deleted_paths = []

for entry in os.scandir(args.base_dir):
    matches_prefix = bool(args.prefix) and entry.name.startswith(args.prefix)
    matches_pattern = bool(args.pattern) and args.pattern in entry.name

    if not matches_prefix and not matches_pattern:
        continue

    try:
        if args.dry_run:
            print(f"would delete: {entry.path}")
        else:
            subprocess.run(
                [JFS_BIN, "rmr", "--skip-trash", entry.path],
                check=True,
                text=True,
                capture_output=True,
            )
            print(f"deleted: {entry.path}")
        deleted_paths.append(entry.path)
    except Exception as e:
        if isinstance(e, subprocess.CalledProcessError):
            stderr = e.stderr.strip() if e.stderr else ""
            stdout = e.stdout.strip() if e.stdout else ""
            details = stderr or stdout or str(e)
            print(f"failed to delete {entry.path}: {details}")
        else:
            print(f"failed to delete {entry.path}: {e}")

summary_key = "matched_count" if args.dry_run else "deleted_count"
print(f"{summary_key}: {len(deleted_paths)}")
