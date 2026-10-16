import os
import argparse
import subprocess

JFS_BIN = "/mnt/ws-jfs/dsh-jfs/.jfs"


def matches_target(name, prefix, substring):
    matches_prefix = bool(prefix) and name.startswith(prefix)
    matches_substring = bool(substring) and substring in name
    return matches_prefix or matches_substring


def delete_path(path, dry_run):
    if dry_run:
        print(f"would delete: {path}")
        return

    subprocess.run(
        [JFS_BIN, "rmr", "--skip-trash", path],
        check=True,
        text=True,
        capture_output=True,
    )
    print(f"deleted: {path}")


parser = argparse.ArgumentParser()
parser.add_argument("-b", "--base_dir", default="/mnt/ws-jfs/dsh-jfs/lrz/verl/ckpts/CMDP", help="Base directory that contains experiment folders.")
parser.add_argument(
    "-p",
    "--prefix",
    default="",
    help="Only process experiment folders whose names start with this prefix.",
)
parser.add_argument(
    "-s", "--substring",
    default="",
    help="Only process experiment folders whose names contain this substring.",
)
parser.add_argument(
    "-a",
    "--actor-dir",
    default="actor",
    help="Delete this subdirectory under each step directory.",
)
parser.add_argument("-d", "--dry-run", action="store_true", help="Only print matching actor paths without deleting them.")
args = parser.parse_args()

if not os.path.isdir(args.base_dir):
    raise FileNotFoundError(f"base_dir does not exist: {args.base_dir}")
if not args.prefix and not args.substring:
    raise ValueError("At least one of --prefix or --substring must be provided.")

print(f"base_dir: {args.base_dir}")
print(f"prefix: {args.prefix}")
print(f"substring: {args.substring}")
print(f"actor_dir: {args.actor_dir}")
print(f"dry_run: {args.dry_run}")
print(f"jfs_bin: {JFS_BIN}")

matched_run_dirs = []
deleted_actor_paths = []

for entry in os.scandir(args.base_dir):
    if not entry.is_dir(follow_symlinks=False):
        continue
    if not matches_target(entry.name, args.prefix, args.substring):
        continue

    matched_run_dirs.append(entry.path)
    print(f"matched_run_dir: {entry.path}")

    for step_entry in os.scandir(entry.path):
        if not step_entry.is_dir(follow_symlinks=False):
            continue

        actor_path = os.path.join(step_entry.path, args.actor_dir)
        if not os.path.lexists(actor_path):
            continue

        try:
            delete_path(actor_path, args.dry_run)
            deleted_actor_paths.append(actor_path)
        except Exception as e:
            if isinstance(e, subprocess.CalledProcessError):
                stderr = e.stderr.strip() if e.stderr else ""
                stdout = e.stdout.strip() if e.stdout else ""
                details = stderr or stdout or str(e)
                print(f"failed to delete {actor_path}: {details}")
            else:
                print(f"failed to delete {actor_path}: {e}")

summary_key = "matched_actor_count" if args.dry_run else "deleted_actor_count"
print(f"matched_run_dir_count: {len(matched_run_dirs)}")
print(f"{summary_key}: {len(deleted_actor_paths)}")
