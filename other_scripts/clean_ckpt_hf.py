import os
import re
import argparse
import subprocess

JFS_BIN = "/mnt/ws-jfs/dsh-jfs/.jfs"


def matches_target(name, prefix, substring):
    matches_prefix = bool(prefix) and name.startswith(prefix)
    matches_substring = bool(substring) and substring in name
    return matches_prefix or matches_substring


def extract_step(step_dir_name):
    match = re.search(r"(\d+)$", step_dir_name)
    if match is None:
        return None
    return int(match.group(1))


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
    "-m",
    "--max-step",
    type=int,
    required=True,
    help="Only delete hf_model under step directories whose step number is less than or equal to this value.",
)
parser.add_argument(
    "--hf-dir",
    default="hf_model",
    help="Delete this subdirectory under each eligible step directory.",
)
parser.add_argument("-d", "--dry-run", action="store_true", help="Only print matching hf_model paths without deleting them.")
args = parser.parse_args()

if not os.path.isdir(args.base_dir):
    raise FileNotFoundError(f"base_dir does not exist: {args.base_dir}")
if not args.prefix and not args.substring:
    raise ValueError("At least one of --prefix or --substring must be provided.")

print(f"base_dir: {args.base_dir}")
print(f"prefix: {args.prefix}")
print(f"substring: {args.substring}")
print(f"max_step: {args.max_step}")
print(f"hf_dir: {args.hf_dir}")
print(f"dry_run: {args.dry_run}")
print(f"jfs_bin: {JFS_BIN}")

matched_run_dirs = []
matched_step_dirs = []
deleted_hf_paths = []

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

        step_num = extract_step(step_entry.name)
        if step_num is None or step_num > args.max_step:
            continue

        hf_path = os.path.join(step_entry.path, args.hf_dir)
        if not os.path.lexists(hf_path):
            continue

        matched_step_dirs.append(step_entry.path)

        try:
            delete_path(hf_path, args.dry_run)
            deleted_hf_paths.append(hf_path)
        except Exception as e:
            if isinstance(e, subprocess.CalledProcessError):
                stderr = e.stderr.strip() if e.stderr else ""
                stdout = e.stdout.strip() if e.stdout else ""
                details = stderr or stdout or str(e)
                print(f"failed to delete {hf_path}: {details}")
            else:
                print(f"failed to delete {hf_path}: {e}")

summary_key = "matched_hf_count" if args.dry_run else "deleted_hf_count"
print(f"matched_run_dir_count: {len(matched_run_dirs)}")
print(f"matched_step_dir_count: {len(matched_step_dirs)}")
print(f"{summary_key}: {len(deleted_hf_paths)}")
