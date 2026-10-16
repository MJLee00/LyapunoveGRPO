#!/usr/bin/env python3
"""Print every eval result under CMDP/v5_*/ as a LaTeX row.

Columns (in order): AIME24 & AIME25 & AMC23 & MATH-500 & Minerva & Olympiad & Avg.

For each metrics_0.json found under
  <val-results-dir>/v5_*/<step>_<bal>_<ws>_<YYYYMMDD_HHMMSS>/metrics_0.json
we extract the largest `avg@N` per data_source (matching what ray_trainer
logs as `val-core/<src>/acc/avg@<N>`) and emit one row.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from glob import glob


DEFAULT_VAL_RESULTS_DIR = "/mnt/ws-jfs/dsh-jfs/lrz/verl/val_results/CMDP"

# (column label, list of case-insensitive substrings to match data_source).
COLUMNS = [
    ("AIME24",   ["aime24", "aime_24", "aime2024"]),
    ("AIME25",   ["aime25", "aime_25", "aime2025"]),
    ("AMC23",    ["amc"]),
    ("MATH-500", ["math"]),
    ("Minerva",  ["minerva"]),
    ("Olympiad", ["olymp"]),
]

METRIC_KEY_RE = re.compile(r"^val-core/(?P<src>[^/]+)/(?P<var>[^/]+)/(?P<metric>.+)$")
AVG_AT_N_RE = re.compile(r"^avg@(\d+)$")
VAL_RESULT_DIR_RE = re.compile(r"^(?P<step>\d+)_(?P<bal>True|False)_(?P<ws>\d+)_(?P<ts>\d{8}_\d{6})$")


DEFAULT_NS = {4, 32}


def extract_avg_at_n(metrics: dict) -> dict:
    """{data_source: (N, score)} at the largest N, from raw metrics_0.json."""
    buckets: dict = defaultdict(dict)
    for key, val in metrics.items():
        km = METRIC_KEY_RE.match(key)
        if not km or km.group("var") not in ("acc", "reward"):
            continue
        am = AVG_AT_N_RE.match(km.group("metric"))
        if not am:
            continue
        buckets[km.group("src")][int(am.group(1))] = val
    return {src: (max(d), d[max(d)]) for src, d in buckets.items()}


def find_entry(sources: dict, needles):
    """Return (N, score) for the first data_source matching `needles`, else None."""
    # MATH-500 needs to avoid matching AIME/minerva/olympiad; disambiguate by
    # excluding the other column keywords when looking up "math".
    for src, entry in sources.items():
        src_l = src.lower()
        if needles == ["math"]:
            if "math" in src_l and not any(
                kw in src_l for kw in ("aime", "amc", "minerva", "olymp", "mmlu", "gsm")
            ):
                return entry
        elif any(n in src_l for n in needles):
            return entry
    return None


def render_row(sources: dict, pct: bool = True, digits: int = 1):
    """Return (row_str, non_default_notes).

    `non_default_notes` is a list of "<Column>=avg@<N>" strings for any column
    whose selected N is not in DEFAULT_NS (4 or 32).
    """
    def fmt(v):
        if v is None:
            return ""
        return f"{v * 100:.{digits}f}" if pct else f"{v:.{digits}f}"

    cells = []
    vals = []
    notes = []
    for label, sel in COLUMNS:
        entry = find_entry(sources, sel)
        if entry is None:
            cells.append("")
            vals.append(None)
            continue
        n, score = entry
        cells.append(fmt(score))
        vals.append(score)
        if n not in DEFAULT_NS:
            notes.append(f"{label}=avg@{n}")
    avg = sum(vals) / len(vals) if all(v is not None for v in vals) else None
    cells.append(fmt(avg))
    return "& " + " & ".join(cells) + r" \\", notes


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--val-results-dir", default=DEFAULT_VAL_RESULTS_DIR)
    p.add_argument("--exp-glob", default="v5_*",
                   help="Glob (under --val-results-dir) for experiment dirs.")
    args = p.parse_args()

    exp_dirs = sorted(d for d in glob(os.path.join(args.val_results_dir, args.exp_glob))
                      if os.path.isdir(d))
    if not exp_dirs:
        print(f"[error] no experiment dirs under {args.val_results_dir} matching "
              f"{args.exp_glob!r}", file=sys.stderr)
        sys.exit(1)

    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir)
        entries = []
        for sub in sorted(os.listdir(exp_dir)):
            m = VAL_RESULT_DIR_RE.match(sub)
            if not m:
                continue
            mfile = os.path.join(exp_dir, sub, "metrics_0.json")
            if not os.path.exists(mfile):
                continue
            entries.append((int(m.group("step")), m.group("ts"), sub, mfile))
        if not entries:
            print(f"[warn] {exp_name}: no metrics_0.json found", file=sys.stderr)
            continue
        entries.sort(key=lambda x: (x[0], x[1]))
        print(f"[{exp_name}]")
        for step, ts, sub, mfile in entries:
            try:
                with open(mfile) as f:
                    raw = json.load(f)
            except Exception as e:
                print(f"[warn] failed to read {mfile}: {e}", file=sys.stderr)
                continue
            sources = extract_avg_at_n(raw)
            row, notes = render_row(sources)
            note_str = f"  [{', '.join(notes)}]" if notes else ""
            print(f"  {sub}{note_str}")
            print("  " + row)


if __name__ == "__main__":
    main()
