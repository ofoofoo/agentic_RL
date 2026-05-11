#!/usr/bin/env python3
"""
compare_dynamic_runs.py

Compares results.json files across all `dynamic *` runs under
  agentic_RL/output/aw_runs/

Statistics reported:
  - Per-run: tasks attempted, tasks completed, accuracy
  - Union of tasks completed by ANY run, accuracy on that union
  - Intersection of tasks completed by ALL runs
  - Per-task breakdown across runs
  - Latency / token averages per run
"""

import json
import os
import glob
import sys
from pathlib import Path
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent / "output" / "aw_runs"
PATTERN = "dynamic*"          # matches any folder starting with "dynamic"
RESULTS_FILE = "results.json"

# ANSI colours
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
GREY   = "\033[90m"
MAGENTA= "\033[95m"

def col(text, colour): return f"{colour}{text}{RESET}"
def header(text):      print(f"\n{BOLD}{CYAN}{'─'*72}\n  {text}\n{'─'*72}{RESET}")
def subhdr(text):      print(f"\n{BOLD}{YELLOW}  {text}{RESET}")

# ── Load runs ─────────────────────────────────────────────────────────────────
run_dirs = sorted(BASE.glob(PATTERN))
if not run_dirs:
    sys.exit(f"No runs matching '{PATTERN}' found under {BASE}")

runs: dict[str, list[dict]] = {}
for d in run_dirs:
    rpath = d / RESULTS_FILE
    if not rpath.exists():
        print(col(f"  [SKIP] {d.name}  — no results.json", GREY))
        continue
    with open(rpath) as f:
        runs[d.name] = json.load(f)

if not runs:
    sys.exit("No results.json files found.")

# ── Per-run index: task → entry ───────────────────────────────────────────────
# task names are assumed unique within a run
run_index: dict[str, dict[str, dict]] = {}
for run_name, entries in runs.items():
    run_index[run_name] = {e["task"]: e for e in entries}

all_task_names: set[str] = set()
for idx in run_index.values():
    all_task_names.update(idx.keys())

# ── Helper ────────────────────────────────────────────────────────────────────
def pct(n, d):
    return f"{100*n/d:.1f}%" if d else "N/A"

def mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")

# ── 1. Per-run summary ────────────────────────────────────────────────────────
header("1. Per-Run Summary")
col_w = max(len(n) for n in runs) + 2

print(f"  {'Run':<{col_w}}  {'Attempted':>10}  {'Completed':>10}  {'Accuracy':>10}  {'Avg Steps':>10}  {'Avg Time(s)':>12}  {'Avg Tokens':>11}")
print(f"  {'─'*col_w}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*11}")

run_stats: dict[str, dict] = {}
for run_name, entries in runs.items():
    attempted   = len(entries)
    completed   = sum(1 for e in entries if e.get("success"))
    acc         = completed / attempted if attempted else 0
    avg_steps   = mean([e.get("steps", 0) for e in entries])
    avg_time    = mean([e.get("time_s", 0) for e in entries])
    avg_tokens  = mean([e.get("token_totals", {}).get("total_tokens", 0) for e in entries])
    run_stats[run_name] = dict(attempted=attempted, completed=completed, acc=acc)

    acc_str = col(f"{100*acc:.1f}%", GREEN if acc >= 0.5 else RED)
    print(f"  {run_name:<{col_w}}  {attempted:>10}  {completed:>10}  {acc_str:>19}  {avg_steps:>10.1f}  {avg_time:>12.1f}  {avg_tokens:>11,.0f}")

# ── 2. Union & Intersection ───────────────────────────────────────────────────
header("2. Union / Intersection of Completed Tasks")

# Sets of tasks that succeeded in each run
success_sets: dict[str, set] = {
    run: {e["task"] for e in entries if e.get("success")}
    for run, entries in runs.items()
}

union_success      = set.union(*success_sets.values())
intersection_success = set.intersection(*success_sets.values())

# Union of all *attempted* tasks
union_attempted = all_task_names

subhdr("Union accuracy  (any run completed it)")
print(f"    Tasks attempted (union): {len(union_attempted)}")
print(f"    Tasks completed by ≥1 run: {len(union_success)}  ({pct(len(union_success), len(union_attempted))})")

subhdr("Intersection accuracy  (every run completed it)")
print(f"    Tasks completed by ALL {len(runs)} runs: {len(intersection_success)}  ({pct(len(intersection_success), len(union_attempted))})")

# ── 3. Per-task breakdown ─────────────────────────────────────────────────────
header("3. Per-Task Breakdown  (tasks attempted in ≥1 run)")

run_names = list(runs.keys())
short_names = [n.replace("dynamic lora qwen base", "DLQ") for n in run_names]

# Header row
pad_task = max(len(t) for t in all_task_names) + 2
hdr_cols = "  ".join(f"{s:^12}" for s in short_names)
print(f"  {'Task':<{pad_task}}  {hdr_cols}")
print(f"  {'─'*pad_task}  {'  '.join(['─'*12]*len(run_names))}")

# Categorise: all-pass / some-pass / all-fail
all_pass   = []
some_pass  = []
all_fail   = []

for task in sorted(all_task_names):
    results_per_run = []
    for run in run_names:
        entry = run_index[run].get(task)
        if entry is None:
            results_per_run.append(None)
        else:
            results_per_run.append(entry.get("success", False))

    successes = [r for r in results_per_run if r is True]
    attempted_count = sum(1 for r in results_per_run if r is not None)

    if successes and len(successes) == attempted_count and attempted_count == len(run_names):
        all_pass.append(task)
    elif successes:
        some_pass.append(task)
    else:
        all_fail.append(task)

    cells = []
    for r in results_per_run:
        if r is None:
            cells.append(col(f"{'N/A':^12}", GREY))
        elif r:
            cells.append(col(f"{'✓':^12}", GREEN))
        else:
            cells.append(col(f"{'✗':^12}", RED))

    row_col = GREEN if task in all_pass else (YELLOW if task in some_pass else "")
    print(f"  {col(f'{task:<{pad_task}}', row_col)}  {'  '.join(cells)}")

# Summary counts
print(f"\n  {col('✓ All runs passed:', GREEN)}  {len(all_pass):>4}  ({pct(len(all_pass), len(all_task_names))})")
print(f"  {col('~ Some runs passed:', YELLOW)}  {len(some_pass):>4}  ({pct(len(some_pass), len(all_task_names))})")
print(f"  {col('✗ No run passed:   ', RED)}  {len(all_fail):>4}  ({pct(len(all_fail), len(all_task_names))})")

# ── 4. Tasks only one run got right (exclusive wins) ─────────────────────────
header("4. Exclusive Wins  (tasks where only ONE run succeeded)")
for run in run_names:
    exclusive = success_sets[run] - set.union(*(success_sets[r] for r in run_names if r != run))
    print(f"\n  {col(run, MAGENTA)} — {len(exclusive)} exclusive win(s):")
    for t in sorted(exclusive):
        print(f"      {col('✓', GREEN)} {t}")
    if not exclusive:
        print(f"      {col('(none)', GREY)}")

# ── 5. Latency / token comparison ─────────────────────────────────────────────
header("5. Latency & Token Stats  (per run averages)")

lat_keys = ["inference_s", "decode_s", "step_total_s", "ttft_s", "tpot_ms", "action_s"]
print(f"  {'Metric':<20}  " + "  ".join(f"{n:>24}" for n in short_names))
print(f"  {'─'*20}  " + "  ".join(["─"*24]*len(run_names)))

for key in lat_keys:
    vals = []
    for run, entries in runs.items():
        v = mean([e.get("latency_avg", {}).get(key, 0) for e in entries if "latency_avg" in e])
        vals.append(v)
    row = "  ".join(f"{v:>24.3f}" for v in vals)
    print(f"  {key:<20}  {row}")

# token row
for tok_key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
    vals = []
    for run, entries in runs.items():
        v = mean([e.get("token_totals", {}).get(tok_key, 0) for e in entries if "token_totals" in e])
        vals.append(v)
    row = "  ".join(f"{v:>24,.0f}" for v in vals)
    print(f"  {tok_key:<20}  {row}")

# ── Done ──────────────────────────────────────────────────────────────────────
header("Done")
print(f"  Compared {len(runs)} run(s) across {len(all_task_names)} unique tasks.\n")
