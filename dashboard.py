"""
Eval-run failure-analysis dashboard.

    .venv/bin/python dashboard.py            # serves on http://localhost:8050
    .venv/bin/python dashboard.py --port 9000 # custom port
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
import socket
from collections import Counter
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

# ---------------------------------------------------------------------------
# Task step-budget lookup
#   RL  benchmark : max_steps = int(complexity * 15)   (run_aw_benchmark.py)
#   MobileAgent   : max_steps = int(complexity * 10)   (suite_utils._allocate_step_budget)
# Parsed from android_world task_eval source files via AST — no imports needed,
# handles inherited complexity attributes across the class hierarchy.
# ---------------------------------------------------------------------------
def _build_complexity_map() -> dict[str, float]:
    """Return {class_name: complexity_float} by AST-parsing all task_eval files."""
    task_root = os.path.expanduser(
        "~/Documents/MobileAgent/Mobile-Agent-v3.5/android_world_v3.5"
        "/android_world/task_evals"
    )
    py_files = glob.glob(os.path.join(task_root, "**", "*.py"), recursive=True)

    class_info: dict[str, tuple] = {}
    for fpath in py_files:
        try:
            tree = ast.parse(open(fpath).read())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = [
                b.id if isinstance(b, ast.Name) else
                (b.attr if isinstance(b, ast.Attribute) else None)
                for b in node.bases
            ]
            bases = [b for b in bases if b]
            complexity = None
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for tgt in item.targets:
                        if isinstance(tgt, ast.Name) and tgt.id == "complexity":
                            val = item.value
                            if isinstance(val, ast.Constant) and isinstance(
                                val.value, (int, float)
                            ):
                                complexity = float(val.value)
            class_info[node.name] = (complexity, bases)

    def _resolve(name, visited=None):
        if visited is None:
            visited = set()
        if name in visited or name not in class_info:
            return None
        visited.add(name)
        comp, bases = class_info[name]
        if comp is not None:
            return comp
        for base in bases:
            c = _resolve(base, visited)
            if c is not None:
                return c
        return None

    return {
        name: _resolve(name)
        for name in class_info
        if _resolve(name) is not None
    }


_TASK_COMPLEXITY: dict[str, float] = _build_complexity_map()

# RL benchmark uses ×15; MobileAgent uses ×10
TASK_MAX_STEPS_RL: dict[str, int] = {n: int(c * 15) for n, c in _TASK_COMPLEXITY.items()}
TASK_MAX_STEPS_MA: dict[str, int] = {n: int(c * 10) for n, c in _TASK_COMPLEXITY.items()}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
# Match config.yaml OUTPUT_DIR="./output" and run_aw_benchmark session layout.
_BASE_DIR_CANDIDATES = [
    os.path.join(_REPO_ROOT, "output", "aw_runs"),
    os.path.join(_REPO_ROOT, "output", "aw_runs_grid2level"),
    # Legacy paths (older machines / clones outside repo tree)
    os.path.expanduser("~/agentic_RL/output/aw_runs"),
    os.path.expanduser("~/agentic_RL/output/aw_runs_grid2level"),
]


def _unique_existing_base_dirs() -> list[str]:
    """De-dupe by realpath so repo + home paths that are the same dir only appear once."""
    seen: set[str] = set()
    out: list[str] = []
    for p in _BASE_DIR_CANDIDATES:
        ap = os.path.abspath(os.path.expanduser(p))
        if not os.path.isdir(ap):
            continue
        rp = os.path.realpath(ap)
        if rp in seen:
            continue
        seen.add(rp)
        out.append(rp)
    return out


BASE_DIRS = _unique_existing_base_dirs()
LOG_DIR = os.path.expanduser("~")
MIN_TASKS_FOR_DISPLAY = 50

BENCHMARK_LOG_MAP: dict[str, str] = {}

# Explicitly list the runs you want to display on the dashboard.
# If this set is empty, ALL valid runs will be displayed.
DISPLAY_RUNS = set([
    # "20260506_145814",
    "dynamic lora qwen base",
    "dynamic lora qwen base better run",
    "dynamic lora qwen base fix clear_text()",
])

# # Controls dropdown / ablation-table ordering (run_ids not listed sort to end)
# RUN_ORDER = [
#     "20260422_020351",  # Gemma-4-E4B
#     "20260408_233955",  # Grid 32x20
#     "20260409_181157",  # Element (committed prompt)
#     "20260410_055115",  # Grid 32x20 (committed prompt)
#     "20260411_005838",  # Grid sweep (unknown)
#     "20260411_175626",  # Grid 20x12
#     "20260412_053603",  # Grid 40x24
#     "20260413_090749",  # Hierarchical Grid
#     "20260413_224723",  # Raw Coords (unnormalized)
#     "20260414_062315",  # Raw Coords (normalized)
#     "20260416_061450",  # Raw Coords Normalized + Reasoning
#     "20260421_084144",  # Element + Escalate (first stall run)
#     "20260424_000816",  # Element + Escalate (stall-only prompt)
#     "20260424_091700",  # Element + Escalate (full stall prompt)
# ]
RUN_ORDER = []

# ---------------------------------------------------------------------------
# MobileAgent / android_world trajectory settings
# ---------------------------------------------------------------------------
MA_TRAJ_ROOT = os.path.expanduser(
    "~/Documents/MobileAgent/Mobile-Agent-v3.5/android_world_v3.5"
)
MA_MIN_TASKS = 10     # min classname-task dirs to show in the dashboard
MA_MAX_STEPS = 120    # step budget threshold for MobileAgent runs

# Optional human-readable labels for specific MA traj runs.
# Keys are traj dir names like "traj_2026-05-07_02-35-28".
MA_RUN_LABELS: dict[str, dict] = {
    "traj_2026-05-07_02-35-28": {
        "name": "MobileAgent v3.5 (full 116-task run with one model)",
        "model": "GUI-OWL-1.5 8B Instruct",
        "agent_mode": "raw",
        "stall_action": "None",
        "thinking": False,
        "notes": "baseline for mobileagent, only run on 93 runnable tasks, others failed due to import errors.",
    },
    "traj_2026-05-07_16-04-34": {
        "name": "MobileAgent v3.5 (post-fix run)",
        "model": "GUI-OWL-1.5 8B Instruct",
        "agent_mode": "raw",
        "stall_action": "None",
        "thinking": False,
        "notes": "run after fixing package name and sqlite errors",
    },
}

# Explicitly list the MA traj runs you want displayed.
# If empty, ALL traj runs with >= MA_MIN_TASKS tasks are shown.
MA_DISPLAY_RUNS: set[str] = set([
    "traj_2026-05-07_02-35-28",
    "traj_2026-05-07_16-04-34"
])

_MA_CLASSNAME_RE = re.compile(r"^[A-Z][A-Za-z0-9]+$")
MA_RUN_CACHE: dict[str, dict] = {}

ABLATION_LABELS: dict[str, dict] = {
    "dynamic lora qwen base": {
        "name": "Dynamic LoRA (Qwen Base)",
        "model": "Qwen3-VL-8B", 
        "agent_mode": "raw",
        "thinking": True, 
        "stall_action": "escalate",
        "notes": "first dynamic lora run"
    },
    "dynamic lora qwen base better run": {
        "name": "Dynamic LoRA (Qwen Base) Run 2",
        "model": "Qwen3-VL-8B", 
        "agent_mode": "raw",
        "thinking": True, 
        "stall_action": "escalate",
        "notes": "fixed text history issues"
    },
    "dynamic lora qwen base fix clear_text()": {
        "name": "Dynamic LoRA (Qwen Base) Run 3",
        "model": "Qwen3-VL-8B", 
        "agent_mode": "raw",
        "thinking": True, 
        "stall_action": "escalate",
        "notes": "fixed clear_text() tool call to work"
    },
}


def _get_run_label(run_id: str) -> str:
    if run_id in ABLATION_LABELS:
        return ABLATION_LABELS[run_id]["name"]
    return BENCHMARK_LOG_MAP.get(run_id, run_id)


# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------
STALL_SEVERE = 5
STALL_MODERATE = 3


def _is_budget_exhaustion(steps: int) -> bool:
    """max_steps = int(complexity * 15) for complexity in [1.0, 12.0]."""
    if steps < 15:
        return False
    for tenths in range(10, 121):
        if int(tenths * 1.5) == steps:
            return True
    return False


def classify_failure(task: dict) -> str:
    if task.get("success"):
        return "success"
    steps = task.get("steps", 0)
    if steps == 0:
        return "skipped"
    stall = task.get("max_stall_count", 0)
    if task.get("stall_terminated", False):
        return "stall_terminated"
    # Env verification passed but the agent never emitted FINISH / task_complete.
    if task.get("env_success") and not task.get("agent_done"):
        return "env_complete_no_finish"
    hit_budget = _is_budget_exhaustion(steps)
    if hit_budget and stall >= STALL_SEVERE:
        return "budget_exhaustion_with_stall"
    if hit_budget:
        return "budget_exhaustion"
    if stall >= STALL_SEVERE:
        return "severe_stall"
    if stall >= STALL_MODERATE:
        return "moderate_stall"
    if steps <= 5:
        return "premature_finish"
    return "other_failure"


FAILURE_LABELS = {
    "success": "Success",
    "skipped": "Skipped (env error)",
    "stall_terminated": "Stall-terminated",
    "env_complete_no_finish": "Env complete, no FINISH",
    "budget_exhaustion_with_stall": "Budget exhausted + stall",
    "budget_exhaustion": "Budget exhausted (no stall)",
    "severe_stall": "Severe stall (\u22655)",
    "moderate_stall": "Moderate stall (3-4)",
    "premature_finish": "Premature FINISH (\u22645 steps)",
    "other_failure": "Other failure",
    # MobileAgent-specific
    "ma_agent_fail": "MA: Agent claimed failure",
    "ma_coord_error": "MA: Coord hammering",
}

FAILURE_COLORS = {
    "success": "#22c55e",
    "skipped": "#94a3b8",
    "stall_terminated": "#ef4444",
    "env_complete_no_finish": "#0ea5e9",
    "budget_exhaustion_with_stall": "#f97316",
    "budget_exhaustion": "#eab308",
    "severe_stall": "#dc2626",
    "moderate_stall": "#fb923c",
    "premature_finish": "#a78bfa",
    "other_failure": "#64748b",
    # MobileAgent-specific
    "ma_agent_fail": "#fb7185",
    "ma_coord_error": "#fdba74",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _discover_log_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for logpath in glob.glob(os.path.join(LOG_DIR, "benchmark_*.log")):
        try:
            with open(logpath) as f:
                head = f.read(4096)
            m = re.search(r"Output directory: \./output/aw_runs(?:_grid2level)?/(\d{8}_\d{6})", head)
            if m:
                mapping[m.group(1)] = os.path.basename(logpath).replace(".log", "")
        except OSError:
            pass
    return mapping


def _success_count_from_tasks(tasks: list) -> int:
    """Count headline successes from each task row in results.json.

    Uses ``success`` when true. Otherwise counts explicit ``agent_success``
    (agent finished the episode and the env verified), then legacy fallbacks
    ``env_success`` and ``agent_done`` for older files missing ``agent_success``.
    """
    n = 0
    for t in tasks:
        if t.get("success"):
            n += 1
        elif t.get("agent_success") is True:
            n += 1
        elif t.get("env_success") and t.get("agent_done"):
            n += 1
    return n


def load_run(run_id: str, base_dir: str) -> dict | None:
    rfile = os.path.join(base_dir, run_id, "results.json")
    if not os.path.isfile(rfile):
        return None
    with open(rfile) as f:
        tasks = json.load(f)
    if not isinstance(tasks, list) or len(tasks) == 0:
        return None
    n = len(tasks)
    succ = _success_count_from_tasks(tasks)

    classifications = [classify_failure(t) for t in tasks]
    failure_counts = dict(Counter(classifications))

    for t, cls in zip(tasks, classifications):
        t["failure_class"] = cls
        t["failure_label"] = FAILURE_LABELS.get(cls, cls)
        t["max_steps"] = TASK_MAX_STEPS_RL.get(t.get("task", ""), 0)

    lat_keys = ["screenshot_s", "preprocess_s", "prompt_s", "inference_s",
                "action_s", "step_total_s", "ttft_s", "decode_s", "tpot_ms"]
    lat_avgs = {}
    for k in lat_keys:
        vals = [t["latency_avg"][k] for t in tasks
                if t.get("latency_avg") and k in t.get("latency_avg", {})]
        lat_avgs[k] = round(sum(vals) / len(vals), 4) if vals else 0

    total_prompt = sum(t.get("token_totals", {}).get("prompt_tokens", 0) for t in tasks)
    total_comp = sum(t.get("token_totals", {}).get("completion_tokens", 0) for t in tasks)

    meta = ABLATION_LABELS.get(run_id, {})
    label = _get_run_label(run_id)

    return {
        "run_id": run_id,
        "label": label,
        "model": meta.get("model", "Qwen3-VL-8B"),
        "agent_mode": meta.get("agent_mode", "element"),
        "thinking": meta.get("thinking", True),
        "stall_action": meta.get("stall_action", "none"),
        "notes": meta.get("notes", ""),
        "n_tasks": n,
        "n_success": succ,
        "accuracy": round(succ / n * 100, 1) if n else 0,
        "avg_steps": round(sum(t.get("steps", 0) for t in tasks) / n, 1),
        "total_wall_clock_s": round(sum(t.get("time_s", 0) for t in tasks)),
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_comp,
        "latency_avg": lat_avgs,
        "failure_counts": failure_counts,
        "tasks": tasks,
    }


def discover_runs() -> list[dict]:
    runs = []
    for base in BASE_DIRS:
        if not os.path.isdir(base):
            continue
        for entry in os.listdir(base):
            rpath = os.path.join(base, entry, "results.json")
            if os.path.isfile(rpath):
                runs.append({"run_id": entry, "base_dir": base, "source": "rl"})
    # MobileAgent traj directories
    if os.path.isdir(MA_TRAJ_ROOT):
        for name in sorted(os.listdir(MA_TRAJ_ROOT)):
            if name.startswith("traj_") and os.path.isdir(os.path.join(MA_TRAJ_ROOT, name)):
                runs.append({"run_id": name, "base_dir": MA_TRAJ_ROOT, "source": "ma"})
    order_idx = {rid: i for i, rid in enumerate(RUN_ORDER)}
    runs.sort(key=lambda r: (order_idx.get(r["run_id"], len(RUN_ORDER)), r["run_id"]))
    return runs


def _should_display(run_id: str, n_tasks: int, accuracy: float) -> bool:
    if DISPLAY_RUNS and run_id not in DISPLAY_RUNS:
        return False
    if n_tasks < MIN_TASKS_FOR_DISPLAY:
        return False
    if accuracy == 0.0 or accuracy == 100.0:
        return False
    return True


# ---------------------------------------------------------------------------
# MobileAgent data loading
# ---------------------------------------------------------------------------

def _ma_find_log(traj_name: str) -> str | None:
    ts = traj_name.replace("traj_", "")
    candidate = os.path.join(MA_TRAJ_ROOT, f"log_ma3_{ts}.log")
    if os.path.isfile(candidate):
        return candidate
    date = ts.split("_")[0]
    found = glob.glob(os.path.join(MA_TRAJ_ROOT, f"log_ma3_{date}_*.log"))
    return found[0] if found else None


def _ma_parse_final_summary(raw: str) -> dict:
    summary: dict = {}
    header_pat = re.compile(
        r"task_num\s+num_complete_trials\s+mean_success_rate"
        r"\s+mean_episode_length\s+total_runtime_s\s+num_fail_trials"
    )
    headers = list(header_pat.finditer(raw))
    if not headers:
        return summary
    table_text = raw[headers[-1].end():]
    row_pat = re.compile(
        r"^(\S.*?)\s+(\d+)\s+([\d.]+)\s+([\d.]+|NaN)\s+([\d.]+|NaN)"
        r"\s+([\d.]+|NaN)\s+([\d.]+)",
        re.MULTILINE,
    )
    for m in row_pat.finditer(table_text):
        name = m.group(1).strip()
        if name.startswith("========="):
            continue
        n_complete = float(m.group(3))
        mean_succ_raw, ep_len_raw, runtime_raw = m.group(4), m.group(5), m.group(6)
        n_fail = float(m.group(7))
        summary[name] = {
            "mean_success": None if mean_succ_raw == "NaN" else float(mean_succ_raw),
            "ep_len": None if ep_len_raw == "NaN" else float(ep_len_raw),
            "runtime_s": None if runtime_raw == "NaN" else float(runtime_raw),
            "n_complete": n_complete,
            "n_fail": n_fail,
            "skipped": n_complete == 0.0 and n_fail > 0,
        }
    return summary


def _ma_parse_log(log_path: str) -> dict:
    if not log_path or not os.path.isfile(log_path):
        return {}
    raw = open(log_path, errors="replace").read()
    raw = re.sub(r"\x1b\[[0-9;]*m", "", raw)

    final_summary = _ma_parse_final_summary(raw)
    parts = re.split(r"Running task: (\S+)\n", raw)
    lat_pat = re.compile(
        r"Step (\d+) Latency: Screenshot=([\d.]+)s, Preprocess=([\d.]+)s,"
        r" Inference=([\d.]+)s, Action=([\d.]+)s, Total=([\d.]+)s"
    )
    action_desc_pat = re.compile(r"^Action: (.+)$", re.MULTILINE)

    results = {name: dict(data, steps=[], goal="")
               for name, data in final_summary.items()}

    for i in range(1, len(parts) - 1, 2):
        name, block = parts[i], parts[i + 1]

        # Extract goal: greedy rfind to handle inner quotes and multiline goals
        goal = ""
        goal_prefix = f'Running task {name} with goal "'
        gp = block.find(goal_prefix)
        if gp != -1:
            gs = gp + len(goal_prefix)
            sm = block.find("\n==[", gs)
            if sm != -1:
                lq = block[gs:sm].rfind('"')
                if lq != -1:
                    goal = block[gs:gs + lq].strip()

        log_steps: list[dict] = []
        for m in lat_pat.finditer(block):
            log_steps.append({
                "step": int(m.group(1)),
                "screenshot_s": float(m.group(2)),
                "preprocess_s": float(m.group(3)),
                "inference_s": float(m.group(4)),
                "action_s": float(m.group(5)),
                "total_s": float(m.group(6)),
            })
        action_descs = action_desc_pat.findall(block)
        for idx, step in enumerate(log_steps):
            step["action_desc"] = action_descs[idx] if idx < len(action_descs) else ""

        if name in results:
            results[name]["steps"].extend(log_steps)
            if not results[name]["goal"] and goal:
                results[name]["goal"] = goal
        else:
            results[name] = {
                "mean_success": None, "ep_len": None, "runtime_s": None,
                "n_complete": None, "n_fail": None, "skipped": False,
                "steps": log_steps, "goal": goal,
            }
    return results


def _ma_avg_latency(log_steps: list) -> dict:
    def avg(key: str) -> float:
        vals = [s[key] for s in log_steps if key in s]
        return round(sum(vals) / len(vals), 4) if vals else 0.0
    return {
        "screenshot_s": avg("screenshot_s"),
        "preprocess_s": avg("preprocess_s"),
        "prompt_s": 0.0,
        "inference_s": avg("inference_s"),
        "action_s": avg("action_s"),
        "step_total_s": avg("total_s"),
        "ttft_s": 0.0,
        "decode_s": 0.0,
        "tpot_ms": 0.0,
    }


def _ma_classify_failure(actions: list, last_action: dict,
                         n_steps: int, success: bool, skipped: bool) -> str:
    if success:
        return "success"
    if skipped or n_steps == 0:
        return "skipped"
    if last_action.get("action") == "terminate" and last_action.get("status") == "failure":
        return "ma_agent_fail"
    if n_steps >= MA_MAX_STEPS:
        return "budget_exhaustion"
    # Coord hammering: last ≥4 coord actions all hit the same pixel
    coord_acts = [a for a in actions if a.get("action") in ("click", "swipe", "long_press")]
    if len(coord_acts) >= 4:
        coords = [tuple(a.get("coordinate", [])) for a in coord_acts[-6:] if a.get("coordinate")]
        if len(coords) >= 4 and len(set(coords)) == 1:
            return "ma_coord_error"
    return "other_failure"


def load_ma_run(traj_name: str) -> dict | None:
    if traj_name in MA_RUN_CACHE:
        return MA_RUN_CACHE[traj_name]

    traj_path = os.path.join(MA_TRAJ_ROOT, traj_name)
    if not os.path.isdir(traj_path):
        return None

    log_data = _ma_parse_log(_ma_find_log(traj_name))

    # Build reverse map: goal-text dir name → class name.
    # Some tasks (e.g. information-retrieval QA tasks) are saved under a
    # directory named after the goal string ("Do_I_have_any_events_...") rather
    # than the class name ("CalendarCheck...") because the random goal at
    # runtime didn't match instances[0].goal stored in agent.task_name.
    goal_dir_to_class: dict[str, str] = {}
    for class_name, ld in log_data.items():
        goal = ld.get("goal", "")
        if goal:
            goal_dir_to_class[goal.replace(" ", "_")[:50]] = class_name

    tasks: list[dict] = []

    for dir_name in sorted(os.listdir(traj_path)):
        task_dir = os.path.join(traj_path, dir_name)
        jsonl = os.path.join(task_dir, "action.jsonl")
        if not os.path.isdir(task_dir) or not os.path.isfile(jsonl):
            continue

        # Resolve the filesystem dir name to a canonical class name.
        if _MA_CLASSNAME_RE.match(dir_name):
            task_name = dir_name          # already a proper class name
        elif dir_name in goal_dir_to_class:
            task_name = goal_dir_to_class[dir_name]  # goal-text → class name
        elif dir_name in log_data:
            task_name = dir_name          # found directly in log
        else:
            continue                      # unrecognisable dir, skip

        with open(jsonl) as f:
            raw_actions = [json.loads(ln) for ln in f if ln.strip()]
        actions = [a.get("arguments", {}) for a in raw_actions]
        n_steps = len(actions)
        last_action = actions[-1] if actions else {}
        def _png_key(fn):
            m = re.search(r"(\d+)", fn)
            return int(m.group(1)) if m else fn
        screenshots = sorted(
            (fn for fn in os.listdir(task_dir) if fn.endswith(".png")),
            key=_png_key,
        )

        ld = log_data.get(task_name, {})
        mean_success = ld.get("mean_success")
        skipped = ld.get("skipped", False)
        runtime_s = ld.get("runtime_s") or 0.0
        ep_len = int(ld.get("ep_len") or n_steps)
        log_steps = ld.get("steps", [])
        goal = ld.get("goal", "")

        if skipped:
            success = False
        elif mean_success is not None:
            success = mean_success >= 0.5
        else:
            success = (last_action.get("action") == "terminate"
                       and last_action.get("status") == "success")

        fc = _ma_classify_failure(actions, last_action, n_steps, success, skipped)
        tasks.append({
            "task": task_name,      # canonical class name (for display & lookup)
            "goal": goal,
            "success": success,
            "steps": ep_len,
            "max_steps": TASK_MAX_STEPS_MA.get(task_name, 0),
            "time_s": runtime_s,
            "max_stall_count": 0,
            "stall_terminated": False,
            "latency_avg": _ma_avg_latency(log_steps),
            "token_totals": {"prompt_tokens": 0, "completion_tokens": 0},
            "combo": 0,
            "source": "ma",
            "failure_class": fc,
            "failure_label": FAILURE_LABELS.get(fc, fc),
            "_traj_name": traj_name,
            "_dir_name": dir_name,  # actual filesystem dir (may differ from task_name)
            "_screenshots": screenshots,
            "_log_steps": log_steps,
            "_actions": actions,
        })

    # Add log-only entries (skipped / env-error tasks with no traj dir)
    traj_names = {t["task"] for t in tasks}
    for task_name, ld in sorted(log_data.items()):
        if task_name in traj_names:
            continue
        skipped = ld.get("skipped", False)
        mean_success = ld.get("mean_success")
        success = (not skipped) and (mean_success is not None) and (mean_success >= 0.5)
        fc = "skipped" if skipped else ("success" if success else "other_failure")
        tasks.append({
            "task": task_name,
            "goal": ld.get("goal", ""),
            "success": success,
            "steps": int(ld.get("ep_len") or 0),
            "max_steps": TASK_MAX_STEPS_MA.get(task_name, 0),
            "time_s": ld.get("runtime_s") or 0.0,
            "max_stall_count": 0,
            "stall_terminated": False,
            "latency_avg": _ma_avg_latency(ld.get("steps", [])),
            "token_totals": {"prompt_tokens": 0, "completion_tokens": 0},
            "combo": 0,
            "source": "ma",
            "failure_class": fc,
            "failure_label": FAILURE_LABELS.get(fc, fc),
            "_traj_name": traj_name,
            "_screenshots": [],
            "_log_steps": [],
            "_actions": [],
        })

    tasks.sort(key=lambda t: t["task"])
    n = len(tasks)
    runnable = [t for t in tasks if t["failure_class"] != "skipped"]
    n_runnable = len(runnable)
    n_success = sum(1 for t in runnable if t["success"])
    accuracy = round(n_success / n_runnable * 100, 1) if n_runnable else 0.0

    failure_counts = dict(Counter(t["failure_class"] for t in tasks))
    lat_keys = ["screenshot_s", "preprocess_s", "prompt_s", "inference_s",
                "action_s", "step_total_s", "ttft_s", "decode_s", "tpot_ms"]
    lat_avgs: dict = {}
    for k in lat_keys:
        vals = [t["latency_avg"].get(k, 0) for t in runnable if t["latency_avg"].get(k, 0) > 0]
        lat_avgs[k] = round(sum(vals) / len(vals), 4) if vals else 0.0

    meta = MA_RUN_LABELS.get(traj_name, {})
    label = meta.get("name", traj_name.replace("traj_", ""))
    result = {
        "run_id": traj_name,
        "label": label,
        "source": "ma",
        "model": meta.get("model", "GUI-OWL"),
        "agent_mode": "ma",
        "thinking": False,
        "stall_action": "none",
        "notes": meta.get("notes", ""),
        "n_tasks": n,
        "n_runnable": n_runnable,
        "n_success": n_success,
        "accuracy": accuracy,
        "avg_steps": round(sum(t["steps"] for t in runnable) / n_runnable, 1) if n_runnable else 0,
        "total_wall_clock_s": round(sum(t["time_s"] for t in tasks)),
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "latency_avg": lat_avgs,
        "failure_counts": failure_counts,
        "tasks": tasks,
    }
    MA_RUN_CACHE[traj_name] = result
    return result


def _ma_steps_response(traj_name: str, task_name: str):
    run = load_ma_run(traj_name)
    if run is None:
        return JSONResponse({"steps": [], "goal": ""})
    task = next((t for t in run["tasks"] if t["task"] == task_name), None)
    if task is None:
        return JSONResponse({"steps": [], "goal": ""})

    screenshots = task.get("_screenshots", [])
    log_steps = task.get("_log_steps", [])
    actions = task.get("_actions", [])
    # Use the actual filesystem dir name (may differ from canonical task_name
    # when the agent saved the task under a goal-text directory).
    dir_name = task.get("_dir_name", task_name)
    n = max(len(screenshots), len(actions))
    steps = []
    for i in range(n):
        img = screenshots[i] if i < len(screenshots) else None
        action_d = actions[i] if i < len(actions) else {}
        ls = log_steps[i] if i < len(log_steps) else {}
        act_type = action_d.get("action", "")
        coord = action_d.get("coordinate")
        text_v = str(action_d.get("text") or action_d.get("status") or "")[:80]
        action_str = act_type + (f" {coord}" if coord else (f" {text_v}" if text_v else ""))
        steps.append({
            "step": i + 1,
            "images": {"main": f"/api/img/{traj_name}/{dir_name}/{img}"} if img else {},
            "action": action_str,
            "summary": ls.get("action_desc", ""),
            "screen_diff": None,
            "stall_count": 0,
        })
    return JSONResponse({"steps": steps, "goal": task.get("goal", "")})


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    global BENCHMARK_LOG_MAP
    BENCHMARK_LOG_MAP = _discover_log_map()
    yield

app = FastAPI(title="Eval Failure Dashboard", lifespan=lifespan)


@app.middleware("http")
async def _no_store_dashboard_api(request, call_next):
    """Avoid stale /api/* JSON when the server or dashboard code is updated."""
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/api/") and not path.startswith("/api/img/"):
        response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


@app.get("/api/runs")
def api_runs():
    entries = discover_runs()
    summaries = []
    for e in entries:
        if e["source"] == "ma":
            if MA_DISPLAY_RUNS and e["run_id"] not in MA_DISPLAY_RUNS:
                continue
            # Quick pre-filter without loading the log
            try:
                traj_path = os.path.join(MA_TRAJ_ROOT, e["run_id"])
                n = sum(1 for d in os.listdir(traj_path)
                        if _MA_CLASSNAME_RE.match(d)
                        and os.path.isdir(os.path.join(traj_path, d))
                        and os.path.isfile(os.path.join(traj_path, d, "action.jsonl")))
            except OSError:
                continue
            if n < MA_MIN_TASKS:
                continue
            meta = MA_RUN_LABELS.get(e["run_id"], {})
            label = meta.get("name", e["run_id"].replace("traj_", ""))
            summaries.append({
                "run_id": e["run_id"],
                "source": "ma",
                "label": label,
                "n_tasks": n,
                "accuracy": None,   # computed on first full load
            })
        else:
            rfile = os.path.join(e["base_dir"], e["run_id"], "results.json")
            try:
                with open(rfile) as f:
                    tasks = json.load(f)
                n = len(tasks)
                TASK_SUCCESSFUL_OVERRIDES = {"20260424_091700": 45}
                succ = TASK_SUCCESSFUL_OVERRIDES.get(e["run_id"],
                           sum(1 for t in tasks if t.get("success")))
            except Exception:
                n, succ = 0, 0
            acc = round(succ / n * 100, 1) if n else 0
            if not _should_display(e["run_id"], n, acc):
                continue
            label = _get_run_label(e["run_id"])
            summaries.append({
                "run_id": e["run_id"],
                "source": "rl",
                "label": label,
                "n_tasks": n,
                "accuracy": acc,
            })
    return JSONResponse(summaries)


@app.get("/api/run/{run_id}")
def api_run(run_id: str):
    if run_id.startswith("traj_"):
        data = load_ma_run(run_id)
        if data is not None:
            return JSONResponse(data)
        return JSONResponse({"error": "run not found"}, status_code=404)
    for base in BASE_DIRS:
        data = load_run(run_id, base)
        if data is not None:
            return JSONResponse(data)
    return JSONResponse({"error": "run not found"}, status_code=404)


@app.get("/api/compare")
def api_compare(ids: str = ""):
    if not ids:
        return JSONResponse([])
    run_ids = [r.strip() for r in ids.split(",") if r.strip()]
    results = []
    for rid in run_ids:
        if rid.startswith("traj_"):
            data = load_ma_run(rid)
        else:
            data = None
            for base in BASE_DIRS:
                data = load_run(rid, base)
                if data is not None:
                    break
        if data is not None:
            d = {k: v for k, v in data.items() if k != "tasks"}
            results.append(d)
    return JSONResponse(results)


@app.get("/api/ablation_table")
def api_ablation_table():
    """Return all qualifying runs as a flat ablation-summary table."""
    entries = discover_runs()
    rows = []
    for e in entries:
        if e["source"] == "ma":
            if MA_DISPLAY_RUNS and e["run_id"] not in MA_DISPLAY_RUNS:
                continue
            try:
                traj_path = os.path.join(MA_TRAJ_ROOT, e["run_id"])
                n_class = sum(1 for d in os.listdir(traj_path)
                              if _MA_CLASSNAME_RE.match(d)
                              and os.path.isdir(os.path.join(traj_path, d))
                              and os.path.isfile(os.path.join(traj_path, d, "action.jsonl")))
            except OSError:
                continue
            if n_class < MA_MIN_TASKS:
                continue
            try:
                data = load_ma_run(e["run_id"])
            except Exception:
                continue
            if data is None:
                continue
            rows.append({
                "run_id": data["run_id"],
                "source": "ma",
                "experiment": data["label"],
                "model": data["model"],
                "agent_mode": "ma",
                "thinking": False,
                "stall_action": "none",
                "notes": data.get("notes", ""),
                "accuracy": data["accuracy"],
                "avg_steps": data["avg_steps"],
                "avg_comp_tokens": 0,
                "avg_prompt_tokens": 0,
                "total_wall_clock_h": round(data["total_wall_clock_s"] / 3600, 2),
                "avg_step_total_s": data["latency_avg"].get("step_total_s", 0),
                "avg_screenshot_s": data["latency_avg"].get("screenshot_s", 0),
                "avg_preprocess_s": data["latency_avg"].get("preprocess_s", 0),
                "avg_inference_s": data["latency_avg"].get("inference_s", 0),
                "avg_action_s": data["latency_avg"].get("action_s", 0),
                "avg_ttft_s": 0,
                "avg_decode_s": 0,
                "avg_tpot_ms": 0,
            })
        else:
            for base in BASE_DIRS:
                data = load_run(e["run_id"], base)
                if data is None:
                    continue
                if not _should_display(e["run_id"], data["n_tasks"], data["accuracy"]):
                    break
                rows.append({
                    "run_id": data["run_id"],
                    "source": "rl",
                    "experiment": data["label"],
                    "model": data["model"],
                    "agent_mode": data["agent_mode"],
                    "thinking": data["thinking"],
                    "stall_action": data["stall_action"],
                    "notes": data.get("notes", ""),
                    "accuracy": data["accuracy"],
                    "avg_steps": data["avg_steps"],
                    "avg_comp_tokens": round(data["total_completion_tokens"] / data["n_tasks"]) if data["n_tasks"] else 0,
                    "avg_prompt_tokens": round(data["total_prompt_tokens"] / data["n_tasks"]) if data["n_tasks"] else 0,
                    "total_wall_clock_h": round(data["total_wall_clock_s"] / 3600, 2),
                    "avg_step_total_s": data["latency_avg"].get("step_total_s", 0),
                    "avg_screenshot_s": data["latency_avg"].get("screenshot_s", 0),
                    "avg_preprocess_s": data["latency_avg"].get("preprocess_s", 0),
                    "avg_inference_s": data["latency_avg"].get("inference_s", 0),
                    "avg_action_s": data["latency_avg"].get("action_s", 0),
                    "avg_ttft_s": data["latency_avg"].get("ttft_s", 0),
                    "avg_decode_s": data["latency_avg"].get("decode_s", 0),
                    "avg_tpot_ms": data["latency_avg"].get("tpot_ms", 0),
                })
                break
    order_idx = {rid: i for i, rid in enumerate(RUN_ORDER)}
    rows.sort(key=lambda r: (order_idx.get(r["run_id"], len(RUN_ORDER)), r["run_id"]))
    return JSONResponse(rows)


@app.get("/api/steps/{run_id}/{task_name}")
def api_steps(run_id: str, task_name: str, combo: int = 0):
    """Return step images and (if available) parsed actions for a task."""
    if run_id.startswith("traj_"):
        return _ma_steps_response(run_id, task_name)
    task_dir_name = f"{task_name}_combo{combo}"
    task_dir = None
    for base in BASE_DIRS:
        candidate = os.path.join(base, run_id, task_dir_name)
        if os.path.isdir(candidate):
            task_dir = candidate
            break
    if task_dir is None:
        return JSONResponse({"steps": []})

    images = sorted(
        (f for f in os.listdir(task_dir) if f.endswith(".png")),
        key=lambda fn: int(m.group(1)) if (m := re.search(r"(\d+)", fn)) else fn,
    )
    step_map: dict[int, dict] = {}
    for img in images:
        m = re.match(r"step_(\d+)(?:_(coarse|fine))?\.png", img)
        if not m:
            continue
        step_num = int(m.group(1))
        variant = m.group(2) or "main"
        if step_num not in step_map:
            step_map[step_num] = {"step": step_num, "images": {}}
        step_map[step_num]["images"][variant] = f"/api/img/{run_id}/{task_dir_name}/{img}"

    actions = _parse_actions_from_log(run_id, task_name)

    steps = []
    for step_num in sorted(step_map):
        entry = step_map[step_num]
        if step_num in actions:
            entry.update(actions[step_num])
        steps.append(entry)
    return JSONResponse({"steps": steps})


def _parse_actions_from_log(run_id: str, task_name: str) -> dict[int, dict]:
    """Best-effort parse of Observation/Thought/Action/Summary per step from
    the benchmark log that matches this run_id."""
    log_name = BENCHMARK_LOG_MAP.get(run_id)
    if not log_name:
        return {}
    log_path = os.path.join(LOG_DIR, f"{log_name}.log")
    if not os.path.isfile(log_path):
        return {}
    try:
        with open(log_path) as f:
            raw = f.read()
    except OSError:
        return {}
    clean = re.sub(r"\x1b\[[0-9;]*m", "", raw)

    start = clean.find(f"[{task_name}]")
    if start < 0:
        return {}
    next_task = clean.find("\n[", start + len(task_name) + 10)
    while next_task > 0:
        after = clean[next_task + 2: next_task + 80]
        if re.match(r"[A-Z][A-Za-z]", after) and "] (combo" in after:
            break
        next_task = clean.find("\n[", next_task + 2)
    section = clean[start:next_task] if next_task > 0 else clean[start:]

    result: dict[int, dict] = {}
    step_pat = re.compile(r"=+\[step (\d+)\].*?=+")
    parts = step_pat.split(section)
    for i in range(1, len(parts) - 1, 2):
        step_num = int(parts[i])
        block = parts[i + 1]
        info: dict[str, str] = {}
        for field in ("Observation", "Thought", "Action", "Summary"):
            m = re.search(rf"^{field}:\s*(.+?)(?=\n(?:Observation|Thought|Action|Summary|  \[)|$)",
                          block, re.MULTILINE | re.DOTALL)
            if m:
                info[field.lower()] = m.group(1).strip()
        diff_m = re.search(r"diff=([\d.]+)\s+stall_count=(\d+)", block)
        if diff_m:
            info["screen_diff"] = float(diff_m.group(1))
            info["stall_count"] = int(diff_m.group(2))
        result[step_num] = info
    return result


@app.get("/api/img/{run_id}/{task_dir}/{filename}")
def api_img(run_id: str, task_dir: str, filename: str):
    """Serve a step screenshot image."""
    if ".." in filename or "/" in filename:
        return JSONResponse({"error": "invalid"}, status_code=400)
    # MobileAgent traj images
    if run_id.startswith("traj_"):
        path = os.path.join(MA_TRAJ_ROOT, run_id, task_dir, filename)
        if os.path.isfile(path):
            return FileResponse(path, media_type="image/png")
        return JSONResponse({"error": "not found"}, status_code=404)
    # agentic_RL images
    for base in BASE_DIRS:
        path = os.path.join(base, run_id, task_dir, filename)
        if os.path.isfile(path):
            return FileResponse(path, media_type="image/png")
    return JSONResponse({"error": "not found"}, status_code=404)


@app.get("/", response_class=HTMLResponse)
def index():
    return DASHBOARD_HTML


# ---------------------------------------------------------------------------
# Inline HTML/CSS/JS
# ---------------------------------------------------------------------------
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Cache-Control" content="no-cache">
<title>Eval Failure Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#f8fafc;--surface:#ffffff;--border:#e2e8f0;--text:#1e293b;--text2:#64748b;--accent:#2563eb;--green:#16a34a;--red:#dc2626;--orange:#ea580c;--yellow:#ca8a04}
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',system-ui,-apple-system,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
a{color:var(--accent);text-decoration:none}

.top-bar{background:var(--surface);border-bottom:1px solid var(--border);padding:12px 24px;display:flex;align-items:center;gap:16px;position:sticky;top:0;z-index:50}
.top-bar h1{font-size:18px;font-weight:600;white-space:nowrap}
.top-bar select{background:var(--bg);color:var(--text);border:1px solid var(--border);border-radius:6px;padding:6px 12px;font-size:14px;min-width:340px}
.btn{border:none;border-radius:6px;padding:6px 16px;cursor:pointer;font-size:13px;font-weight:500;color:#fff}
.btn:hover{opacity:.85}
.btn-accent{background:var(--accent)}
.btn-purple{background:#7c3aed}

.container{max-width:1440px;margin:0 auto;padding:20px 24px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:20px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px}
.card .label{font-size:12px;color:var(--text2);text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px}
.card .value{font-size:26px;font-weight:700}
.card .sub{font-size:12px;color:var(--text2);margin-top:2px}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:20px}
.chart-box{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px}
.chart-box h3{font-size:14px;font-weight:600;margin-bottom:12px}
.chart-box canvas{width:100%!important;max-height:320px}

.tbl-wrap{overflow-x:auto;background:var(--surface);border:1px solid var(--border);border-radius:10px;margin-bottom:20px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;padding:8px 10px;background:var(--surface);border-bottom:2px solid var(--border);font-weight:600;position:sticky;top:0;z-index:10;cursor:pointer;user-select:none;white-space:nowrap}
th:hover{color:var(--accent)}
th .arrow{font-size:10px;margin-left:3px;color:var(--text2)}
td{padding:7px 10px;border-bottom:1px solid var(--border);white-space:nowrap}
tr:hover td{background:rgba(37,99,235,.06)}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:11px;font-weight:600;color:#fff}
.filter-row{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
.filter-btn{background:var(--surface);border:1px solid var(--border);color:var(--text2);padding:4px 12px;border-radius:16px;font-size:12px;cursor:pointer}
.filter-btn.active{border-color:var(--accent);color:var(--accent);background:rgba(37,99,235,.08)}
.hidden{display:none}
.section-title{font-size:16px;font-weight:600;margin:20px 0 12px}

.tab-bar{display:flex;gap:0;margin-bottom:0;border-bottom:2px solid var(--border)}
.tab{padding:10px 20px;cursor:pointer;font-size:14px;font-weight:500;color:var(--text2);border-bottom:2px solid transparent;margin-bottom:-2px}
.tab.active{color:var(--accent);border-bottom-color:var(--accent)}
.tab:hover{color:var(--text)}

.acc-bar{display:inline-block;height:16px;border-radius:3px;min-width:2px;vertical-align:middle}
td.num{text-align:right;font-variant-numeric:tabular-nums}
.row-highlight td{background:rgba(22,163,74,.06)}

.step-viewer{padding:16px 10px;background:var(--bg);border-bottom:2px solid var(--border)}
.step-viewer .steps-scroll{display:flex;gap:16px;overflow-x:auto;padding-bottom:12px}
.step-card{flex:0 0 280px;background:var(--surface);border:1px solid var(--border);border-radius:8px;overflow:hidden}
.step-card img{width:280px;height:auto;display:block;cursor:pointer}
.step-card .step-info{padding:8px 10px;font-size:12px}
.step-card .step-num{font-weight:700;color:var(--accent);margin-bottom:4px}
.step-card .step-action{color:var(--text);font-family:'SF Mono',Consolas,monospace;font-size:11px;background:var(--bg);padding:4px 6px;border-radius:4px;margin:4px 0;word-break:break-all}
.step-card .step-summary{color:var(--text2);font-size:11px;line-height:1.4}
.step-card .step-meta{color:var(--text2);font-size:10px;margin-top:4px}
.step-card .stall-badge{color:#dc2626;font-weight:600}
.step-viewer .close-btn{float:right;background:none;border:1px solid var(--border);color:var(--text2);border-radius:4px;padding:2px 8px;cursor:pointer;font-size:12px}
.step-viewer .close-btn:hover{background:var(--border)}
.step-viewer .viewer-title{font-weight:600;font-size:14px;margin-bottom:6px}
.step-viewer .viewer-goal{font-size:12px;color:var(--text);background:rgba(37,99,235,.06);border:1px solid var(--border);border-radius:6px;padding:8px 12px;margin-bottom:10px;line-height:1.5;white-space:pre-wrap;word-break:break-word}
.source-badge{display:inline-block;padding:2px 7px;border-radius:4px;font-size:11px;font-weight:700;line-height:1.6}
.source-badge.ma{background:#6366f1;color:#fff}
.source-badge.rl{background:#22c55e;color:#fff}

.lightbox{position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,.85);z-index:100;display:flex;align-items:center;justify-content:center;cursor:zoom-out}
.lightbox img{max-width:95vw;max-height:95vh;border-radius:8px}

tr.expandable{cursor:pointer}
tr.expandable:hover td:first-child{text-decoration:underline}

@media(max-width:900px){.charts{grid-template-columns:1fr}}
</style>
</head>
<body>

<div class="top-bar">
  <h1>Eval Failure Dashboard</h1>
  <select id="run-select"><option value="">Loading runs…</option></select>
  <button class="btn btn-accent" onclick="toggleCompare()">Compare Runs</button>
  <button class="btn btn-purple" onclick="switchTab('ablation')">Ablation Table</button>
  <button class="btn" style="background:#16a34a" onclick="exportToCSV()">Export to CSV</button>
</div>

<div class="container">

  <!-- TAB: Per-run analysis -->
  <div id="tab-run">
    <div class="cards" id="kpi-cards"></div>
    <div class="charts">
      <div class="chart-box"><h3>Failure Mode Breakdown</h3><canvas id="failureChart"></canvas></div>
      <div class="chart-box"><h3>Per-Step Time Breakdown (avg seconds)</h3><canvas id="latencyChart"></canvas></div>
    </div>
    <div id="compare-panel" class="hidden">
      <div class="section-title">Cross-Run Comparison</div>
      <div class="chart-box"><canvas id="compareChart"></canvas></div>
    </div>
    <div class="section-title">Per-Task Results</div>
    <div class="filter-row" id="filters"></div>
    <div class="tbl-wrap">
      <table>
        <thead><tr>
          <th data-key="task">Task</th>
          <th data-key="success">Result</th>
          <th data-key="failure_label">Failure Mode</th>
          <th data-key="steps" data-num>Steps</th>
          <th data-key="max_steps" data-num>Budget</th>
          <th data-key="max_stall_count" data-num>Max Stall</th>
          <th data-key="time_s" data-num>Time (s)</th>
          <th data-key="prompt_tokens" data-num>Prompt Tok</th>
          <th data-key="completion_tokens" data-num>Comp Tok</th>
          <th data-key="inference_s" data-num>Inference (s)</th>
          <th data-key="step_total_s" data-num>Step Total (s)</th>
        </tr></thead>
        <tbody id="task-tbody"></tbody>
      </table>
    </div>
  </div>

  <!-- TAB: Ablation summary table -->
  <div id="tab-ablation" class="hidden">
    <div class="section-title">Ablation Summary</div>
    <div class="tbl-wrap">
      <table id="ablation-tbl">
        <thead><tr>
          <th data-abl="source">Source</th>
          <th data-abl="experiment">Experiment</th>
          <th data-abl="model">Model</th>
          <th data-abl="agent_mode">Mode</th>
          <th data-abl="thinking">Thinking</th>
          <th data-abl="stall_action">Stall</th>
          <th data-abl="notes">Notes</th>
          <th data-abl="accuracy" data-num>Accuracy %</th>
          <th data-abl="avg_steps" data-num>Avg Steps</th>
          <th data-abl="avg_comp_tokens" data-num>Avg Decode Tok</th>
          <th data-abl="avg_prompt_tokens" data-num>Avg Prefill Tok</th>
          <th data-abl="total_wall_clock_h" data-num>Wall Clock (h)</th>
          <th data-abl="avg_step_total_s" data-num>Step Total (s)</th>
          <th data-abl="avg_screenshot_s" data-num>Screenshot (s)</th>
          <th data-abl="avg_preprocess_s" data-num>Preprocess (s)</th>
          <th data-abl="avg_inference_s" data-num>Inference (s)</th>
          <th data-abl="avg_action_s" data-num>Action (s)</th>
          <th data-abl="avg_ttft_s" data-num>TTFT (s)</th>
          <th data-abl="avg_decode_s" data-num>Decode (s)</th>
          <th data-abl="avg_tpot_ms" data-num>TPOT (ms)</th>
        </tr></thead>
        <tbody id="ablation-tbody"></tbody>
      </table>
    </div>
  </div>

</div>

<script>
const COLORS = {
  success:"#22c55e",skipped:"#94a3b8",stall_terminated:"#ef4444",
  env_complete_no_finish:"#0ea5e9",
  budget_exhaustion_with_stall:"#f97316",budget_exhaustion:"#eab308",
  severe_stall:"#dc2626",moderate_stall:"#fb923c",
  premature_finish:"#a78bfa",other_failure:"#64748b",
  ma_agent_fail:"#fb7185",ma_coord_error:"#fdba74"
};
const LABELS = {
  success:"Success",skipped:"Skipped",stall_terminated:"Stall-terminated",
  env_complete_no_finish:"Env complete, no FINISH",
  budget_exhaustion_with_stall:"Budget + stall",budget_exhaustion:"Budget exhausted",
  severe_stall:"Severe stall (>=5)",moderate_stall:"Moderate stall (3-4)",
  premature_finish:"Premature FINISH",other_failure:"Other failure",
  ma_agent_fail:"MA: Agent claimed failure",ma_coord_error:"MA: Coord hammering"
};

let allRuns = [];
let currentRun = null;
let failureChart = null, latencyChart = null, compareChart = null;
let activeFilter = null;
let sortKey = null, sortAsc = true;
let ablSortKey = 'accuracy', ablSortAsc = false;
let ablationData = null;
let currentTab = 'run';

function switchTab(tab) {
  currentTab = tab;
  document.getElementById('tab-run').classList.toggle('hidden', tab !== 'run');
  document.getElementById('tab-ablation').classList.toggle('hidden', tab !== 'ablation');
  if (tab === 'ablation' && !ablationData) loadAblation();
}

// ---- Ablation table ----
async function loadAblation() {
  const res = await fetch('/api/ablation_table', { cache: 'no-store' });
  ablationData = await res.json();
  renderAblation();
}

function renderAblation() {
  if (!ablationData) return;
  let data = [...ablationData];
  if (ablSortKey) {
    data.sort((a,b) => {
      let va = a[ablSortKey], vb = b[ablSortKey];
      if (typeof va === 'string') return ablSortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
      if (typeof va === 'boolean') { va = va?1:0; vb = vb?1:0; }
      return ablSortAsc ? va-vb : vb-va;
    });
  }
  const maxAcc = Math.max(...data.map(d=>d.accuracy), 1);
  const tbody = document.getElementById('ablation-tbody');
  tbody.innerHTML = data.map(r => {
    const barW = Math.round(r.accuracy / maxAcc * 100);
    const barColor = r.accuracy >= 25 ? '#22c55e' : r.accuracy >= 15 ? '#eab308' : '#ef4444';
    const srcBadge = r.source === 'ma'
      ? '<span class="source-badge ma">MA</span>'
      : '<span class="source-badge rl">RL</span>';
    const isMa = r.source === 'ma';
    return `<tr class="${r.accuracy >= 30 ? 'row-highlight':''}" style="cursor:pointer" onclick="loadRunFromAblation('${r.run_id}')">
      <td>${srcBadge}</td>
      <td><strong>${r.experiment}</strong></td>
      <td>${r.model}</td>
      <td>${r.agent_mode}</td>
      <td>${r.thinking ? 'Yes' : 'No'}</td>
      <td>${r.stall_action}</td>
      <td>${r.notes || ''}</td>
      <td class="num"><span class="acc-bar" style="width:${barW}px;background:${barColor}"></span> ${r.accuracy}%</td>
      <td class="num">${r.avg_steps}</td>
      <td class="num">${isMa ? '—' : r.avg_comp_tokens.toLocaleString()}</td>
      <td class="num">${isMa ? '—' : r.avg_prompt_tokens.toLocaleString()}</td>
      <td class="num">${r.total_wall_clock_h}</td>
      <td class="num">${r.avg_step_total_s}</td>
      <td class="num">${r.avg_screenshot_s}</td>
      <td class="num">${r.avg_preprocess_s}</td>
      <td class="num">${r.avg_inference_s}</td>
      <td class="num">${r.avg_action_s}</td>
      <td class="num">${isMa ? '—' : r.avg_ttft_s}</td>
      <td class="num">${isMa ? '—' : r.avg_decode_s}</td>
      <td class="num">${isMa ? '—' : r.avg_tpot_ms}</td>
    </tr>`;
  }).join('');
}

function loadRunFromAblation(runId) {
  switchTab('run');
  const sel = document.getElementById('run-select');
  if (sel) { sel.value = runId; }
  loadRun(runId);
}

document.querySelectorAll('th[data-abl]').forEach(th => {
  th.addEventListener('click', () => {
    const key = th.dataset.abl;
    if (ablSortKey === key) ablSortAsc = !ablSortAsc;
    else { ablSortKey = key; ablSortAsc = !th.hasAttribute('data-num'); }
    renderAblation();
  });
});

// ---- Per-run analysis ----
async function init() {
  const res = await fetch('/api/runs', { cache: 'no-store' });
  allRuns = await res.json();
  const sel = document.getElementById('run-select');
  sel.innerHTML = '<option value="">-- select a run --</option>';
  for (const r of allRuns) {
    const o = document.createElement('option');
    o.value = r.run_id;
    const src = r.source === 'ma' ? '[MA]' : '[RL]';
    const accStr = r.accuracy !== null ? r.accuracy + '%' : '?%';
    o.textContent = `${src} ${r.label} — ${accStr} (${r.n_tasks} tasks)`;
    sel.appendChild(o);
  }
  sel.addEventListener('change', () => {
    if (sel.value) { switchTab('run'); loadRun(sel.value); }
  });
  if (allRuns.length) { sel.value = allRuns[allRuns.length-1].run_id; loadRun(sel.value); }
}

async function loadRun(runId) {
  const res = await fetch(`/api/run/${runId}`, { cache: 'no-store' });
  currentRun = await res.json();
  renderKPI(); renderFailureChart(); renderLatencyChart(); renderFilters(); renderTable();
}

function renderKPI() {
  const d = currentRun;
  const isMa = d.source === 'ma';
  const wh = (d.total_wall_clock_s/3600).toFixed(1);
  const runnable = d.tasks.filter(t => t.failure_class !== 'skipped');
  const failedTasks = runnable.filter(t => !t.success);
  const accSub = isMa
    ? `${d.n_success}/${runnable.length} runnable tasks`
    : `${d.n_success}/${d.n_tasks} tasks`;

  let card4, card6;
  if (isMa) {
    const nSkipped = d.tasks.filter(t => t.failure_class === 'skipped').length;
    const avgRuntime = runnable.length
      ? (runnable.reduce((s,t) => s + (t.time_s||0), 0) / runnable.length).toFixed(1) : '—';
    card4 = `<div class="card"><div class="label">Avg Task Runtime</div><div class="value">${avgRuntime}s</div><div class="sub">per runnable task</div></div>`;
    card6 = `<div class="card"><div class="label">Env Skipped</div><div class="value" style="color:var(--orange)">${nSkipped}</div><div class="sub">env setup failures</div></div>`;
  } else {
    const avgComp = d.n_tasks ? Math.round(d.total_completion_tokens / d.n_tasks) : 0;
    const avgPrompt = d.n_tasks ? Math.round(d.total_prompt_tokens / d.n_tasks) : 0;
    const avgStallFailed = failedTasks.length
      ? (failedTasks.reduce((s,t) => s + (t.max_stall_count||0), 0) / failedTasks.length).toFixed(1) : '0';
    card4 = `<div class="card"><div class="label">Avg Decode Tokens</div><div class="value">${avgComp.toLocaleString()}</div><div class="sub">${avgPrompt.toLocaleString()} prefill/task</div></div>`;
    card6 = `<div class="card"><div class="label">Avg Stall (failed)</div><div class="value">${avgStallFailed}</div><div class="sub">max consecutive</div></div>`;
  }

  document.getElementById('kpi-cards').innerHTML = `
    <div class="card"><div class="label">Accuracy</div><div class="value" style="color:${d.accuracy>=25?'var(--green)':'var(--red)'}">${d.accuracy}%</div><div class="sub">${accSub}</div></div>
    <div class="card"><div class="label">Avg Steps</div><div class="value">${d.avg_steps}</div><div class="sub">per task</div></div>
    <div class="card"><div class="label">Wall Clock</div><div class="value">${wh}h</div><div class="sub">${d.total_wall_clock_s.toLocaleString()}s total</div></div>
    ${card4}
    <div class="card"><div class="label">Avg Step Time</div><div class="value">${d.latency_avg.step_total_s}s</div><div class="sub">inference ${d.latency_avg.inference_s}s</div></div>
    ${card6}
  `;
}

function renderFailureChart() {
  const fc = currentRun.failure_counts;
  const keys = Object.keys(LABELS).filter(k => fc[k]);
  const ctx = document.getElementById('failureChart').getContext('2d');
  if (failureChart) failureChart.destroy();
  failureChart = new Chart(ctx, {
    type:'doughnut',
    data:{labels:keys.map(k=>`${LABELS[k]} (${fc[k]})`),datasets:[{data:keys.map(k=>fc[k]),backgroundColor:keys.map(k=>COLORS[k]),borderWidth:0}]},
    options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{position:'right',labels:{color:'#334155',font:{size:12},padding:8}}}}
  });
}

function renderLatencyChart() {
  const la = currentRun.latency_avg;
  const keys = ['screenshot_s','preprocess_s','inference_s','action_s'];
  const labels = ['Screenshot','Preprocess','Inference','Action'];
  const colors = ['#3b82f6','#8b5cf6','#f97316','#22c55e'];
  const ctx = document.getElementById('latencyChart').getContext('2d');
  if (latencyChart) latencyChart.destroy();
  latencyChart = new Chart(ctx, {
    type:'bar',
    data:{labels,datasets:[{data:keys.map(k=>la[k]),backgroundColor:colors,borderRadius:4}]},
    options:{responsive:true,maintainAspectRatio:false,indexAxis:'y',
      scales:{x:{grid:{color:'#e2e8f0'},ticks:{color:'#64748b'}},y:{grid:{display:false},ticks:{color:'#334155'}}},
      plugins:{legend:{display:false}}}
  });
}

function renderFilters() {
  const fc = currentRun.failure_counts;
  const div = document.getElementById('filters');
  div.innerHTML = '';
  const allBtn = document.createElement('button');
  allBtn.className = 'filter-btn' + (!activeFilter ? ' active' : '');
  allBtn.textContent = `All (${currentRun.n_tasks})`;
  allBtn.onclick = () => { activeFilter = null; renderFilters(); renderTable(); };
  div.appendChild(allBtn);
  for (const [cls, label] of Object.entries(LABELS)) {
    if (!fc[cls]) continue;
    const btn = document.createElement('button');
    btn.className = 'filter-btn' + (activeFilter===cls?' active':'');
    btn.innerHTML = `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${COLORS[cls]};margin-right:4px"></span>${label} (${fc[cls]})`;
    btn.onclick = () => { activeFilter = activeFilter===cls?null:cls; renderFilters(); renderTable(); };
    div.appendChild(btn);
  }
}

function renderTable() {
  let tasks = currentRun.tasks;
  if (activeFilter) tasks = tasks.filter(t => t.failure_class === activeFilter);
  if (sortKey) {
    tasks = [...tasks].sort((a,b) => {
      let va = getVal(a,sortKey), vb = getVal(b,sortKey);
      if (typeof va === 'string') return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
      return sortAsc ? va-vb : vb-va;
    });
  }
  const tbody = document.getElementById('task-tbody');
  tbody.innerHTML = tasks.map((t,i) => {
    const bg = COLORS[t.failure_class]||'#64748b';
    const pt = t.token_totals?.prompt_tokens??0;
    const ct = t.token_totals?.completion_tokens??0;
    const inf = t.latency_avg?.inference_s?.toFixed(2)??'-';
    const st = t.latency_avg?.step_total_s?.toFixed(2)??'-';
    const isMaTask = t.source === 'ma';
    const budget = t.max_steps || 0;
    const budgetCell = budget
      ? `<span title="${t.steps}/${budget} steps used">${budget}</span>`
      : '—';
    return `<tr class="expandable" data-task="${t.task}" data-combo="${t.combo||0}">
      <td>${t.task}</td>
      <td>${t.success?'<span style="color:#16a34a">&#10003;</span>':'<span style="color:#dc2626">&#10007;</span>'}</td>
      <td><span class="badge" style="background:${bg}">${t.failure_label}</span></td>
      <td class="num">${t.steps}</td>
      <td class="num">${budgetCell}</td>
      <td class="num">${isMaTask ? '—' : (t.max_stall_count??'-')}</td>
      <td class="num">${t.time_s?.toFixed(1)??'-'}</td>
      <td class="num">${isMaTask ? '—' : pt.toLocaleString()}</td>
      <td class="num">${isMaTask ? '—' : ct.toLocaleString()}</td>
      <td class="num">${inf}</td>
      <td class="num">${st}</td>
    </tr>`;
  }).join('');

  tbody.querySelectorAll('tr.expandable').forEach(tr => {
    tr.addEventListener('click', () => toggleStepViewer(tr));
  });
}

// ---- Step viewer ----
let openViewerRow = null;

async function toggleStepViewer(tr) {
  const existing = tr.nextElementSibling;
  if (existing && existing.classList.contains('step-viewer-row')) {
    existing.remove();
    openViewerRow = null;
    return;
  }
  if (openViewerRow) { openViewerRow.remove(); openViewerRow = null; }

  const task = tr.dataset.task;
  const combo = tr.dataset.combo || 0;
  const runId = currentRun.run_id;
  const res = await fetch(`/api/steps/${runId}/${task}?combo=${combo}`, { cache: 'no-store' });
  const data = await res.json();

  if (!data.steps || !data.steps.length) {
    const noData = document.createElement('tr');
    noData.className = 'step-viewer-row';
    noData.innerHTML = `<td colspan="11" class="step-viewer"><em>No step images found for this task.</em></td>`;
    tr.after(noData);
    openViewerRow = noData;
    return;
  }

  const viewerRow = document.createElement('tr');
  viewerRow.className = 'step-viewer-row';
  const td = document.createElement('td');
  td.colSpan = 11;
  td.className = 'step-viewer';

  let html = `<button class="close-btn" onclick="this.closest('.step-viewer-row').remove()">Close</button>`;
  html += `<div class="viewer-title">${escHtml(task)} — ${data.steps.length} steps</div>`;
  if (data.goal) html += `<div class="viewer-goal">${escHtml(data.goal)}</div>`;
  html += `<div class="steps-scroll">`;

  for (const s of data.steps) {
    const imgUrl = s.images.main || s.images.coarse || Object.values(s.images)[0];
    const fineUrl = s.images.fine;
    const action = s.action || '';
    const summary = s.summary || '';
    const stall = s.stall_count > 0 ? `<span class="stall-badge">stall ${s.stall_count}</span>` : '';
        const diff = s.screen_diff != null ? `diff=${s.screen_diff.toFixed(3)}` : '';
    html += `<div class="step-card">
      <img src="${imgUrl}" loading="lazy" onclick="showLightbox(this.src)" title="Click to zoom">
      ${fineUrl ? `<img src="${fineUrl}" loading="lazy" onclick="showLightbox(this.src)" style="border-top:1px solid var(--border)" title="Fine grid view">` : ''}
      <div class="step-info">
        <div class="step-num">Step ${s.step} ${stall}</div>
        ${action ? `<div class="step-action">${escHtml(action)}</div>` : ''}
        ${summary ? `<div class="step-summary">${escHtml(summary)}</div>` : ''}
        ${diff ? `<div class="step-meta">${diff}</div>` : ''}
      </div>
    </div>`;
  }
  html += `</div>`;
  td.innerHTML = html;
  viewerRow.appendChild(td);
  tr.after(viewerRow);
  openViewerRow = viewerRow;
  // Constrain scroll container to the visible table width so overflow-x kicks in
  const scrollDiv = td.querySelector('.steps-scroll');
  if (scrollDiv) {
    const tblWrap = tr.closest('.tbl-wrap');
    scrollDiv.style.maxWidth = ((tblWrap ? tblWrap.clientWidth : window.innerWidth) - 24) + 'px';
  }
  td.querySelector('.steps-scroll').scrollIntoView({behavior:'smooth', block:'nearest'});
}

function escHtml(s) { return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

function showLightbox(src) {
  const lb = document.createElement('div');
  lb.className = 'lightbox';
  lb.innerHTML = `<img src="${src}">`;
  lb.onclick = () => lb.remove();
  document.body.appendChild(lb);
}

function getVal(t, key) {
  if (key==='prompt_tokens') return t.token_totals?.prompt_tokens??0;
  if (key==='completion_tokens') return t.token_totals?.completion_tokens??0;
  if (key==='inference_s') return t.latency_avg?.inference_s??0;
  if (key==='step_total_s') return t.latency_avg?.step_total_s??0;
  if (key==='max_steps') return t.max_steps??0;
  return t[key]??'';
}

document.querySelectorAll('#tab-run th[data-key]').forEach(th => {
  th.addEventListener('click', () => {
    const key = th.dataset.key;
    if (sortKey===key) sortAsc=!sortAsc; else { sortKey=key; sortAsc=th.hasAttribute('data-num'); }
    renderTable();
  });
});

// ---- Compare ----
let compareMode = false;
function toggleCompare() {
  switchTab('run');
  compareMode = !compareMode;
  const panel = document.getElementById('compare-panel');
  panel.classList.toggle('hidden', !compareMode);
  if (compareMode) renderCompare();
}

async function renderCompare() {
  const ids = allRuns.map(r => r.run_id).join(',');
  const res = await fetch(`/api/compare?ids=${ids}`, { cache: 'no-store' });
  const data = await res.json();
  if (!data.length) return;

  const labels = data.map(d => d.label);
  const accs = data.map(d => d.accuracy);

  const failureKeys = Object.keys(LABELS).filter(k => k!=='success');
  const datasets = failureKeys.map(fk => ({
    label: LABELS[fk],
    data: data.map(d => d.failure_counts[fk]||0),
    backgroundColor: COLORS[fk], borderRadius: 2
  })).filter(ds => ds.data.some(v=>v>0));

  const ctx = document.getElementById('compareChart').getContext('2d');
  if (compareChart) compareChart.destroy();
  compareChart = new Chart(ctx, {
    type:'bar', data:{labels,datasets},
    options:{responsive:true,maintainAspectRatio:false,
      scales:{x:{stacked:true,grid:{display:false},ticks:{color:'#64748b',maxRotation:45,font:{size:10}}},
              y:{stacked:true,grid:{color:'#e2e8f0'},ticks:{color:'#64748b'},title:{display:true,text:'Failed tasks',color:'#64748b'}}},
      plugins:{legend:{position:'bottom',labels:{color:'#334155',font:{size:11},padding:6}},
        tooltip:{callbacks:{afterTitle:(items)=>{const idx=items[0].dataIndex;return `Accuracy: ${accs[idx]}%`;}}}}}
  });
}

function exportToCSV() {
  if (currentTab === 'ablation') {
    if (!ablationData) return;
    const cols = ["experiment", "model", "agent_mode", "thinking", "stall_action", "notes", "accuracy", "avg_steps", "avg_comp_tokens", "avg_prompt_tokens", "total_wall_clock_h", "avg_step_total_s", "avg_screenshot_s", "avg_preprocess_s", "avg_inference_s", "avg_action_s", "avg_ttft_s", "avg_decode_s", "avg_tpot_ms"];
    let csv = cols.join(",") + "\n";
    let data = [...ablationData];
    if (ablSortKey) {
      data.sort((a,b) => {
        let va = a[ablSortKey], vb = b[ablSortKey];
        if (typeof va === 'string') return ablSortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
        if (typeof va === 'boolean') { va = va?1:0; vb = vb?1:0; }
        return ablSortAsc ? va-vb : vb-va;
      });
    }
    data.forEach(r => {
      csv += cols.map(c => `"${r[c] ?? ''}"`).join(",") + "\n";
    });
    downloadCSV(csv, 'ablation_results.csv');
  } else if (currentTab === 'run') {
    if (!currentRun || !currentRun.tasks) return;
    const cols = ["task", "success", "failure_label", "steps", "max_stall_count", "time_s"];
    let csv = cols.join(",") + "\n";
    let tasks = currentRun.tasks;
    if (activeFilter) tasks = tasks.filter(t => t.failure_class === activeFilter);
    if (sortKey) {
      tasks = [...tasks].sort((a,b) => {
        let va = getVal(a,sortKey), vb = getVal(b,sortKey);
        if (typeof va === 'string') return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
        return sortAsc ? va-vb : vb-va;
      });
    }
    tasks.forEach(t => {
      csv += cols.map(c => `"${t[c] ?? ''}"`).join(",") + "\n";
    });
    downloadCSV(csv, `run_${currentRun.run_id}_tasks.csv`);
  }
}

function downloadCSV(csv, filename) {
  const blob = new Blob([csv], { type: 'text/csv' });
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
}

init();
</script>
</body>
</html>"""

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval failure dashboard")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    hn = socket.gethostname()
    if args.host in ("0.0.0.0", "::", "[::]"):
        print(f"\n  Dashboard listening on {args.host}:{args.port}")
        print(f"    From this server:   http://127.0.0.1:{args.port}")
        print(f"    From your laptop:   http://{hn}:{args.port}  (use this host/IP, not localhost)\n")
    else:
        print(f"\n  Dashboard -> http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
