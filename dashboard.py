"""
Eval-run failure-analysis dashboard.

    .venv/bin/python dashboard.py            # serves on http://localhost:8050
    .venv/bin/python dashboard.py --port 9000 # custom port
"""

import argparse
import glob
import json
import os
import re
from collections import Counter
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIRS = [
    os.path.expanduser("~/Documents/agentic_RL/output/aw_runs"),
    os.path.expanduser("~/Documents/agentic_RL/output/aw_runs_grid2level"),
]
LOG_DIR = os.path.expanduser("~/Documents/agentic_RL")
MIN_TASKS_FOR_DISPLAY = 50

BENCHMARK_LOG_MAP: dict[str, str] = {}

# Human-readable ablation labels (run_id → display name)
HIDDEN_RUNS = {"20260407_063049", "20260407_161831", "20260409_045014"}

# Controls dropdown / ablation-table ordering (run_ids not listed sort to end)
RUN_ORDER = [
    "20260422_020351",  # Gemma-4-E4B
    "20260408_233955",  # Grid 32x20
    "20260409_181157",  # Element (committed prompt)
    "20260410_055115",  # Grid 32x20 (committed prompt)
    "20260411_005838",  # Grid sweep (unknown)
    "20260411_175626",  # Grid 20x12
    "20260412_053603",  # Grid 40x24
    "20260413_090749",  # Hierarchical Grid
    "20260413_224723",  # Raw Coords (unnormalized)
    "20260414_062315",  # Raw Coords (normalized)
    "20260416_061450",  # Raw Coords Normalized + Reasoning
    "20260421_084144",  # Element + Escalate (first stall run)
    "20260424_000816",  # Element + Escalate (stall-only prompt)
    "20260424_091700",  # Element + Escalate (full stall prompt)
]

ABLATION_LABELS: dict[str, dict] = {
    "20260408_233955": {
        "name": "Grid 32x20",
        "model": "Qwen3-VL-8B", "agent_mode": "grid",
        "thinking": True, "stall_action": "none",
    },
    "20260409_181157": {
        "name": "Element (committed prompt)",
        "model": "Qwen3-VL-8B", "agent_mode": "element",
        "thinking": True, "stall_action": "none",
    },
    "20260410_055115": {
        "name": "Grid 32x20 (committed prompt)",
        "model": "Qwen3-VL-8B", "agent_mode": "grid",
        "thinking": True, "stall_action": "none",
    },
    "20260411_005838": {
        "name": "Grid sweep (unknown cell size)",
        "model": "Qwen3-VL-8B", "agent_mode": "grid",
        "thinking": True, "stall_action": "none",
    },
    "20260411_175626": {
        "name": "Grid 20x12",
        "model": "Qwen3-VL-8B", "agent_mode": "grid",
        "thinking": True, "stall_action": "none",
    },
    "20260412_053603": {
        "name": "Grid 40x24",
        "model": "Qwen3-VL-8B", "agent_mode": "grid",
        "thinking": True, "stall_action": "none",
    },
    "20260413_090749": {
        "name": "Hierarchical Grid (6x4 -> 8x6)",
        "model": "Qwen3-VL-8B", "agent_mode": "grid2level",
        "thinking": True, "stall_action": "none",
    },
    "20260413_224723": {
        "name": "Raw Coords (unnormalized)",
        "model": "Qwen3-VL-8B", "agent_mode": "raw",
        "thinking": True, "stall_action": "none",
    },
    "20260414_062315": {
        "name": "Raw Coords (normalized)",
        "model": "Qwen3-VL-8B", "agent_mode": "raw",
        "thinking": True, "stall_action": "none",
    },
    "20260416_061450": {
        "name": "Raw Coords Normalized + Reasoning",
        "model": "Qwen3-VL-8B", "agent_mode": "raw",
        "thinking": True, "stall_action": "none",
    },
    "20260421_084144": {
        "name": "Element + Escalate (first stall run)",
        "model": "Qwen3-VL-8B", "agent_mode": "element",
        "thinking": True, "stall_action": "escalate",
    },
    "20260422_020351": {
        "name": "Gemma-4-E4B (raw, no reasoning)",
        "model": "Gemma-4-E4B", "agent_mode": "raw",
        "thinking": False, "stall_action": "none",
    },
    "20260424_000816": {
        "name": "Element + Escalate (stall-only prompt)",
        "model": "Qwen3-VL-8B", "agent_mode": "element",
        "thinking": True, "stall_action": "escalate",
    },
    "20260424_091700": {
        "name": "Element + Escalate (full stall prompt)",
        "model": "Qwen3-VL-8B", "agent_mode": "element",
        "thinking": True, "stall_action": "escalate",
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
    "budget_exhaustion_with_stall": "Budget exhausted + stall",
    "budget_exhaustion": "Budget exhausted (no stall)",
    "severe_stall": "Severe stall (\u22655)",
    "moderate_stall": "Moderate stall (3-4)",
    "premature_finish": "Premature FINISH (\u22645 steps)",
    "other_failure": "Other failure",
}

FAILURE_COLORS = {
    "success": "#22c55e",
    "skipped": "#94a3b8",
    "stall_terminated": "#ef4444",
    "budget_exhaustion_with_stall": "#f97316",
    "budget_exhaustion": "#eab308",
    "severe_stall": "#dc2626",
    "moderate_stall": "#fb923c",
    "premature_finish": "#a78bfa",
    "other_failure": "#64748b",
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
    """Count successes from results.json fields (matches run_aw_benchmark success logic).

    Uses ``success`` when set. If a legacy row has ``success`` false but both
    ``env_success`` and ``agent_done`` are true, count it anyway so old runs
    do not need a hard-coded per-run override.
    """
    n = 0
    for t in tasks:
        if t.get("success"):
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
                runs.append({"run_id": entry, "base_dir": base})
    order_idx = {rid: i for i, rid in enumerate(RUN_ORDER)}
    runs.sort(key=lambda r: order_idx.get(r["run_id"], len(RUN_ORDER)))
    return runs


def _should_display(run_id: str, n_tasks: int, accuracy: float) -> bool:
    if run_id in HIDDEN_RUNS:
        return False
    if n_tasks < MIN_TASKS_FOR_DISPLAY:
        return False
    if accuracy == 0.0 or accuracy == 100.0:
        return False
    return True


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    global BENCHMARK_LOG_MAP
    BENCHMARK_LOG_MAP = _discover_log_map()
    yield

app = FastAPI(title="Eval Failure Dashboard", lifespan=lifespan)


@app.get("/api/runs")
def api_runs():
    entries = discover_runs()
    summaries = []
    for e in entries:
        rfile = os.path.join(e["base_dir"], e["run_id"], "results.json")
        try:
            with open(rfile) as f:
                tasks = json.load(f)
            n = len(tasks)
            succ = _success_count_from_tasks(tasks)
        except Exception:
            n, succ = 0, 0
        acc = round(succ / n * 100, 1) if n else 0
        if not _should_display(e["run_id"], n, acc):
            continue
        label = _get_run_label(e["run_id"])
        summaries.append({
            "run_id": e["run_id"],
            "base_dir": os.path.basename(e["base_dir"]),
            "label": label,
            "n_tasks": n,
            "accuracy": acc,
        })
    return JSONResponse(summaries)


@app.get("/api/run/{run_id}")
def api_run(run_id: str):
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
        for base in BASE_DIRS:
            data = load_run(rid, base)
            if data is not None:
                del data["tasks"]
                results.append(data)
                break
    return JSONResponse(results)


@app.get("/api/ablation_table")
def api_ablation_table():
    """Return all runs as a flat ablation-summary table."""
    entries = discover_runs()
    rows = []
    for e in entries:
        for base in BASE_DIRS:
            data = load_run(e["run_id"], base)
            if data is None:
                continue
            if not _should_display(e["run_id"], data["n_tasks"], data["accuracy"]):
                break
            rows.append({
                "run_id": data["run_id"],
                "experiment": data["label"],
                "model": data["model"],
                "agent_mode": data["agent_mode"],
                "thinking": data["thinking"],
                "stall_action": data["stall_action"],
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
    rows.sort(key=lambda r: order_idx.get(r["run_id"], len(RUN_ORDER)))
    return JSONResponse(rows)


@app.get("/api/steps/{run_id}/{task_name}")
def api_steps(run_id: str, task_name: str, combo: int = 0):
    """Return step images and (if available) parsed actions for a task."""
    task_dir_name = f"{task_name}_combo{combo}"
    task_dir = None
    for base in BASE_DIRS:
        candidate = os.path.join(base, run_id, task_dir_name)
        if os.path.isdir(candidate):
            task_dir = candidate
            break
    if task_dir is None:
        return JSONResponse({"steps": []})

    images = sorted(f for f in os.listdir(task_dir) if f.endswith(".png"))
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
.step-viewer .viewer-title{font-weight:600;font-size:14px;margin-bottom:10px}

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
          <th data-abl="experiment">Experiment</th>
          <th data-abl="model">Model</th>
          <th data-abl="agent_mode">Mode</th>
          <th data-abl="thinking">Thinking</th>
          <th data-abl="stall_action">Stall</th>
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
  budget_exhaustion_with_stall:"#f97316",budget_exhaustion:"#eab308",
  severe_stall:"#dc2626",moderate_stall:"#fb923c",
  premature_finish:"#a78bfa",other_failure:"#64748b"
};
const LABELS = {
  success:"Success",skipped:"Skipped",stall_terminated:"Stall-terminated",
  budget_exhaustion_with_stall:"Budget + stall",budget_exhaustion:"Budget exhausted",
  severe_stall:"Severe stall (>=5)",moderate_stall:"Moderate stall (3-4)",
  premature_finish:"Premature FINISH",other_failure:"Other failure"
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
  const res = await fetch('/api/ablation_table');
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
    return `<tr class="${r.accuracy >= 30 ? 'row-highlight':''}">
      <td><strong>${r.experiment}</strong></td>
      <td>${r.model}</td>
      <td>${r.agent_mode}</td>
      <td>${r.thinking ? 'Yes' : 'No'}</td>
      <td>${r.stall_action}</td>
      <td class="num"><span class="acc-bar" style="width:${barW}px;background:${barColor}"></span> ${r.accuracy}%</td>
      <td class="num">${r.avg_steps}</td>
      <td class="num">${r.avg_comp_tokens.toLocaleString()}</td>
      <td class="num">${r.avg_prompt_tokens.toLocaleString()}</td>
      <td class="num">${r.total_wall_clock_h}</td>
      <td class="num">${r.avg_step_total_s}</td>
      <td class="num">${r.avg_screenshot_s}</td>
      <td class="num">${r.avg_preprocess_s}</td>
      <td class="num">${r.avg_inference_s}</td>
      <td class="num">${r.avg_action_s}</td>
      <td class="num">${r.avg_ttft_s}</td>
      <td class="num">${r.avg_decode_s}</td>
      <td class="num">${r.avg_tpot_ms}</td>
    </tr>`;
  }).join('');
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
  const res = await fetch('/api/runs');
  allRuns = await res.json();
  const sel = document.getElementById('run-select');
  sel.innerHTML = '<option value="">-- select a run --</option>';
  for (const r of allRuns) {
    const o = document.createElement('option');
    o.value = r.run_id;
    o.textContent = `${r.label} -- ${r.accuracy}% (${r.n_tasks} tasks)`;
    sel.appendChild(o);
  }
  sel.addEventListener('change', () => {
    if (sel.value) { switchTab('run'); loadRun(sel.value); }
  });
  if (allRuns.length) { sel.value = allRuns[allRuns.length-1].run_id; loadRun(sel.value); }
}

async function loadRun(runId) {
  const res = await fetch(`/api/run/${runId}`);
  currentRun = await res.json();
  renderKPI(); renderFailureChart(); renderLatencyChart(); renderFilters(); renderTable();
}

function renderKPI() {
  const d = currentRun;
  const wh = (d.total_wall_clock_s/3600).toFixed(1);
  const avgComp = d.n_tasks ? Math.round(d.total_completion_tokens / d.n_tasks) : 0;
  const avgPrompt = d.n_tasks ? Math.round(d.total_prompt_tokens / d.n_tasks) : 0;
  const failedTasks = d.tasks.filter(t => !t.success);
  const avgStallFailed = failedTasks.length
    ? (failedTasks.reduce((s,t) => s + (t.max_stall_count||0), 0) / failedTasks.length).toFixed(1) : '0';
  document.getElementById('kpi-cards').innerHTML = `
    <div class="card"><div class="label">Accuracy</div><div class="value" style="color:${d.accuracy>=25?'var(--green)':'var(--red)'}">${d.accuracy}%</div><div class="sub">${d.n_success}/${d.n_tasks} tasks</div></div>
    <div class="card"><div class="label">Avg Steps</div><div class="value">${d.avg_steps}</div><div class="sub">per task</div></div>
    <div class="card"><div class="label">Wall Clock</div><div class="value">${wh}h</div><div class="sub">${d.total_wall_clock_s.toLocaleString()}s total</div></div>
    <div class="card"><div class="label">Avg Decode Tokens</div><div class="value">${avgComp.toLocaleString()}</div><div class="sub">${avgPrompt.toLocaleString()} prefill/task</div></div>
    <div class="card"><div class="label">Avg Step Time</div><div class="value">${d.latency_avg.step_total_s}s</div><div class="sub">inference ${d.latency_avg.inference_s}s</div></div>
    <div class="card"><div class="label">Avg Stall (failed)</div><div class="value">${avgStallFailed}</div><div class="sub">max consecutive</div></div>
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
    return `<tr class="expandable" data-task="${t.task}" data-combo="${t.combo||0}">
      <td>${t.task}</td>
      <td>${t.success?'<span style="color:#16a34a">&#10003;</span>':'<span style="color:#dc2626">&#10007;</span>'}</td>
      <td><span class="badge" style="background:${bg}">${t.failure_label}</span></td>
      <td class="num">${t.steps}</td>
      <td class="num">${t.max_stall_count??'-'}</td>
      <td class="num">${t.time_s?.toFixed(1)??'-'}</td>
      <td class="num">${pt.toLocaleString()}</td>
      <td class="num">${ct.toLocaleString()}</td>
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
  const res = await fetch(`/api/steps/${runId}/${task}?combo=${combo}`);
  const data = await res.json();

  if (!data.steps || !data.steps.length) {
    const noData = document.createElement('tr');
    noData.className = 'step-viewer-row';
    noData.innerHTML = `<td colspan="10" class="step-viewer"><em>No step images found for this task.</em></td>`;
    tr.after(noData);
    openViewerRow = noData;
    return;
  }

  const viewerRow = document.createElement('tr');
  viewerRow.className = 'step-viewer-row';
  const td = document.createElement('td');
  td.colSpan = 10;
  td.className = 'step-viewer';

  let html = `<button class="close-btn" onclick="this.closest('.step-viewer-row').remove()">Close</button>`;
  html += `<div class="viewer-title">${task} — ${data.steps.length} steps</div>`;
  html += `<div class="steps-scroll">`;

  for (const s of data.steps) {
    const imgUrl = s.images.main || s.images.coarse || Object.values(s.images)[0];
    const fineUrl = s.images.fine;
    const action = s.action || '';
    const summary = s.summary || '';
    const stall = s.stall_count > 0 ? `<span class="stall-badge">stall ${s.stall_count}</span>` : '';
    const diff = s.screen_diff !== undefined ? `diff=${s.screen_diff.toFixed(3)}` : '';
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
  const res = await fetch(`/api/compare?ids=${ids}`);
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
    print(f"\n  Dashboard -> http://localhost:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
