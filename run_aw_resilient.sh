#!/usr/bin/env bash
# run_aw_resilient.sh
#
# Runs the AndroidWorld benchmark one task at a time, with a full emulator
# restart between every task so that a crash on one task can't affect the next.
#
# Usage:
#   bash run_aw_resilient.sh [OPTIONS]
#
# Options:
#   --tasks         Comma-separated task names (default: all tasks)
#   --backend       gemini | vllm  (default: gemini)
#   --n_combos      Number of random param combos per task (default: 1)
#   --output_dir    Where to write per-task results (default: ./output/aw_runs)
#   --console_port  ADB console port                  (default: 5554)
#   --grpc_port     Emulator gRPC port                (default: 8554)
#   --boot_timeout  Seconds to wait for emulator boot (default: 120)
#
# Example – run three specific tasks with two combos each:
#   bash run_aw_resilient.sh --tasks ContactsAddContact,ClockStopWatchRunning \
#                             --n_combos 2 --backend gemini

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
TASKS=""
BACKEND="vllm"
N_COMBOS=1
OUTPUT_DIR="./output/aw_runs"
CONSOLE_PORT=5554
GRPC_PORT=8554
BOOT_TIMEOUT=120
AVD_NAME="AndroidWorldAvd"
EMULATOR="$HOME/Library/Android/sdk/emulator/emulator"
ADB="$HOME/Library/Android/sdk/platform-tools/adb"

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --tasks)        TASKS="$2";         shift ;;
        --backend)      BACKEND="$2";       shift ;;
        --n_combos)     N_COMBOS="$2";      shift ;;
        --output_dir)   OUTPUT_DIR="$2";    shift ;;
        --console_port) CONSOLE_PORT="$2";  shift ;;
        --grpc_port)    GRPC_PORT="$2";     shift ;;
        --boot_timeout) BOOT_TIMEOUT="$2";  shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# ── Helpers ───────────────────────────────────────────────────────────────────
EMULATOR_SERIAL="emulator-${CONSOLE_PORT}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "$LOG_DIR"
SUMMARY_FILE="${OUTPUT_DIR}/resilient_run_$(date +%Y%m%d_%H%M%S)_summary.json"

start_emulator() {
    echo "[emulator] Starting $AVD_NAME on gRPC port $GRPC_PORT..."
    # Launch emulator in background, redirect its output to a log file
    "$EMULATOR" -avd "$AVD_NAME" -no-snapshot -grpc "$GRPC_PORT" \
        > "$LOG_DIR/emulator.log" 2>&1 &
    EMULATOR_PID=$!
    echo "[emulator] PID=$EMULATOR_PID"
}

wait_for_boot() {
    echo "[emulator] Waiting for boot (timeout ${BOOT_TIMEOUT}s)..."
    local deadline=$(( $(date +%s) + BOOT_TIMEOUT ))
    until "$ADB" -s "$EMULATOR_SERIAL" shell getprop sys.boot_completed 2>/dev/null | grep -q "1"; do
        if [[ $(date +%s) -gt $deadline ]]; then
            echo "[emulator] ❌ Boot timed out after ${BOOT_TIMEOUT}s. Aborting."
            kill "$EMULATOR_PID" 2>/dev/null || true
            exit 1
        fi
        sleep 3
    done
    # Extra settling time so the launcher is fully ready
    sleep 5
    echo "[emulator] ✅ Boot complete."
}

kill_emulator() {
    echo "[emulator] Killing emulator (PID=${EMULATOR_PID:-unknown})..."
    "$ADB" -s "$EMULATOR_SERIAL" emu kill 2>/dev/null || true
    sleep 2
    # Force-kill if still running
    kill "$EMULATOR_PID" 2>/dev/null || true
    # Wait for the device to disappear from adb so the port is free
    local deadline=$(( $(date +%s) + 30 ))
    while "$ADB" devices | grep -q "$EMULATOR_SERIAL"; do
        if [[ $(date +%s) -gt $deadline ]]; then break; fi
        sleep 1
    done
    echo "[emulator] Emulator stopped."
}

# ── Resolve task list ─────────────────────────────────────────────────────────
# If --tasks not provided, ask the python script for the full registry
if [[ -z "$TASKS" ]]; then
    echo "[setup] Fetching full task list from android_world registry..."
    TASKS=$(python3 - <<'EOF'
import sys, pysqlite3; sys.modules["sqlite3"] = pysqlite3
from android_world import registry
r = registry.TaskRegistry()
names = sorted(r.get_registry(r.ANDROID_WORLD_FAMILY).keys())
print(",".join(names))
EOF
    )
    echo "[setup] Found $(echo "$TASKS" | tr ',' '\n' | wc -l | tr -d ' ') tasks."
fi

IFS=',' read -ra TASK_LIST <<< "$TASKS"
TOTAL=${#TASK_LIST[@]}

# ── Summary state ─────────────────────────────────────────────────────────────
declare -a RESULTS=()
SUCCESS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

echo ""
echo "========================================================"
echo "  AndroidWorld Resilient Benchmark"
echo "  Tasks:   $TOTAL  |  Combos: $N_COMBOS  |  Backend: $BACKEND"
echo "  Output:  $OUTPUT_DIR"
echo "========================================================"

# ── Main loop ─────────────────────────────────────────────────────────────────
IDX=0
for TASK in "${TASK_LIST[@]}"; do
    IDX=$(( IDX + 1 ))
    echo ""
    echo "──────────────────────────────────────────────────────"
    echo "  [$IDX/$TOTAL] Task: $TASK"
    echo "──────────────────────────────────────────────────────"

    # 1. Start a fresh emulator
    start_emulator
    wait_for_boot

    # 2. Run this single task via run_aw_benchmark.py
    TASK_LOG="$LOG_DIR/${TASK}.log"
    EXIT_CODE=0
    python3 -u run_aw_benchmark.py \
        --tasks      "$TASK" \
        --backend    "$BACKEND" \
        --n_task_combinations "$N_COMBOS" \
        --console_port "$CONSOLE_PORT" \
        --output_dir   "$OUTPUT_DIR" \
        2>&1 | tee "$TASK_LOG" || EXIT_CODE=$?

    # 3. Parse success/fail from the log
    if grep -q "✅" "$TASK_LOG"; then
        STATUS="success"
        SUCCESS_COUNT=$(( SUCCESS_COUNT + 1 ))
        echo "  → ✅ SUCCESS"
        echo " SUCCESS COUNT: $SUCCESS_COUNT"
        echo " TOTAL NUMBER OF TASKS: $IDX"
        echo " SUCCESS RATE: $(( SUCCESS_COUNT / IDX * 100 ))%"
    elif [[ $EXIT_CODE -ne 0 ]]; then
        STATUS="crash"
        SKIP_COUNT=$(( SKIP_COUNT + 1 ))
        echo "  → 💥 CRASHED (exit $EXIT_CODE) — continuing to next task"
        echo " CRASH COUNT: $SKIP_COUNT"
        echo " TOTAL NUMBER OF TASKS: $IDX"
        echo " SUCCESS RATE: $(( SUCCESS_COUNT / IDX * 100 ))%"
    else
        STATUS="fail"
        FAIL_COUNT=$(( FAIL_COUNT + 1 ))
        echo "  → ❌ FAILED"
        echo " FAIL COUNT: $FAIL_COUNT"
        echo " TOTAL NUMBER OF TASKS: $IDX"
        echo " SUCCESS RATE: $(( SUCCESS_COUNT / IDX * 100 ))%"
    fi

    RESULTS+=("{\"task\":\"$TASK\",\"status\":\"$STATUS\"}")

    # 4. Kill emulator before next task
    kill_emulator
done

# ── Write summary JSON ────────────────────────────────────────────────────────
ACCURACY=0
if [[ $TOTAL -gt 0 ]]; then
    ACCURACY=$(echo "scale=1; $SUCCESS_COUNT * 100 / $TOTAL" | bc)
fi

# Build JSON array manually (no jq dependency)
JSON_ARRAY=$(IFS=','; echo "[${RESULTS[*]}]")
cat > "$SUMMARY_FILE" <<EOF
{
  "total": $TOTAL,
  "success": $SUCCESS_COUNT,
  "fail": $FAIL_COUNT,
  "crash": $SKIP_COUNT,
  "accuracy_pct": $ACCURACY,
  "results": $JSON_ARRAY
}
EOF

echo ""
echo "========================================================"
echo "  FINAL RESULTS"
echo "========================================================"
echo "  Total:    $TOTAL"
echo "  Success:  $SUCCESS_COUNT  (${ACCURACY}%)"
echo "  Fail:     $FAIL_COUNT"
echo "  Crash:    $SKIP_COUNT"
echo "  Summary:  $SUMMARY_FILE"
echo "========================================================"
