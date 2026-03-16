# Swap out the system sqlite3 (which lacks FTS4) with our custom-built pysqlite3.
# This MUST be the very first import before anything from android_world is loaded.
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import argparse
import json
import os
import time
from datetime import datetime

os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_TRACE"] = ""

import yaml
from dotenv import load_dotenv

from android_world import registry
from android_world.env import env_launcher
from agent.aw_adapter import AWAgentAdapter


def main():
    parser = argparse.ArgumentParser(description="Run AndroidWorld benchmark")
    parser.add_argument(
        "--tasks", type=str, default=None,
        help="Comma-separated task names (e.g. ContactsAddContact,ClockStopWatchRunning). "
             "Leave empty to run all tasks.",
    )
    parser.add_argument(
        "--backend", type=str, default="gemini", choices=["gemini", "vllm"],
        help="Model backend to use.",
    )
    parser.add_argument(
        "--n_task_combinations", type=int, default=1,
        help="Number of random parameter combos per task.",
    )
    parser.add_argument(
        "--console_port", type=int, default=5554,
        help="Emulator console port (from `adb devices`).",
    )
    parser.add_argument(
        "--perform_emulator_setup", action="store_true",
        help="One-time setup: installs AndroidWorld apps on the emulator.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./output/aw_runs",
        help="Directory for screenshots and results.",
    )
    parser.add_argument(
        "--manual", type=bool, default=False,
        help="Manual mode: set to True to manually control the emulator for debugging a task."
    )
    args = parser.parse_args()

    load_dotenv()
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    config["BACKEND"] = args.backend
    config["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY") # uses one of these API keys depending on selected backend
    config["VLLM_API_KEY"] = os.environ.get("VLLM_API_KEY")

    adb_path = os.path.expanduser("~/Library/Android/sdk/platform-tools/adb")
    env = env_launcher.load_and_setup_env( # launch aw env
        console_port=args.console_port,
        emulator_setup=args.perform_emulator_setup,
        adb_path=adb_path,
    )
    env.reset(go_home=True)

    task_registry = registry.TaskRegistry()
    aw_registry = task_registry.get_registry(task_registry.ANDROID_WORLD_FAMILY)
    if args.tasks:
        task_names = [t.strip() for t in args.tasks.split(",")]
        for name in task_names:
            if name not in aw_registry:
                raise ValueError(
                    f"Task '{name}' not in registry. "
                    f"Available: {sorted(aw_registry.keys())}"
                )
    else:
        task_names = sorted(aw_registry.keys())

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(args.output_dir, run_id)
    os.makedirs(session_dir, exist_ok=True)
    results = []
    print(f"\n{'=' * 60}")
    print(f"AndroidWorld Benchmark: {len(task_names)} tasks x {args.n_task_combinations} combos")
    print(f"Backend: {args.backend}  |  Output directory: {session_dir}")
    print(f"{'=' * 60}\n")

    for task_name in task_names:
        task_type = aw_registry[task_name]
        for combo_idx in range(args.n_task_combinations):
            params = task_type.generate_random_params()
            task = task_type(params)

            env.reset(go_home=True) # reset env to home screen
            task.initialize_task(env) # initialize task

            goal = str(task.goal)
            max_steps = int(task.complexity * 15)

            print(f"[{task_name}] (combo {combo_idx + 1}/{args.n_task_combinations})")
            print(f"Goal: {goal}")
            print(f"Max steps: {max_steps}")

            task_dir = os.path.join(session_dir, f"{task_name}_combo{combo_idx}")
            os.makedirs(task_dir, exist_ok=True)

            if args.manual: # skip everything with the agent, just let the user control the emulator
                print(f"Manual mode enabled. Complete the task manually.")
                print(f"Will be checking if the task is complete every second.")
                while True:
                    if task.is_successful(env) == 1.0:
                        print("\033[32menv confirms task complete!\033[0m")
                        break
                    time.sleep(1.0)
                continue

            adapter = AWAgentAdapter(env=env, config=config, output_dir=task_dir, transition_pause=1.0)
            adapter.set_max_steps(max_steps)
            adapter.reset_episode()

            # run agent loop
            t_start = time.perf_counter()
            step_records: list[dict] = []
            agent_done = False
            for step_idx in range(max_steps):
                response = adapter.step(goal) # main loop driver
                if response.data and "latency" in response.data:
                    step_records.append(response.data)
                if response.done: # model said FINISH — stop immediately
                    agent_done = True
                    print("model said FINISH")
                    break
            t_elapsed = time.perf_counter() - t_start
            # success = env confirms AND agent explicitly terminated
            task_successful = task.is_successful(env) == 1.0
            if task_successful:
                print("\033[32menv confirms task complete!\033[0m")
            success = task_successful if agent_done else False
            print(f"agent_done: {agent_done}, task_successful: {task_successful}, success: {success}")

            status = "✅" if success else "❌"
            print(f"{status} {task_name} — {'success' if success else 'failed'} "
                  f"({step_idx + 1} steps, {t_elapsed:.1f}s)")

            if step_records:
                print(f"{'Step':>4}  {'Screenshot':>10}  {'Preprocess':>10}  {'Prompt':>7}  {'Inference':>9}  {'Total':>7}")
                print("   " + "-" * 58)
                for rec in step_records:
                    lat = rec["latency"]
                    print(f"   {rec['step']:>4}  "
                          f"{lat['screenshot_s']:>9.2f}s  "
                          f"{lat['preprocess_s']:>9.2f}s  "
                          f"{lat['prompt_s']:>6.2f}s  "
                          f"{lat['inference_s']:>8.2f}s  "
                          f"{lat['step_total_s']:>6.2f}s")
                def avg(key): return sum(r["latency"][key] for r in step_records) / len(step_records)
                print("   " + "-" * 58)
                print(f"   {'avg':>4}  {avg('screenshot_s'):>9.2f}s  {avg('preprocess_s'):>9.2f}s  "
                      f"{avg('prompt_s'):>6.2f}s  {avg('inference_s'):>8.2f}s  {avg('step_total_s'):>6.2f}s")

            results.append({
                "task": task_name,
                "combo": combo_idx,
                "goal": goal,
                "success": success,
                "steps": step_idx + 1,
                "time_s": round(t_elapsed, 2),
                "latency_avg": {
                    "screenshot_s":  round(sum(r["latency"]["screenshot_s"]  for r in step_records) / len(step_records), 3),
                    "preprocess_s":  round(sum(r["latency"]["preprocess_s"]  for r in step_records) / len(step_records), 3),
                    "prompt_s":      round(sum(r["latency"]["prompt_s"]      for r in step_records) / len(step_records), 3),
                    "inference_s":   round(sum(r["latency"]["inference_s"]   for r in step_records) / len(step_records), 3),
                    "step_total_s":  round(sum(r["latency"]["step_total_s"]  for r in step_records) / len(step_records), 3),
                } if step_records else {},
            })

            # ── Running accuracy table ────────────────────────────────
            n_done = len(results)
            n_success_so_far = sum(1 for r in results if r["success"])
            acc_so_far = n_success_so_far / n_done * 100
            print(f"\n{'─' * 60}")
            print(f"  RUNNING ACCURACY: {n_success_so_far}/{n_done} = {acc_so_far:.1f}%  "
                  f"({len(task_names) - n_done} tasks remaining)")
            print(f"  {'Task':<40}  {'Status':>6}  {'Steps':>5}  {'Time':>6}")
            print(f"  {'─'*40}  {'─'*6}  {'─'*5}  {'─'*6}")
            for r in results:
                icon = "✅" if r["success"] else "❌"
                print(f"  {r['task']:<40}  {icon:>6}  {r['steps']:>5}  {r['time_s']:>5.1f}s")
            print(f"{'─' * 60}\n")

            try:
                task.tear_down(env)
            except Exception:
                pass

    n_success = sum(1 for r in results if r["success"])
    n_total = len(results)
    accuracy = (n_success / n_total * 100) if n_total else 0

    print(f"\n{'=' * 60}")
    print(f"RESULTS  ({run_id})")
    print(f"{'=' * 60}")
    print(f"Tasks run:  {n_total}")
    print(f"Successes:  {n_success}")
    print(f"Accuracy:   {accuracy:.1f}%")
    if n_success > 0:
        avg_steps = sum(r["steps"] for r in results if r["success"]) / n_success
        print(f"Avg steps (success): {avg_steps:.1f}")
    print(f"{'=' * 60}")

    results_path = os.path.join(session_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")
    env.close()

if __name__ == "__main__":
    main()
