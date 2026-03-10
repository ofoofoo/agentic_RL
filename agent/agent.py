from __future__ import annotations
"""
Core agent loop with dual-mode UI interaction:
  1. Element mode (primary) — uses UI hierarchy for pixel-perfect taps
  2. Grid mode (fallback)  — numbered grid overlay when elements aren't labeled
"""

import json
import os
import re
import time
from datetime import datetime

from .android_controller import AndroidController, UIElement
from .model import GeminiModel, VLLMModel
from .prompt import (
    build_element_prompt,
    build_grid_prompt,
    build_element_text_list,
    load_examples,
)

def parse_element_response(rsp: str) -> dict | None:
    """
    Parse a structured Observation/Thought/Action/Summary response
    for element mode.  Returns a dict with keys:
      observation, thought, action_raw, summary, parsed_action
    or None if unparseable.
    """
    try:
        observation = re.findall(r"Observation:\s*(.*?)$", rsp, re.MULTILINE)[0]
        thought = re.findall(r"Thought:\s*(.*?)$", rsp, re.MULTILINE)[0]
        act_str = re.findall(r"Action:\s*(.*?)$", rsp, re.MULTILINE)[0]
        summary = re.findall(r"Summary:\s*(.*?)$", rsp, re.MULTILINE)[0]
    except IndexError:
        return None

    parsed = _parse_action_string(act_str, grid_mode=False)
    if parsed is None:
        return None

    parsed_response = {
        "observation": observation,
        "thought": thought,
        "action_raw": act_str,
        "summary": summary,
        "parsed_action": parsed,
    }

    return parsed_response


def parse_grid_response(rsp: str) -> dict | None:
    """Same as parse_element_response but for grid-mode actions."""
    try:
        observation = re.findall(r"Observation:\s*(.*?)$", rsp, re.MULTILINE)[0]
        thought = re.findall(r"Thought:\s*(.*?)$", rsp, re.MULTILINE)[0]
        act_str = re.findall(r"Action:\s*(.*?)$", rsp, re.MULTILINE)[0]
        summary = re.findall(r"Summary:\s*(.*?)$", rsp, re.MULTILINE)[0]
    except IndexError:
        return None

    parsed = _parse_action_string(act_str, grid_mode=True)
    if parsed is None:
        return None

    return {
        "observation": observation,
        "thought": thought,
        "action_raw": act_str,
        "summary": summary,
        "parsed_action": parsed,
    }


def _to_int(s: str) -> int:
    """Extract integer from strings like 'element_6', 'elem6', '6'."""
    nums = re.findall(r'\d+', s)
    if nums:
        return int(nums[0])
    raise ValueError(f"No integer found in {s!r}")


def _parse_action_string(act_str: str, grid_mode: bool) -> dict | None:
    """
    Fault-tolerant parser for function-call style action strings.
    Handles common hallucinations / off-by-one naming from models.
    """
    act_str = act_str.strip()

    if "FINISH" in act_str:
        return {"action": "done"}

    # normalise: extract the function name before first "("
    if "(" not in act_str:
        return None
    act_name = act_str.split("(")[0].strip().lower()

    # ── aliases so hallucinated names still work ─────────────────────
    SWIPE_ALIASES = {"swipe", "swipe_element", "swipe_on", "swipe_to"}
    TAP_ALIASES   = {"tap", "click", "press", "tap_element"}
    TEXT_ALIASES  = {"text", "type", "input", "enter", "input_text"}
    LP_ALIASES    = {"long_press", "longpress", "long_tap"}
    CLEAR_ALIASES = {"clear_text", "clear", "delete_text", "erase_text"}

    try:
        # ── tap / click ───────────────────────────────────────────────
        if act_name in TAP_ALIASES:
            inner = re.findall(r'\((.*)\)', act_str)[0]
            parts = [p.strip().strip('"').strip("'") for p in inner.split(",")]
            if grid_mode and len(parts) >= 2:
                return {"action": "tap_grid", "area": _to_int(parts[0]), "subarea": parts[1]}
            else:
                return {"action": "tap", "element": _to_int(parts[0])}

        # ── long_press ────────────────────────────────────────────────
        elif act_name in LP_ALIASES:
            inner = re.findall(r'\((.*)\)', act_str)[0]
            parts = [p.strip().strip('"').strip("'") for p in inner.split(",")]
            if grid_mode and len(parts) >= 2:
                return {"action": "long_press_grid", "area": _to_int(parts[0]), "subarea": parts[1]}
            else:
                return {"action": "long_press", "element": _to_int(parts[0])}

        # ── text / type ───────────────────────────────────────────────
        elif act_name in TEXT_ALIASES:
            inner = re.findall(r'\((.*)\)', act_str)[0]
            text_val = inner.strip().strip('"').strip("'")
            return {"action": "text", "text": text_val}

        # ── clear_text ────────────────────────────────────────────────
        elif act_name in CLEAR_ALIASES:
            return {"action": "clear_text"}

        # ── swipe ─────────────────────────────────────────────────────
        elif act_name in SWIPE_ALIASES:
            inner = re.findall(r'\((.*)\)', act_str)[0]
            parts = [p.strip().strip('"').strip("'") for p in inner.split(",")]
            if grid_mode and len(parts) >= 4:
                return {
                    "action": "swipe_grid",
                    "start_area": _to_int(parts[0]), "start_subarea": parts[1],
                    "end_area": _to_int(parts[2]), "end_subarea": parts[3],
                }
            elif len(parts) >= 3:
                return {
                    "action": "swipe",
                    "element": _to_int(parts[0]),
                    "direction": parts[1],
                    "dist": parts[2],
                }
            else:
                return None

        elif act_name == "grid":
            return {"action": "grid"}

        elif act_name == "back":
            return {"action": "back"}

        elif act_name == "home":
            return {"action": "home"}

        else:
            return None

    except (IndexError, ValueError):
        return None


class Agent:
    def __init__(self, config: dict):
        backend = config.get("BACKEND").lower()
        if backend == "vllm":
            self.model = VLLMModel(
                api_key=config["VLLM_API_KEY"],
                model_name=config["VLLM_MODEL"],
                base_url=config.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
            )
            print(f"[agent] Backend: vLLM — {config['VLLM_MODEL']} @ {config.get('VLLM_BASE_URL', 'http://127.0.0.1:8000/v1')}")
        else:
            self.model = GeminiModel(
                api_key=config["GEMINI_API_KEY"],
                model_name=config["GEMINI_MODEL"],
            )
            print(f"[agent] Backend: Gemini — {config['GEMINI_MODEL']}")
        self.controller = AndroidController(serial=config["DEVICE_SERIAL"])
        self.output_dir = config["OUTPUT_DIR"]
        self.max_steps = config.get("MAX_STEPS", 20)
        self.screen_w, self.screen_h = self.controller.screen_size()

        # Prompts for both modes
        self.element_prompt = build_element_prompt(self.screen_w, self.screen_h)
        self.grid_prompt = build_grid_prompt(
            self.screen_w, self.screen_h,
            self.screen_w // 16, self.screen_h // 24,  # cell size
        )

        # load ICL examples (these still work with the new format)
        examples_dir = config.get("EXAMPLES_DIR", "./examples")
        self.examples = load_examples(examples_dir)
        if self.examples:
            print(f"[agent] Loaded {len(self.examples)} ICL example(s) from {examples_dir}")
        else:
            print(f"[agent] No ICL examples found in {examples_dir}: running zero-shot")

        os.makedirs(self.output_dir, exist_ok=True)

    def _build_prompt(
        self, task: str, step: int, history: list[dict],
        grid_on: bool, elem_list: list | None = None,
    ) -> str:
        sys_prompt = self.grid_prompt if grid_on else self.element_prompt

        history_text = ""
        if history:
            lines = []
            for i, h in enumerate(history):
                lines.append(f"  Step {i + 1}: {h['summary']}")
            history_text = "Actions taken so far:\n" + "\n".join(lines) + "\n\n"

        elem_text = ""
        if not grid_on and elem_list:
            elem_text = build_element_text_list(elem_list) + "\n\n"

        return (
            f"{sys_prompt}\n\n"
            f"Task: {task}\n\n"
            f"{elem_text}"
            f"{history_text}"
            f"Current step: {step + 1} / {self.max_steps}\n"
            f"What is the next action?"
        )

    def area_to_xy(self, area: int, subarea: str, rows: int, cols: int) -> tuple[int, int]:
        """Convert a numbered grid cell + subarea string to pixel (x, y)."""
        area -= 1
        row, col = area // cols, area % cols
        cell_w = self.screen_w // cols
        cell_h = self.screen_h // rows
        x0, y0 = col * cell_w, row * cell_h
        offsets = {
            "top-left":     (cell_w // 4,     cell_h // 4),
            "top":          (cell_w // 2,     cell_h // 4),
            "top-right":    (cell_w * 3 // 4, cell_h // 4),
            "left":         (cell_w // 4,     cell_h // 2),
            "center":       (cell_w // 2,     cell_h // 2),
            "right":        (cell_w * 3 // 4, cell_h // 2),
            "bottom-left":  (cell_w // 4,     cell_h * 3 // 4),
            "bottom":       (cell_w // 2,     cell_h * 3 // 4),
            "bottom-right": (cell_w * 3 // 4, cell_h * 3 // 4),
        }
        dx, dy = offsets.get(subarea, (cell_w // 2, cell_h // 2))
        return x0 + dx, y0 + dy

    def execute_action(
        self,
        parsed_action: dict,
        elem_list: list[UIElement] | None = None,
        rows: int = 1,
        cols: int = 1,
    ) -> None:
        name = parsed_action["action"]

        if name == "tap":
            idx = parsed_action["element"]
            if elem_list and 1 <= idx <= len(elem_list):
                x, y = elem_list[idx - 1].center
            else:
                raise ValueError(f"Element {idx} out of range (have {len(elem_list or [])})")
            self.controller.tap(x, y)

        elif name == "tap_grid":
            x, y = self.area_to_xy(parsed_action["area"], parsed_action.get("subarea", "center"), rows, cols)
            self.controller.tap(x, y)

        elif name == "long_press":
            idx = parsed_action["element"]
            if elem_list and 1 <= idx <= len(elem_list):
                x, y = elem_list[idx - 1].center
            else:
                raise ValueError(f"Element {idx} out of range")
            self.controller.long_press(x, y)

        elif name == "long_press_grid":
            x, y = self.area_to_xy(parsed_action["area"], parsed_action.get("subarea", "center"), rows, cols)
            self.controller.long_press(x, y)

        elif name == "swipe":
            idx = parsed_action["element"]
            if elem_list and 1 <= idx <= len(elem_list):
                cx, cy = elem_list[idx - 1].center
            else:
                raise ValueError(f"Element {idx} out of range")
            direction = parsed_action["direction"]
            dist_name = parsed_action.get("dist", "medium")
            # Use screen_h as the base so distances are meaningful regardless
            # of orientation. short=25%, medium=40%, long=65% of screen height.
            # A "long" up-swipe of ~65% is enough to reliably open the app drawer.
            dist_frac = {"short": 0.25, "medium": 0.40, "long": 0.65}.get(dist_name, 0.40)
            dist_px = int(self.screen_h * dist_frac)
            dx_map = {"left": -dist_px, "right": dist_px, "up": 0, "down": 0}
            dy_map = {"left": 0, "right": 0, "up": -dist_px, "down": dist_px}
            print(f"[agent] swipe element {idx} ({cx},{cy}) → {direction} {dist_name} ({dist_px}px)")
            self.controller.swipe(
                cx, cy,
                cx + dx_map.get(direction, 0),
                cy + dy_map.get(direction, 0),
                600,  # 600 ms: deliberate enough for app-drawer & scroll triggers
            )

        elif name == "swipe_grid":
            sx, sy = self.area_to_xy(parsed_action["start_area"], parsed_action["start_subarea"], rows, cols)
            ex, ey = self.area_to_xy(parsed_action["end_area"], parsed_action["end_subarea"], rows, cols)
            self.controller.swipe(sx, sy, ex, ey, 400)

        elif name == "text":
            self.controller.type_text(parsed_action["text"])

        elif name == "clear_text":
            print("[agent] clear_text: select-all + delete")
            self.controller.clear_text()

        elif name == "back":
            self.controller.back()
        elif name == "home":
            self.controller.home()
        elif name in ("done", "grid"):
            pass
        else:
            raise ValueError(f"Unknown action: {name!r}")

    def run(self, task: str) -> None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_dir = os.path.join(self.output_dir, run_id)
        trajectory_path = os.path.join(self.output_dir, f"{run_id}_trajectory.jsonl")
        os.makedirs(screenshot_dir, exist_ok=True)

        print(f"\n[agent] Task: {task}")
        print(f"[agent] Output: {self.output_dir}/{run_id}/")
        print(f"[agent] Max steps: {self.max_steps}\n")

        history: list[dict] = []
        grid_on = False
        rows, cols = 24, 16  # grid dimensions (only used in fallback)
        elem_list: list[UIElement] = []

        trajectory_start = time.perf_counter()

        for step in range(self.max_steps):
            step_start = time.perf_counter()

            # ── Observation ──────────────────────────────────────────────
            if grid_on:
                screenshot_path = os.path.join(screenshot_dir, f"step_{step:03d}.png")
                grid_path = os.path.join(screenshot_dir, f"step_{step:03d}_grid.png")
                grid_path, rows, cols, t_adb, t_preprocess = (
                    self.controller.screenshot_with_numbered_grid(screenshot_path, grid_path)
                )
                image_path = grid_path
                elem_list = []
                print(f"[step {step + 1}] Grid mode: {rows}r x {cols}c")
            else:
                labeled_path = os.path.join(screenshot_dir, f"step_{step:03d}_labeled.png")
                xml_path = os.path.join(screenshot_dir, f"step_{step:03d}.xml")
                labeled_path, elem_list, t_adb, t_hierarchy, t_label = (
                    self.controller.screenshot_with_elements(labeled_path, xml_path)
                )
                t_preprocess = t_hierarchy + t_label
                image_path = labeled_path
                print(f"[step {step + 1}] Element mode: {len(elem_list)} elements")

            # ── Prompt ───────────────────────────────────────────────────
            prompt = self._build_prompt(task, step, history, grid_on, elem_list)

            # ── Inference ────────────────────────────────────────────────
            t0 = time.perf_counter()
            raw_response = self.model.generate(prompt, image_path=image_path)
            t_inference = time.perf_counter() - t0
            print(f"[step {step + 1}] Model response ({t_inference:.2f}s):\n{raw_response}")

            # ── Parse ────────────────────────────────────────────────────
            if grid_on:
                result = parse_grid_response(raw_response)
            else:
                result = parse_element_response(raw_response)

            if result is None:
                print(f"[step {step + 1}] ERROR: could not parse structured response")
                print("Raw response was:", repr(raw_response))
                break

            parsed_action = result["parsed_action"]

            # ── Log ──────────────────────────────────────────────────────
            t_step = time.perf_counter() - step_start
            print(
                f"[step {step + 1}] Latency "
                f"adb={t_adb:.2f}s  "
                f"preprocess={t_preprocess:.2f}s  "
                f"inference={t_inference:.2f}s  "
                f"step_total={t_step:.2f}s"
            )
            record = {
                "step": step,
                "action": parsed_action,
                "observation": result.get("observation", ""),
                "thought": result.get("thought", ""),
                "summary": result.get("summary", ""),
                "timestamp": time.time(),
                "grid_on": grid_on,
                "n_elements": len(elem_list),
                "latency": {
                    "adb_s":         round(t_adb, 3),
                    "preprocess_s":  round(t_preprocess, 3),
                    "inference_s":   round(t_inference, 3),
                    "step_total_s":  round(t_step, 3),
                },
            }
            with open(trajectory_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            history.append({
                "summary": result.get("summary", raw_response[:100]),
                "action": parsed_action,
            })

            # ── Check done ───────────────────────────────────────────────
            if parsed_action["action"] == "done":
                print(f"\n[agent] Task complete after {step + 1} step(s).")
                break

            # ── Handle grid toggle ───────────────────────────────────────
            if parsed_action["action"] == "grid":
                grid_on = True
                print(f"[step {step + 1}] Switching to grid mode")
                continue
            else:
                grid_on = False

            # ── Execute ──────────────────────────────────────────────────
            try:
                self.execute_action(parsed_action, elem_list=elem_list, rows=rows, cols=cols)
            except Exception as e:
                print(f"[step {step + 1}] ERROR executing action: {e}")
                break

        else:
            print(f"\n[agent] Reached max steps ({self.max_steps}) without finishing.")

        t_total = time.perf_counter() - trajectory_start

        # ── Latency summary ──────────────────────────────────────────────
        print(f"\n{'─' * 58}")
        print(f"  Latency summary  |  Task: {task}")
        print(f"{'─' * 58}")
        print(f"  {'Step':<6} {'ADB':>7} {'Preprocess':>11} {'Inference':>10} {'Total':>8}")
        print(f"  {'─'*6} {'─'*7} {'─'*11} {'─'*10} {'─'*8}")
        with open(trajectory_path) as f:
            for line in f:
                rec = json.loads(line)
                lat = rec.get("latency", {})
                print(
                    f"  {rec['step'] + 1:<6} "
                    f"{lat.get('adb_s', 0):>6.2f}s "
                    f"{lat.get('preprocess_s', 0):>10.2f}s "
                    f"{lat.get('inference_s', 0):>9.2f}s "
                    f"{lat.get('step_total_s', 0):>7.2f}s"
                )
        print(f"{'─' * 58}")
        print(f"  Total wall-clock: {t_total:.2f}s")
        print(f"{'─' * 58}\n")
        print(f"[agent] Trajectory saved to: {trajectory_path}")
