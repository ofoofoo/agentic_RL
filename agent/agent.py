"""
Core agent loop: screenshot, build prompt (with histroy), call gemini, parse response, execute action, add to trajectory log
"""

import json
import os
import time
from datetime import datetime

from .android_controller import AndroidController
from .model import GeminiModel
from .prompt import build_system_prompt, load_examples


class Agent:
    def __init__(self, config: dict):
        self.model = GeminiModel(
            api_key=config["GEMINI_API_KEY"],
            model_name=config["GEMINI_MODEL"],
        )
        self.controller = AndroidController(serial=config["DEVICE_SERIAL"])
        self.output_dir = config["OUTPUT_DIR"]
        self.max_steps = config.get("MAX_STEPS", 20)
        self.screen_w, self.screen_h = self.controller.screen_size()
        self.system_prompt = build_system_prompt(self.screen_w, self.screen_h)

        # load ICL examples
        examples_dir = config.get("EXAMPLES_DIR", "./examples")
        self.examples = load_examples(examples_dir)
        if self.examples:
            print(f"[agent] Loaded {len(self.examples)} ICL example(s) from {examples_dir}")
        else:
            print(f"[agent] No ICL examples found in {examples_dir}: running zero-shot")

        os.makedirs(self.output_dir, exist_ok=True)

    def build_prompt(self, task: str, step: int, history: list[dict]) -> str:
        history_text = ""
        if history:
            lines = [
                f"  Step {i + 1}: {json.dumps(h)}"
                for i, h in enumerate(history)
            ]
            history_text = "Actions taken so far:\n" + "\n".join(lines) + "\n\n"

        return (
            f"{self.system_prompt}\n\n"
            f"Task: {task}\n\n"
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

    def execute_action(self, action: dict, rows: int = 1, cols: int = 1) -> None:
        name = action["action"]
        args = action.get("args", {})

        if name == "tap":
            if "area" in args:
                x, y = self.area_to_xy(args["area"], args.get("subarea", "center"), rows, cols)
            else:
                x, y = args["x"], args["y"]
            self.controller.tap(x, y)
        elif name == "swipe":
            self.controller.swipe(
                args["x1"], args["y1"], args["x2"], args["y2"],
                args.get("duration_ms", 400),
            )
        elif name == "type":
            self.controller.type_text(args["text"])
        elif name == "back":
            self.controller.back()
        elif name == "home":
            self.controller.home()
        elif name == "done":
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

        rows, cols = 1, 1  # updated each step after screenshot

        for step in range(self.max_steps):
            # screenshot
            screenshot_path = os.path.join(screenshot_dir, f"step_{step:03d}.png")
            grid_path = os.path.join(screenshot_dir, f"step_{step:03d}_grid.png")
            grid_path, rows, cols = self.controller.screenshot_with_numbered_grid(screenshot_path, grid_path)
            print(f"[step {step + 1}] Grid: {rows}r x {cols}c — saved to {grid_path}")

            # build prompt
            prompt = self.build_prompt(task, step, history)

            # call gemini
            raw_response = self.model.generate(
                prompt,
                image_path=grid_path,
                examples=self.examples,
            )
            print(f"[step {step + 1}] Model response: {raw_response}")
            # parse json action
            action = None
            for line in reversed(raw_response.strip().splitlines()):
                line = line.strip().removeprefix("```json").removesuffix("```").strip()
                if not line.startswith("{"):
                    continue
                try:
                    action = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue

            if action is None:
                print(f"[step {step + 1}] ERROR: no valid JSON found in response")
                print("Raw response was:", repr(raw_response))
                break

            # log step
            record = {
                "step": step,
                "screenshot": screenshot_path,
                "action": action,
                "timestamp": time.time(),
            }
            with open(trajectory_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            # check for done
            if action.get("action") == "done":
                print(f"\n[agent] Task complete after {step + 1} step(s).")
                break

            # execute
            try:
                self.execute_action(action, rows=rows, cols=cols)
            except Exception as e:
                print(f"[step {step + 1}] ERROR executing action: {e}")
                break
            time.sleep(3.0) # give UI sufficient time to update

        else:
            print(f"\n[agent] Reached max steps ({self.max_steps}) without finishing.")

        print(f"[agent] Trajectory saved to: {trajectory_path}")
