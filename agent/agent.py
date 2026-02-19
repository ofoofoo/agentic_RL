"""
Core agent loop.

Each step:
  1. Take a screenshot via AndroidController (ppadb)
  2. Build a prompt with task description + action history
  3. Call GeminiModel.generate() with the screenshot
  4. Parse the JSON response
  5. Dispatch the action to AndroidController
  6. Append the step to a JSONL trajectory log
"""

import json
import os
import time
from datetime import datetime

from .android_controller import AndroidController
from .model import GeminiModel
from .prompt import SYSTEM_PROMPT


class Agent:
    def __init__(self, config: dict):
        self.model = GeminiModel(
            api_key=config["GEMINI_API_KEY"],
            model_name=config["GEMINI_MODEL"],
        )
        self.controller = AndroidController(serial=config["DEVICE_SERIAL"])
        self.output_dir = config["OUTPUT_DIR"]
        self.max_steps = config.get("MAX_STEPS", 20)

        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def build_prompt(self, task: str, step: int, history: list[dict]) -> str:
        history_text = ""
        if history:
            lines = [
                f"  Step {i + 1}: {json.dumps(h)}"
                for i, h in enumerate(history)
            ]
            history_text = "Actions taken so far:\n" + "\n".join(lines) + "\n\n"

        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Task: {task}\n\n"
            f"{history_text}"
            f"Current step: {step + 1} / {self.max_steps}\n"
            f"What is the next action?"
        )

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, action: dict) -> None:
        name = action["action"]
        args = action.get("args", {})

        if name == "tap":
            self.controller.tap(args["x"], args["y"])
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
            pass  # Handled in run()
        else:
            raise ValueError(f"Unknown action: {name!r}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, task: str) -> None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_dir = os.path.join(self.output_dir, run_id)
        trajectory_path = os.path.join(self.output_dir, f"{run_id}_trajectory.jsonl")
        os.makedirs(screenshot_dir, exist_ok=True)

        print(f"\n[agent] Task: {task}")
        print(f"[agent] Output: {self.output_dir}/{run_id}/")
        print(f"[agent] Max steps: {self.max_steps}\n")

        history: list[dict] = []

        for step in range(self.max_steps):
            # 1. Screenshot
            screenshot_path = os.path.join(screenshot_dir, f"step_{step:03d}.png")
            self.controller.screenshot(screenshot_path)
            print(f"[step {step + 1}] Screenshot saved: {screenshot_path}")

            # 2. Build prompt
            prompt = self.build_prompt(task, step, history)

            # 3. Call Gemini
            raw_response = self.model.generate(prompt, image_path=screenshot_path)
            print(f"[step {step + 1}] Model response: {raw_response}")

            # 4. Parse JSON action — strip markdown fences if the model slips one in
            clean = raw_response.strip().removeprefix("```json").removesuffix("```").strip()
            try:
                action = json.loads(clean)
            except json.JSONDecodeError as e:
                print(f"[step {step + 1}] ERROR: could not parse response as JSON: {e}")
                print("         Raw response was:", repr(raw_response))
                break

            # 5. Log step
            record = {
                "step": step,
                "screenshot": screenshot_path,
                "action": action,
                "timestamp": time.time(),
            }
            with open(trajectory_path, "a") as f:
                f.write(json.dumps(record) + "\n")

            # 6. Check for done
            if action.get("action") == "done":
                print(f"\n[agent] Task complete after {step + 1} step(s).")
                break

            # 7. Execute
            try:
                self._dispatch(action)
            except Exception as e:
                print(f"[step {step + 1}] ERROR executing action: {e}")
                break

            # Small pause so the UI can settle before the next screenshot
            time.sleep(1.0)

        else:
            print(f"\n[agent] Reached max steps ({self.max_steps}) without finishing.")

        print(f"[agent] Trajectory saved to: {trajectory_path}")
