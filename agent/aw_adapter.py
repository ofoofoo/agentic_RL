"""
Wraps agent logic into AndroidWorld's EnvironmentInteractingAgent interface.
"""

import io
import json
import os
import time

from PIL import Image, ImageDraw, ImageFont

from android_world.agents import base_agent
from android_world.env import json_action

from .model import GeminiModel, VLLMModel
from .prompt import build_system_prompt

# constants for setting the grid for screenshots
CELL_W, CELL_H = 54, 75
SCREEN_W, SCREEN_H = 1080, 2400
GRID_COLS = SCREEN_W // CELL_W   # 20
GRID_ROWS = SCREEN_H // CELL_H   # 32

# helper functions:
def _draw_numbered_grid(img: Image.Image) -> Image.Image:
    draw = ImageDraw.Draw(img)
    color = (255, 116, 113)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=25)
    except OSError:
        font = ImageFont.load_default()

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            label = r * GRID_COLS + c + 1
            x0, y0 = c * CELL_W, r * CELL_H
            x1, y1 = x0 + CELL_W, y0 + CELL_H
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            draw.text((x0 + 4, y0 + 4), str(label), fill=color, font=font)
    return img


def _area_to_xy(area: int, subarea: str) -> tuple[int, int]:
    area -= 1
    row, col = area // GRID_COLS, area % GRID_COLS
    x0, y0 = col * CELL_W, row * CELL_H
    offsets = {
        "top-left":     (CELL_W // 4,     CELL_H // 4),
        "top":          (CELL_W // 2,     CELL_H // 4),
        "top-right":    (CELL_W * 3 // 4, CELL_H // 4),
        "left":         (CELL_W // 4,     CELL_H // 2),
        "center":       (CELL_W // 2,     CELL_H // 2),
        "right":        (CELL_W * 3 // 4, CELL_H // 2),
        "bottom-left":  (CELL_W // 4,     CELL_H * 3 // 4),
        "bottom":       (CELL_W // 2,     CELL_H * 3 // 4),
        "bottom-right": (CELL_W * 3 // 4, CELL_H * 3 // 4),
    }
    dx, dy = offsets.get(subarea, (CELL_W // 2, CELL_H // 2))
    return x0 + dx, y0 + dy


def _our_action_to_aw(action: dict) -> json_action.JSONAction:
    """JSON action dict -> AndroidWorld JSONAction."""
    name = action["action"]
    args = action.get("args", {})

    if name == "tap":
        if "area" in args:
            x, y = _area_to_xy(args["area"], args.get("subarea", "center"))
        else:
            x, y = args["x"], args["y"]
        return json_action.JSONAction(action_type=json_action.CLICK, x=x, y=y)

    if name == "swipe":
        dx = args["x2"] - args["x1"]
        dy = args["y2"] - args["y1"]
        if abs(dx) >= abs(dy):
            direction = "right" if dx > 0 else "left"
        else:
            direction = "down" if dy > 0 else "up"
        return json_action.JSONAction(
            action_type=json_action.SWIPE,
            x=args["x1"], y=args["y1"],
            direction=direction,
        )

    if name == "type":
        return json_action.JSONAction(
            action_type=json_action.INPUT_TEXT,
            text=args["text"],
        )

    if name == "back":
        return json_action.JSONAction(action_type=json_action.NAVIGATE_BACK)

    if name == "home":
        return json_action.JSONAction(action_type=json_action.NAVIGATE_HOME)

    if name == "done":
        return json_action.JSONAction(
            action_type=json_action.STATUS,
            goal_status="complete",
        )

    raise ValueError(f"Unknown action: {name!r}")


class AWAgentAdapter(base_agent.EnvironmentInteractingAgent):
    """Wraps Gemini/vLLM agent to run inside AndroidWorld's harness."""

    def __init__(
        self,
        env,
        config: dict,
        output_dir: str = "./output/aw_runs",
        transition_pause: float = 3.0,
    ):
        super().__init__(env=env, name="agentic_rl", transition_pause=transition_pause)

        backend = config.get("BACKEND", "gemini").lower()
        if backend == "vllm":
            self.model = VLLMModel(
                api_key=config["VLLM_API_KEY"],
                model_name=config["VLLM_MODEL"],
                base_url=config.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
            )
        else:
            self.model = GeminiModel(
                api_key=config["GEMINI_API_KEY"],
                model_name=config["GEMINI_MODEL"],
            )

        self.system_prompt = build_system_prompt(SCREEN_W, SCREEN_H, CELL_W, CELL_H)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self._history: list[dict] = []
        self._step_count = 0

    def reset_episode(self) -> None:
        self._history = []
        self._step_count = 0

    def _build_prompt(self, goal: str) -> str:
        history_text = ""
        if self._history:
            lines = []
            for i, h in enumerate(self._history):
                lines.append(f"  Step {i + 1} reasoning: {h['reasoning']}")
                lines.append(f"  Step {i + 1} action:    {json.dumps(h['action'])}")
            history_text = "Actions taken so far:\n" + "\n".join(lines) + "\n\n"

        max_steps = self._max_steps or 25
        return (
            f"{self.system_prompt}\n\n"
            f"Task: {goal}\n\n"
            f"{history_text}"
            f"Current step: {self._step_count + 1} / {max_steps}\n"
            f"What is the next action?"
        )

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        self._step_count += 1
        t_step_start = time.perf_counter()

        # 1. screenshot env, includes transition pause
        t0 = time.perf_counter()
        state = self.get_post_transition_state()
        t_screenshot = time.perf_counter() - t0
        pixels = state.pixels  # numpy array (H, W, 3)
        img = Image.fromarray(pixels).convert("RGB")

        # 2. pre-process and draw grid
        t0 = time.perf_counter()
        grid_img = _draw_numbered_grid(img)
        grid_path = os.path.join(self.output_dir, f"step_{self._step_count:03d}_grid.png")
        grid_img.save(grid_path)
        t_preprocess = time.perf_counter() - t0

        # 3. build prompt with ICL examples
        t0 = time.perf_counter()
        prompt = self._build_prompt(goal)
        t_prompt = time.perf_counter() - t0

        # 4. inference
        t0 = time.perf_counter()
        raw_response = self.model.generate(prompt, image_path=grid_path)
        t_inference = time.perf_counter() - t0
        t_step_total = time.perf_counter() - t_step_start

        print(
            f"[step {self._step_count}] "
            f"raw response: {raw_response} "
            f"screenshot={t_screenshot:.2f}s  "
            f"preprocess={t_preprocess:.2f}s  "
            f"prompt={t_prompt:.2f}s  "
            f"inference={t_inference:.2f}s  "
            f"total={t_step_total:.2f}s"
        )

        if not raw_response:
            print(f"  [step {self._step_count}] ERROR: model returned empty/None response (check model name in config.yaml matches the server)")
            return base_agent.AgentInteractionResult(done=True, data={"error": "empty_response"})

        # 5. parse action fron JSON
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
            print(f"  [step {self._step_count}] ERROR: no valid JSON in response")
            return base_agent.AgentInteractionResult(done=True, data={"error": "no_json"})

        is_done = action.get("action") == "done"

        # 6. execute action
        if not is_done:
            try:
                aw_action = _our_action_to_aw(action)
                self._env.execute_action(aw_action)
            except Exception as e:
                print(f"[step {self._step_count}] ERROR executing: {e}")
                return base_agent.AgentInteractionResult(done=True, data={"error": str(e)})

        # 7. update history
        self._history.append({"reasoning": raw_response, "action": action})

        return base_agent.AgentInteractionResult(
            done=is_done,
            data={
                "step": self._step_count,
                "action": action,
                "latency": {
                    "screenshot_s":  round(t_screenshot, 3),
                    "preprocess_s":  round(t_preprocess, 3),
                    "prompt_s":      round(t_prompt, 3),
                    "inference_s":   round(t_inference, 3),
                    "step_total_s":  round(t_step_total, 3),
                },
                "grid_path": grid_path,
            },
        )
