from __future__ import annotations
"""
Wraps agent logic into AndroidWorld's EnvironmentInteractingAgent interface.

Dual-mode UI interaction:
  1. Element mode (primary) — uiautomator dump + labeled elements
  2. Grid mode (fallback) — numbered grid overlay
"""

import io
import json
import os
import re
import subprocess
import tempfile
import time

from PIL import Image, ImageDraw, ImageFont

from android_world.agents import base_agent
from android_world.env import json_action, adb_utils, tools

from .android_controller import UIElement, _traverse_tree, MIN_DIST
from .agent import parse_element_response, parse_grid_response
from .model import GeminiModel, VLLMModel
from .prompt import (
    build_element_prompt,
    build_grid_prompt,
    build_element_text_list,
)

# constants for grid fallback
CELL_W, CELL_H = 54, 75
SCREEN_W, SCREEN_H = 1080, 2400
GRID_COLS = SCREEN_W // CELL_W   # 20
GRID_ROWS = SCREEN_H // CELL_H   # 32

# swipe distance/duration by dist name
_SWIPE_DIST_FRAC  = {"short": 0.25, "medium": 0.40, "long": 0.65}
_SWIPE_DURATION   = {"short": 400,  "medium": 600,  "long": 800}

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


# ── Helper: draw element labels ─────────────────────────────────────
def _draw_element_labels(img: Image.Image, elem_list: list[UIElement]) -> Image.Image:
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=28)
    except OSError:
        font = ImageFont.load_default()

    for idx, elem in enumerate(elem_list, 1):
        (x1, y1), (x2, y2) = elem.bbox
        # Normalize: uiautomator can return inverted coords (x1>x2 or y1>y2)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        # Skip zero-area elements — nothing visible to draw
        if x1 == x2 or y1 == y2:
            continue
        draw.rectangle([x1, y1, x2, y2], outline=(255, 116, 113), width=3)
        # label at center
        cx, cy = elem.center
        label = str(idx)
        tw = len(label) * 14 + 8
        th = 28
        lx = cx - tw // 2
        ly = cy - th // 2
        draw.rectangle([lx, ly, lx + tw, ly + th], fill=(0, 0, 0, 180))
        draw.text((lx + 4, ly + 2), label, fill=(255, 116, 113), font=font)
    return img

def _annotate_thinking(img: Image.Image, thinking: str) -> Image.Image:
    import textwrap
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=24)
    except OSError:
        font = ImageFont.load_default()
    
    sidebar_width = 800
    new_height = max(img.height, 1200)
    new_img = Image.new("RGB", (img.width + sidebar_width, new_height), "white")
    new_img.paste(img, (0, 0))
    
    draw = ImageDraw.Draw(new_img)
    wrapped_text = textwrap.fill(thinking, width=65)
    draw.multiline_text((img.width + 20, 20), wrapped_text, fill=(0, 0, 0), font=font, spacing=8)
    
    return new_img


def _process_aw_ui_elements(aw_elements: list) -> list[UIElement]:
    """
    Convert AndroidWorld's State.ui_elements into our UIElement format,
    applying deduplication based on MIN_DIST.
    """
    import hashlib
    
    clickable = []
    other = []
    
    for e in aw_elements:
        if not e.bbox_pixels: continue
        x1, x2 = int(e.bbox_pixels.x_min), int(e.bbox_pixels.x_max)
        y1, y2 = int(e.bbox_pixels.y_min), int(e.bbox_pixels.y_max)
        
        # Normalize just in case
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        
        # skip zero area
        if x1 == x2 or y1 == y2:
            continue
            
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Synthetic stable UID
        uid_str = f"{e.resource_name}_{e.class_name}_{x1}_{y1}_{x2}_{y2}"
        uid = hashlib.md5(uid_str.encode()).hexdigest()[:8]
        
        if e.is_clickable:
            attrib = "clickable"
        elif e.is_focusable:
            attrib = "focusable"
        elif e.is_scrollable:
            attrib = "scrollable"
        else:
            attrib = "visible"
            
        ui_elem = UIElement(
            uid=uid,
            bbox=((x1, y1), (x2, y2)),
            center=(cx, cy),
            attrib=attrib,
            text=e.text or "",
            content_desc=e.content_description or ""
        )
        
        if e.is_clickable:
            clickable.append(ui_elem)
        else:
            other.append(ui_elem)
            
    # deduplicate: clickable first
    merged = []
    for ce in clickable:
        close = False
        for me in merged:
            dist = ((ce.center[0] - me.center[0]) ** 2 + (ce.center[1] - me.center[1]) ** 2) ** 0.5
            if dist <= MIN_DIST:
                close = True
                break
        if not close:
            merged.append(ce)
            
    for oe in other:
        close = False
        for me in merged:
            dist = ((oe.center[0] - me.center[0]) ** 2 + (oe.center[1] - me.center[1]) ** 2) ** 0.5
            if dist <= MIN_DIST:
                close = True
                break
        if not close:
            merged.append(oe)
            
    return merged


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


def _action_to_aw(
    parsed_action: dict,
    elem_list: list[UIElement] | None = None,
) -> json_action.JSONAction:
    """Convert parsed action to AndroidWorld JSONAction."""
    name = parsed_action["action"]

    # ── Element-mode actions ─────────────────────────────────────────
    if name == "tap":
        idx = parsed_action["element"]
        if elem_list and 1 <= idx <= len(elem_list):
            x, y = elem_list[idx - 1].center
        else:
            raise ValueError(f"Element {idx} out of range (have {len(elem_list or [])})")
        return json_action.JSONAction(action_type=json_action.CLICK, x=x, y=y)

    if name == "long_press":
        idx = parsed_action["element"]
        if elem_list and 1 <= idx <= len(elem_list):
            x, y = elem_list[idx - 1].center
        else:
            raise ValueError(f"Element {idx} out of range")
        return json_action.JSONAction(action_type=json_action.LONG_PRESS, x=x, y=y)

    if name == "swipe":
        idx = parsed_action["element"]
        if elem_list and 1 <= idx <= len(elem_list):
            x, y = elem_list[idx - 1].center
        else:
            raise ValueError(f"Element {idx} out of range")
        return json_action.JSONAction(
            action_type=json_action.SWIPE,
            x=x, y=y,
            direction=parsed_action["direction"],
        )
    
    if name == "open":
        return json_action.JSONAction(
            action_type=json_action.OPEN_APP,
            app_name=parsed_action["app"],
        )

    # ── Grid-mode actions ────────────────────────────────────────────
    if name == "tap_grid":
        x, y = _area_to_xy(parsed_action["area"], parsed_action.get("subarea", "center"))
        return json_action.JSONAction(action_type=json_action.CLICK, x=x, y=y)

    if name == "long_press_grid":
        x, y = _area_to_xy(parsed_action["area"], parsed_action.get("subarea", "center"))
        return json_action.JSONAction(action_type=json_action.LONG_PRESS, x=x, y=y)

    if name == "swipe_grid":
        sx, sy = _area_to_xy(parsed_action["start_area"], parsed_action["start_subarea"])
        ex, ey = _area_to_xy(parsed_action["end_area"], parsed_action["end_subarea"])
        dx, dy = ex - sx, ey - sy
        if abs(dx) >= abs(dy):
            direction = "right" if dx > 0 else "left"
        else:
            direction = "down" if dy > 0 else "up"
        return json_action.JSONAction(
            action_type=json_action.SWIPE,
            x=sx, y=sy,
            direction=direction,
        )

    # ── Common actions ───────────────────────────────────────────────
    if name == "text":
        return json_action.JSONAction(
            action_type=json_action.INPUT_TEXT,
            text=parsed_action["text"],
        )

    if name == "answer":
        return json_action.JSONAction(
            action_type=getattr(json_action, "ANSWER", "answer"),
            text=parsed_action["text"],
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

        self.element_prompt = build_element_prompt(SCREEN_W, SCREEN_H)
        self.grid_prompt = build_grid_prompt(SCREEN_W, SCREEN_H, CELL_W, CELL_H)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # ADB path for uiautomator
        self._adb_path = os.path.expanduser(
            config.get("ADB_PATH", "~/Library/Android/sdk/platform-tools/adb")
        )

        self.max_history_steps = config.get("MAX_HISTORY_STEPS", 0)
        print(f"max history steps: {self.max_history_steps}")
        self._history: list[dict] = []
        self._step_count = 0
        self._grid_on = False
        self._elem_list: list[UIElement] = []

    def reset_episode(self) -> None:
        self._history = []
        self._step_count = 0
        self._grid_on = False
        self._elem_list = []
    
    def initialize_chrome(self):
        print("Running additional chrome initialization...")
        # handle chrome initialization problem for browser tasks
        adb_utils.launch_app("chrome", self.env.controller)
        time.sleep(5)

        tool_controller = tools.AndroidToolController(env=self.env.controller)
        time.sleep(2)

        first_op = False
        try:
            print("try first variant...")
            tool_controller.click_element("Use without an account")
            time.sleep(5.0)
            first_op = True
        except:
            print("Failed to click 'Use without an account' button.")
            pass
        
        if not first_op:
            print("try second variant...")
            try:
                tool_controller.click_element("Accept & continue")
            except:
                pass
            time.sleep(3.0)
            try:
                tool_controller.click_element("No thanks")
            except:
                pass
            time.sleep(5.0)
        
        adb_utils.press_home_button(self.env.controller)
        time.sleep(2.0)
        print("Done additional chrome initialization")

    def _build_prompt(self, goal: str) -> str:
        sys_prompt = self.grid_prompt if self._grid_on else self.element_prompt

        # Full text history
        history_text = ""
        if self._history:
            lines = [f"  Step {i + 1}: {h['summary']}" for i, h in enumerate(self._history)]
            history_text = "Actions taken so far:\n" + "\n".join(lines) + "\n\n"

        elem_text = ""
        if not self._grid_on and self._elem_list:
            elem_text = build_element_text_list(self._elem_list) + "\n\n"

        max_steps = self._max_steps or 25
        return (
            f"{sys_prompt}\n\n"
            f"Task: {goal}\n\n"
            f"{elem_text}"
            f"{history_text}"
            f"Current step: {self._step_count + 1} / {max_steps}\n"
            f"What is the next action?"
        )

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        if self._step_count == 0 and "chrome" in goal.lower():
            self.initialize_chrome()

        self._step_count += 1
        t_step_start = time.perf_counter()

        # 1. screenshot env, includes transition pause
        t0 = time.perf_counter()
        state = self.get_post_transition_state()
        t_screenshot = time.perf_counter() - t0
        pixels = state.pixels  # numpy array (H, W, 3)
        img = Image.fromarray(pixels).convert("RGB")

        # 2. observation: element mode or grid mode
        t0 = time.perf_counter()
        image_path = os.path.join(self.output_dir, f"step_{self._step_count:03d}.png")
        if self._grid_on: # this is grid mode
            grid_img = _draw_numbered_grid(img.copy())
            grid_img.save(image_path) # Temporarily save for the model to read
            mode_img = grid_img
            self._elem_list = []
            mode_str = f"grid ({GRID_ROWS}x{GRID_COLS})"
        else: # this is element mode
            # process UI from synced state
            self._elem_list = _process_aw_ui_elements(state.ui_elements)
            labeled_img = _draw_element_labels(img.copy(), self._elem_list)
            labeled_img.save(image_path)  # Temporarily save for the model to read
            mode_img = labeled_img
            mode_str = f"element ({len(self._elem_list)} elements)"
        t_preprocess = time.perf_counter() - t0

        # 3. build prompt
        t0 = time.perf_counter()
        prompt = self._build_prompt(goal)
        t_prompt = time.perf_counter() - t0

        # 4. inference
        t0 = time.perf_counter()
        history_window = self._history[-self.max_history_steps:] if self.max_history_steps > 0 else []
        raw_response, token_usage = self.model.generate(prompt, image_path=image_path, history=history_window)
        t_inference = time.perf_counter() - t0

        img_annotated = _annotate_thinking(mode_img, raw_response)
        img_annotated.save(image_path)

        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        total_tokens = token_usage.get("total_tokens", 0)
        ttft_s = token_usage.get("ttft_s", 0.0)
        decode_s = token_usage.get("decode_s", 0.0)
        tpot_s = token_usage.get("tpot_s", 0.0)
        print(
            f"\033[36m ====================[step {self._step_count}]mode={mode_str}====================\033[0m"
            f"screenshot={t_screenshot:.2f}s  "
            f"preprocess={t_preprocess:.2f}s  "
            f"inference={t_inference:.2f}s (ttft={ttft_s:.3f}s / decode={decode_s:.3f}s / tpot={tpot_s*1000:.1f}ms)  "
            f"tokens(prompt={prompt_tokens} / completion={completion_tokens} / total={total_tokens})"
            f"\nTASK: {goal}"
        )
        print(raw_response)

        if not raw_response:
            print(f"  [step {self._step_count}] ERROR: model returned empty/None response")
            return base_agent.AgentInteractionResult(done=True, data={"error": "empty_response"})

        # 5. parse structured response
        if self._grid_on:
            result = parse_grid_response(raw_response)
        else:
            result = parse_element_response(raw_response)

        if result is None:
            print(f"  [step {self._step_count}] ERROR: could not parse structured response")
            print("  Raw response was:", repr(raw_response))
            return base_agent.AgentInteractionResult(done=True, data={"error": "parse_failed"})

        parsed_action = result["parsed_action"]
        is_done = parsed_action["action"] == "done"

        # 6. handle grid toggle
        if parsed_action["action"] == "grid":
            self._grid_on = True
            print(f"  [step {self._step_count}] Switching to grid mode")
            self._history.append({
                "summary": result.get("summary", "Switched to grid mode"),
                "action": parsed_action,
                "image_path": image_path,
            })
            # don't execute, just re-loop
            return base_agent.AgentInteractionResult(
                done=False,
                data={
                    "step": self._step_count,
                    "action": parsed_action,
                    "latency": {
                        "screenshot_s":  round(t_screenshot, 3),
                        "preprocess_s":  round(t_preprocess, 3),
                        "prompt_s":      round(t_prompt, 3),
                        "inference_s":   round(t_inference, 3),
                        "action_s":      0,
                        "step_total_s":  round(t_step_total, 3),
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens":  total_tokens,
                        "ttft_s":        ttft_s,
                        "decode_s":      decode_s,
                        "tpot_ms":       round(tpot_s * 1000, 2),
                    },
                    "image_path": image_path,
                    "mode": "grid" if self._grid_on else "element",
                },
            )
        else:
            # after any non-grid action, switch back to element mode
            self._grid_on = False

        # 7. execute action
        t0 = time.perf_counter()
        if not is_done:
            try:
                if parsed_action["action"] == "swipe":
                    # Issue the ADB swipe directly so we control distance & speed, as aw swipe is too fast
                    elem = self._elem_list[parsed_action["element"] - 1]
                    x, y = elem.center
                    dist_name = parsed_action.get("dist", "medium")
                    direction = parsed_action["direction"]
                    dist_px  = int(SCREEN_H * _SWIPE_DIST_FRAC.get(dist_name, 0.40))
                    duration = _SWIPE_DURATION.get(dist_name, 600)
                    dx = {"left": -dist_px, "right": dist_px, "up": 0,        "down": 0}.get(direction, 0)
                    dy = {"left": 0,        "right": 0,        "up": -dist_px, "down": dist_px}.get(direction, 0)
                    x2 = max(0, min(SCREEN_W - 1, x + dx))
                    y2 = max(0, min(SCREEN_H - 1, y + dy))
                    print(f"[aw_adapter] direct ADB swipe ({x},{y})\u2192({x2},{y2}) {direction} {dist_name} {dist_px}px {duration}ms")
                    subprocess.run(
                        [self._adb_path, "shell", "input", "swipe",
                         str(x), str(y), str(x2), str(y2), str(duration)],
                        timeout=10,
                    )
                elif parsed_action["action"] == "clear_text":
                    print("[aw_adapter] clear_text: keycombination 113 29 + keyevent 67")
                    subprocess.run(
                        [self._adb_path, "shell", "input", "keycombination", "113 29"],
                        timeout=5,
                    )
                    subprocess.run(
                        [self._adb_path, "shell", "input", "keyevent", "67"],
                        timeout=5,
                    )
                elif parsed_action["action"] == "enter":
                    print("[aw_adapter] enter: KEYCODE_ENTER")
                    subprocess.run(
                        [self._adb_path, "shell", "input", "keyevent", "KEYCODE_ENTER"],
                        timeout=5,
                    )
                elif parsed_action["action"] == "wait":
                    sec = parsed_action.get("time", 2)
                    print(f"[aw_adapter] wait: {sec}s")
                    time.sleep(sec)
                elif parsed_action["action"] == "scroll":
                    direction = parsed_action["direction"]
                    cx = SCREEN_W // 2         # 540
                    cy = SCREEN_H // 2         # 1200
                    dist_px = int(SCREEN_H * 0.50)  # 50% of screen = 1200px — long visible scroll
                    duration = 600
                    dy = -dist_px if direction == "up" else dist_px
                    y2 = max(0, min(SCREEN_H - 1, cy + dy))
                    print(f"[aw_adapter] scroll {direction}: ADB swipe ({cx},{cy})\u2192({cx},{y2}) {dist_px}px {duration}ms")
                    subprocess.run(
                        [self._adb_path, "shell", "input", "swipe",
                         str(cx), str(cy), str(cx), str(y2), str(duration)],
                        timeout=10,
                    )
                else:
                    aw_action = _action_to_aw(parsed_action, elem_list=self._elem_list)
                    self._env.execute_action(aw_action)
            except Exception as e:
                print(f"[step {self._step_count}] ERROR executing: {e}")
                return base_agent.AgentInteractionResult(done=True, data={"error": str(e)})
        t_action = time.perf_counter() - t0
        t_step_total = time.perf_counter() - t_step_start

        # 8. update history
        self._history.append({
            "summary": result.get("summary", raw_response[:100]),
            "action": parsed_action,
            "image_path": image_path,
        })

        return base_agent.AgentInteractionResult(
            done=is_done,
            data={
                "step": self._step_count,
                "action": parsed_action,
                "latency": {
                    "screenshot_s":  round(t_screenshot, 3),
                    "preprocess_s":  round(t_preprocess, 3),
                    "prompt_s":      round(t_prompt, 3),
                    "inference_s":   round(t_inference, 3),
                    "action_s":      round(t_action, 3),
                    "step_total_s":  round(t_step_total, 3),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens":  total_tokens,
                    "ttft_s":        ttft_s,
                    "decode_s":      decode_s,
                    "tpot_ms":       round(tpot_s * 1000, 2),
                },
                "image_path": image_path,
                "mode": "grid" if self._grid_on else "element",
                "n_elements": len(self._elem_list),
            },
        )
