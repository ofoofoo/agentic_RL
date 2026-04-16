from __future__ import annotations
"""
Wraps agent logic into AndroidWorld's EnvironmentInteractingAgent interface.
"""

import os
import subprocess
import time

from PIL import Image, ImageDraw, ImageFont

from android_world.agents import base_agent
from android_world.env import json_action, adb_utils, tools

from .android_controller import UIElement, _traverse_tree, MIN_DIST
from .parse import parse_element_response, parse_grid_response, parse_rawcoord_response
from .model import GeminiModel, VLLMModel
from .prompt import (
    build_element_prompt,
    build_grid_prompt,
    build_coarse_grid_prompt,
    build_fine_grid_prompt,
    build_rawcoord_prompt,
    build_element_text_list,
)

SCREEN_W, SCREEN_H = 1080, 2400

DEFAULT_GRID_ROWS = 8
DEFAULT_GRID_COLS = 10

DEFAULT_COARSE_ROWS = 6
DEFAULT_COARSE_COLS = 4
DEFAULT_FINE_ROWS = 8
DEFAULT_FINE_COLS = 6
FINE_IMG_TARGET_SIZE = (1080, 1080)

# swipe distance/duration by dist name
_SWIPE_DIST_FRAC  = {"short": 0.25, "medium": 0.40, "long": 0.65}
_SWIPE_DURATION   = {"short": 400,  "medium": 600,  "long": 800}

def _draw_numbered_grid(
    img: Image.Image,
    grid_rows: int,
    grid_cols: int,
    cell_w: int,
    cell_h: int,
) -> Image.Image:
    draw = ImageDraw.Draw(img)
    color = (255, 116, 113)
    font_size = max(20, min(cell_w, cell_h) // 4)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size=font_size)
    except OSError:
        font = ImageFont.load_default()

    for r in range(grid_rows):
        for c in range(grid_cols):
            label = r * grid_cols + c + 1
            x0, y0 = c * cell_w, r * cell_h
            x1, y1 = x0 + cell_w, y0 + cell_h
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
    for elem in clickable + other:
        if not any(((elem.center[0] - m.center[0]) ** 2 + (elem.center[1] - m.center[1]) ** 2) ** 0.5 <= MIN_DIST for m in merged):
            merged.append(elem)
            
    return merged


def _area_to_xy(
    area: int,
    subarea: str,
    grid_cols: int,
    cell_w: int,
    cell_h: int,
) -> tuple[int, int]:
    area -= 1
    row, col = area // grid_cols, area % grid_cols
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


def _get_element_center(idx: int, elem_list: list[UIElement]) -> tuple[int, int]:
    if elem_list and 1 <= idx <= len(elem_list):
        return elem_list[idx - 1].center
    raise ValueError(f"Element {idx} out of range (have {len(elem_list or [])})")

def _action_to_aw(
    parsed_action: dict,
    elem_list: list[UIElement] | None = None,
    grid_cols: int = DEFAULT_GRID_COLS,
    cell_w: int = SCREEN_W // DEFAULT_GRID_COLS,
    cell_h: int = SCREEN_H // DEFAULT_GRID_ROWS,
) -> json_action.JSONAction:
    """Convert parsed action to AndroidWorld JSONAction."""
    name = parsed_action["action"]

    # ── Element-mode actions ─────────────────────────────────────────
    if name == "tap":
        x, y = _get_element_center(parsed_action["element"], elem_list)
        return json_action.JSONAction(action_type=json_action.CLICK, x=x, y=y)

    if name == "long_press":
        x, y = _get_element_center(parsed_action["element"], elem_list)
        return json_action.JSONAction(action_type=json_action.LONG_PRESS, x=x, y=y)

    if name == "swipe":
        x, y = _get_element_center(parsed_action["element"], elem_list)
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
        x, y = _area_to_xy(parsed_action["area"], parsed_action.get("subarea", "center"), grid_cols, cell_w, cell_h)
        return json_action.JSONAction(action_type=json_action.CLICK, x=x, y=y)

    if name == "long_press_grid":
        x, y = _area_to_xy(parsed_action["area"], parsed_action.get("subarea", "center"), grid_cols, cell_w, cell_h)
        return json_action.JSONAction(action_type=json_action.LONG_PRESS, x=x, y=y)

    if name == "swipe_grid":
        sx, sy = _area_to_xy(parsed_action["start_area"], parsed_action["start_subarea"], grid_cols, cell_w, cell_h)
        ex, ey = _area_to_xy(parsed_action["end_area"], parsed_action["end_subarea"], grid_cols, cell_w, cell_h)
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
    """Wraps agent to run inside AndroidWorld's harness."""

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

        self.agent_mode = config.get("AGENT_MODE", "element")

        self._grid_rows = config.get("GRID_ROWS", DEFAULT_GRID_ROWS)
        self._grid_cols = config.get("GRID_COLS", DEFAULT_GRID_COLS)
        self._cell_w = SCREEN_W // self._grid_cols
        self._cell_h = SCREEN_H // self._grid_rows
        print(f"grid: {self._grid_rows}x{self._grid_cols} = {self._grid_rows * self._grid_cols} cells, "
              f"cell size {self._cell_w}x{self._cell_h} px")

        self._coarse_rows = config.get("COARSE_GRID_ROWS", DEFAULT_COARSE_ROWS)
        self._coarse_cols = config.get("COARSE_GRID_COLS", DEFAULT_COARSE_COLS)
        self._coarse_cell_w = SCREEN_W // self._coarse_cols
        self._coarse_cell_h = SCREEN_H // self._coarse_rows
        self._fine_rows = config.get("FINE_GRID_ROWS", DEFAULT_FINE_ROWS)
        self._fine_cols = config.get("FINE_GRID_COLS", DEFAULT_FINE_COLS)
        if self.agent_mode == "grid2level":
            print(f"grid2level: coarse {self._coarse_rows}x{self._coarse_cols} "
                  f"({self._coarse_cell_w}x{self._coarse_cell_h} px/cell), "
                  f"fine {self._fine_rows}x{self._fine_cols}")

        self.element_prompt = build_element_prompt(SCREEN_W, SCREEN_H)
        self.grid_prompt = build_grid_prompt(SCREEN_W, SCREEN_H, self._cell_w, self._cell_h)
        self.coarse_prompt = build_coarse_grid_prompt(
            SCREEN_W, SCREEN_H, self._coarse_cell_w, self._coarse_cell_h,
            self._coarse_rows, self._coarse_cols)
        self.rawcoord_prompt = build_rawcoord_prompt(SCREEN_W, SCREEN_H)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self._adb_path = os.path.expanduser(
            config.get("ADB_PATH", "") or os.path.expanduser("~/Android/Sdk/platform-tools/adb")
        )

        self.max_history_steps = config.get("MAX_HISTORY_STEPS", 0)
        print(f"max history steps: {self.max_history_steps}")
        self._grid_only = self.agent_mode == "grid"
        self._history: list[dict] = []
        self._step_count = 0
        self._grid_on = self._grid_only
        self._elem_list: list[UIElement] = []

    def _adb_shell(self, *args, timeout: int = 5):
        return subprocess.run([self._adb_path, "shell"] + list(args), timeout=timeout)

    def _build_latency_dict(
        self, t_screenshot: float, t_preprocess: float, t_prompt: float,
        t_inference: float, t_action: float, t_step_total: float, token_usage: dict
    ) -> dict:
        return {
            "screenshot_s":  round(t_screenshot, 3),
            "preprocess_s":  round(t_preprocess, 3),
            "prompt_s":      round(t_prompt, 3),
            "inference_s":   round(t_inference, 3),
            "action_s":      round(t_action, 3),
            "step_total_s":  round(t_step_total, 3),
            "prompt_tokens": token_usage.get("prompt_tokens", 0),
            "completion_tokens": token_usage.get("completion_tokens", 0),
            "total_tokens":  token_usage.get("total_tokens", 0),
            "ttft_s":        token_usage.get("ttft_s", 0.0),
            "decode_s":      token_usage.get("decode_s", 0.0),
            "tpot_ms":       round(token_usage.get("tpot_s", 0.0) * 1000, 2),
        }

    def reset_episode(self) -> None:
        self._history = []
        self._step_count = 0
        self._grid_on = self._grid_only
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

    def _build_prompt(self, goal: str, is_coarse: bool = False) -> str:
        if is_coarse:
            sys_prompt = self.coarse_prompt
        elif self.agent_mode == "raw":
            sys_prompt = self.rawcoord_prompt
        elif self._grid_on:
            sys_prompt = self.grid_prompt
        else:
            sys_prompt = self.element_prompt

        # Full text history
        history_text = ""
        if self._history:
            lines = [f"  Step {i + 1}: {h['summary']}" for i, h in enumerate(self._history)]
            history_text = "Actions taken so far:\n" + "\n".join(lines) + "\n\n"

        elem_text = ""
        if not self._grid_on and not is_coarse and self.agent_mode != "raw" and self._elem_list:
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

    def _build_fine_prompt(self, goal: str, fine_cell_w: int, fine_cell_h: int) -> str:
        fine_sys = build_fine_grid_prompt(
            SCREEN_W, SCREEN_H, fine_cell_w, fine_cell_h,
            self._fine_rows, self._fine_cols)
        return (
            f"{fine_sys}\n\n"
            f"Task: {goal}\n\n"
            f"Target the element you need to interact with."
        )

    def _coarse_area_to_crop(self, area: int) -> tuple[int, int, int, int]:
        max_area = self._coarse_rows * self._coarse_cols
        area = max(1, min(area, max_area))
        area -= 1
        row, col = area // self._coarse_cols, area % self._coarse_cols
        x0 = col * self._coarse_cell_w
        y0 = row * self._coarse_cell_h
        x1 = x0 + self._coarse_cell_w
        y1 = y0 + self._coarse_cell_h
        return x0, y0, x1, y1

    def _step_grid2level(self, goal: str) -> base_agent.AgentInteractionResult:
        """2-level hierarchical grid: coarse -> zoom -> fine."""
        if self._step_count == 0 and "chrome" in goal.lower():
            self.initialize_chrome()

        self._step_count += 1
        t_step_start = time.perf_counter()

        t0 = time.perf_counter()
        state = self.get_post_transition_state()
        t_screenshot = time.perf_counter() - t0
        pixels = state.pixels
        img = Image.fromarray(pixels).convert("RGB")

        t0 = time.perf_counter()
        coarse_path = os.path.join(self.output_dir, f"step_{self._step_count:03d}_coarse.png")
        coarse_img = _draw_numbered_grid(
            img.copy(), self._coarse_rows, self._coarse_cols,
            self._coarse_cell_w, self._coarse_cell_h)
        coarse_img.save(coarse_path)
        t_preprocess = time.perf_counter() - t0

        t0 = time.perf_counter()
        coarse_prompt = self._build_prompt(goal, is_coarse=True)
        t_prompt = time.perf_counter() - t0

        t0 = time.perf_counter()
        history_window = self._history[-self.max_history_steps:] if self.max_history_steps > 0 else []
        coarse_raw, coarse_usage = self.model.generate(
            coarse_prompt, image_path=coarse_path, history=history_window)
        t_inference_coarse = time.perf_counter() - t0

        coarse_annotated = _annotate_thinking(coarse_img, coarse_raw)
        coarse_annotated.save(coarse_path)

        print(
            f"\033[36m ==[step {self._step_count} COARSE]=="
            f" ({self._coarse_rows}x{self._coarse_cols})==\033[0m"
            f" inference={t_inference_coarse:.2f}s"
            f"\nTASK: {goal}"
        )
        print(coarse_raw)

        if not coarse_raw:
            print(f"  [step {self._step_count}] WARNING: coarse model returned empty response")
            return base_agent.AgentInteractionResult(done=False, data={"error": "empty_response"})

        coarse_result = parse_element_response(coarse_raw)
        if coarse_result is None:
            print(f"  [step {self._step_count}] WARNING: could not parse coarse response")
            self._history.append({"summary": "Parse error, retrying", "action": {"action": "noop"}, "image_path": coarse_path})
            t_step_total = time.perf_counter() - t_step_start
            return base_agent.AgentInteractionResult(done=False, data={
                "step": self._step_count,
                "latency": self._build_latency_dict(t_screenshot, t_preprocess, t_prompt, t_inference_coarse, 0.0, t_step_total, coarse_usage),
            })

        coarse_action = coarse_result["parsed_action"]

        targeting_actions = {"tap", "long_press", "zoom", "tap_grid", "long_press_grid"}
        is_targeting = coarse_action["action"] in targeting_actions

        if not is_targeting:
            is_done = coarse_action["action"] == "done"
            t0 = time.perf_counter()
            if not is_done:
                try:
                    self._execute_non_grid_action(coarse_action)
                except Exception as e:
                    print(f"[step {self._step_count}] WARNING executing (continuing): {e}")
            t_action = time.perf_counter() - t0
            t_step_total = time.perf_counter() - t_step_start

            self._history.append({
                "summary": coarse_result.get("summary", coarse_raw[:100]),
                "action": coarse_action,
                "image_path": coarse_path,
            })
            return base_agent.AgentInteractionResult(
                done=is_done,
                data={
                    "step": self._step_count,
                    "action": coarse_action,
                    "latency": self._build_latency_dict(
                        t_screenshot, t_preprocess, t_prompt,
                        t_inference_coarse, t_action, t_step_total, coarse_usage),
                    "image_path": coarse_path,
                    "mode": "grid2level_coarse",
                },
            )

        zoom_area = coarse_action.get("area") or coarse_action.get("element")
        if zoom_area is None:
            print(f"  [step {self._step_count}] WARNING: targeting action but no area/element")
            self._history.append({
                "summary": "Could not determine zoom area",
                "action": coarse_action, "image_path": coarse_path})
            t_step_total = time.perf_counter() - t_step_start
            return base_agent.AgentInteractionResult(done=False, data={
                "step": self._step_count,
                "latency": self._build_latency_dict(
                    t_screenshot, t_preprocess, t_prompt,
                    t_inference_coarse, 0.0, t_step_total, coarse_usage)})
        cx0, cy0, cx1, cy1 = self._coarse_area_to_crop(zoom_area)
        print(f"  [step {self._step_count}] ZOOM into area {zoom_area} -> crop ({cx0},{cy0})-({cx1},{cy1})")

        t0 = time.perf_counter()
        crop = img.crop((cx0, cy0, cx1, cy1))
        target_w, target_h = FINE_IMG_TARGET_SIZE
        enlarged = crop.resize((target_w, target_h), Image.LANCZOS)

        fine_cell_w = target_w // self._fine_cols
        fine_cell_h = target_h // self._fine_rows
        fine_img = _draw_numbered_grid(
            enlarged.copy(), self._fine_rows, self._fine_cols,
            fine_cell_w, fine_cell_h)
        fine_path = os.path.join(self.output_dir, f"step_{self._step_count:03d}_fine.png")
        fine_img.save(fine_path)
        t_preprocess_fine = time.perf_counter() - t0

        t0 = time.perf_counter()
        fine_prompt = self._build_fine_prompt(goal, fine_cell_w, fine_cell_h)
        t_prompt_fine = time.perf_counter() - t0

        t0 = time.perf_counter()
        fine_raw, fine_usage = self.model.generate(fine_prompt, image_path=fine_path)
        t_inference_fine = time.perf_counter() - t0

        fine_annotated = _annotate_thinking(fine_img, fine_raw)
        fine_annotated.save(fine_path)

        print(
            f"\033[35m ==[step {self._step_count} FINE]=="
            f" ({self._fine_rows}x{self._fine_cols} in area {zoom_area})==\033[0m"
            f" inference={t_inference_fine:.2f}s"
        )
        print(fine_raw)

        fine_result = parse_grid_response(fine_raw)
        if fine_result is None:
            print(f"  [step {self._step_count}] WARNING: could not parse fine response")
            self._history.append({
                "summary": f"Zoomed into area {zoom_area} but failed to parse fine action",
                "action": coarse_action,
                "image_path": coarse_path,
            })
            combined_usage = self._combine_usage(coarse_usage, fine_usage)
            t_step_total = time.perf_counter() - t_step_start
            return base_agent.AgentInteractionResult(done=False, data={
                "step": self._step_count,
                "latency": self._build_latency_dict(
                    t_screenshot, t_preprocess + t_preprocess_fine,
                    t_prompt + t_prompt_fine,
                    t_inference_coarse + t_inference_fine,
                    0.0, t_step_total, combined_usage),
            })

        fine_action = fine_result["parsed_action"]

        t0 = time.perf_counter()
        try:
            screen_action = self._fine_to_screen_action(
                fine_action, zoom_area, fine_cell_w, fine_cell_h,
                target_w, target_h)
            self._execute_screen_action(screen_action)
        except Exception as e:
            print(f"[step {self._step_count}] WARNING executing fine action (continuing): {e}")
        t_action = time.perf_counter() - t0

        combined_usage = self._combine_usage(coarse_usage, fine_usage)
        t_step_total = time.perf_counter() - t_step_start

        summary = fine_result.get("summary", fine_raw[:100])
        self._history.append({
            "summary": f"[zoom {zoom_area}] {summary}",
            "action": fine_action,
            "image_path": coarse_path,
        })

        return base_agent.AgentInteractionResult(
            done=False,
            data={
                "step": self._step_count,
                "action": fine_action,
                "latency": self._build_latency_dict(
                    t_screenshot, t_preprocess + t_preprocess_fine,
                    t_prompt + t_prompt_fine,
                    t_inference_coarse + t_inference_fine,
                    t_action, t_step_total, combined_usage),
                "image_path": coarse_path,
                "mode": "grid2level_fine",
                "zoom_area": zoom_area,
            },
        )

    def _combine_usage(self, u1: dict, u2: dict) -> dict:
        return {
            "prompt_tokens": u1.get("prompt_tokens", 0) + u2.get("prompt_tokens", 0),
            "completion_tokens": u1.get("completion_tokens", 0) + u2.get("completion_tokens", 0),
            "total_tokens": u1.get("total_tokens", 0) + u2.get("total_tokens", 0),
            "ttft_s": u1.get("ttft_s", 0.0),
            "decode_s": u1.get("decode_s", 0.0) + u2.get("decode_s", 0.0),
            "tpot_s": u1.get("tpot_s", 0.0),
        }

    def _fine_to_screen_action(
        self, fine_action: dict, zoom_area: int,
        fine_cell_w: int, fine_cell_h: int,
        enlarged_w: int, enlarged_h: int,
    ) -> dict:
        cx0, cy0, cx1, cy1 = self._coarse_area_to_crop(zoom_area)
        crop_w = cx1 - cx0
        crop_h = cy1 - cy0

        def fine_to_screen(fx: int, fy: int) -> tuple[int, int]:
            sx = cx0 + int(fx * crop_w / enlarged_w)
            sy = cy0 + int(fy * crop_h / enlarged_h)
            return max(0, min(SCREEN_W - 1, sx)), max(0, min(SCREEN_H - 1, sy))

        name = fine_action["action"]

        if name == "tap_grid":
            fx, fy = _area_to_xy(
                fine_action["area"], fine_action.get("subarea", "center"),
                self._fine_cols, fine_cell_w, fine_cell_h)
            sx, sy = fine_to_screen(fx, fy)
            print(f"  [fine->screen] tap fine({fx},{fy}) -> screen({sx},{sy})")
            return {"action": "tap_screen", "x": sx, "y": sy}

        if name == "long_press_grid":
            fx, fy = _area_to_xy(
                fine_action["area"], fine_action.get("subarea", "center"),
                self._fine_cols, fine_cell_w, fine_cell_h)
            sx, sy = fine_to_screen(fx, fy)
            print(f"  [fine->screen] long_press fine({fx},{fy}) -> screen({sx},{sy})")
            return {"action": "long_press_screen", "x": sx, "y": sy}

        if name == "swipe_grid":
            sfx, sfy = _area_to_xy(
                fine_action["start_area"], fine_action["start_subarea"],
                self._fine_cols, fine_cell_w, fine_cell_h)
            efx, efy = _area_to_xy(
                fine_action["end_area"], fine_action["end_subarea"],
                self._fine_cols, fine_cell_w, fine_cell_h)
            sx, sy = fine_to_screen(sfx, sfy)
            ex, ey = fine_to_screen(efx, efy)
            dx, dy = ex - sx, ey - sy
            direction = "right" if abs(dx) >= abs(dy) and dx > 0 else \
                        "left"  if abs(dx) >= abs(dy) else \
                        "down"  if dy > 0 else "up"
            print(f"  [fine->screen] swipe ({sx},{sy})->({ex},{ey}) {direction}")
            return {"action": "swipe_screen", "x": sx, "y": sy, "direction": direction,
                    "x2": ex, "y2": ey}

        return fine_action

    def _execute_screen_action(self, action: dict):
        name = action["action"]
        if name == "tap_screen":
            aw_action = json_action.JSONAction(
                action_type=json_action.CLICK, x=action["x"], y=action["y"])
            self._env.execute_action(aw_action)
        elif name == "long_press_screen":
            aw_action = json_action.JSONAction(
                action_type=json_action.LONG_PRESS, x=action["x"], y=action["y"])
            self._env.execute_action(aw_action)
        elif name == "swipe_screen":
            x, y = action["x"], action["y"]
            x2, y2 = action["x2"], action["y2"]
            duration = 600
            print(f"[aw_adapter] swipe_screen ({x},{y})->({x2},{y2}) {duration}ms")
            self._adb_shell("input", "swipe", str(x), str(y), str(x2), str(y2), str(duration), timeout=10)
        else:
            self._execute_non_grid_action(action)

    def _execute_non_grid_action(self, parsed_action: dict):
        name = parsed_action["action"]
        if name == "open":
            aw_action = json_action.JSONAction(
                action_type=json_action.OPEN_APP, app_name=parsed_action["app"])
            self._env.execute_action(aw_action)
        elif name == "text":
            aw_action = json_action.JSONAction(
                action_type=json_action.INPUT_TEXT, text=parsed_action["text"])
            self._env.execute_action(aw_action)
        elif name == "answer":
            aw_action = json_action.JSONAction(
                action_type=getattr(json_action, "ANSWER", "answer"),
                text=parsed_action["text"])
            self._env.execute_action(aw_action)
        elif name == "clear_text":
            self._adb_shell("input", "keycombination", "113 29")
            self._adb_shell("input", "keyevent", "67")
        elif name == "enter":
            self._adb_shell("input", "keyevent", "KEYCODE_ENTER")
        elif name == "wait":
            time.sleep(parsed_action.get("time", 2))
        elif name == "scroll":
            direction = parsed_action["direction"]
            cx = SCREEN_W // 2
            cy = SCREEN_H // 2
            dist_px = int(SCREEN_H * 0.50)
            duration = 600
            dy = -dist_px if direction == "up" else dist_px
            y2 = max(0, min(SCREEN_H - 1, cy + dy))
            self._adb_shell("input", "swipe", str(cx), str(cy), str(cx), str(y2), str(duration), timeout=10)
        elif name == "back":
            aw_action = json_action.JSONAction(action_type=json_action.NAVIGATE_BACK)
            self._env.execute_action(aw_action)
        elif name == "home":
            aw_action = json_action.JSONAction(action_type=json_action.NAVIGATE_HOME)
            self._env.execute_action(aw_action)

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        if self.agent_mode == "grid2level":
            return self._step_grid2level(goal)

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
        if self.agent_mode == "raw":
            img.save(image_path)
            mode_img = img
            self._elem_list = _process_aw_ui_elements(state.ui_elements)
            mode_str = f"raw ({len(self._elem_list)} elements)"
        elif self._grid_on:
            grid_img = _draw_numbered_grid(img.copy(), self._grid_rows, self._grid_cols, self._cell_w, self._cell_h)
            grid_img.save(image_path)
            mode_img = grid_img
            self._elem_list = []
            mode_str = f"grid ({self._grid_rows}x{self._grid_cols})"
        else:
            self._elem_list = _process_aw_ui_elements(state.ui_elements)
            labeled_img = _draw_element_labels(img.copy(), self._elem_list)
            labeled_img.save(image_path)
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
            print(f"  [step {self._step_count}] WARNING: model returned empty/None response, retrying")
            return base_agent.AgentInteractionResult(done=False, data={"error": "empty_response"})

        # 5. parse structured response
        if self.agent_mode == "raw":
            result = parse_rawcoord_response(raw_response)
        elif self._grid_on:
            result = parse_grid_response(raw_response)
        else:
            result = parse_element_response(raw_response)

        if result is None:
            print(f"  [step {self._step_count}] WARNING: could not parse response, retrying")
            print("  Raw response was:", repr(raw_response))
            self._history.append({"summary": "Parse error, retrying", "action": {"action": "noop"}, "image_path": image_path})
            t_step_total = time.perf_counter() - t_step_start
            return base_agent.AgentInteractionResult(done=False, data={
                "step": self._step_count,
                "latency": self._build_latency_dict(t_screenshot, t_preprocess, t_prompt, t_inference, 0.0, t_step_total, token_usage),
            })

        parsed_action = result["parsed_action"]
        is_done = parsed_action["action"] == "done"

        # 6. handle grid toggle (only in element mode with grid fallback)
        if parsed_action["action"] == "grid" and not self._grid_only:
            print(f"  [step {self._step_count}] Switching to grid mode")
            self._history.append({
                "summary": result.get("summary", "Switched to grid mode"),
                "action": parsed_action,
                "image_path": image_path,
            })
            # don't execute, just re-loop
            t_step_total = time.perf_counter() - t_step_start
            return base_agent.AgentInteractionResult(
                done=False,
                data={
                    "step": self._step_count,
                    "action": parsed_action,
                    "latency": self._build_latency_dict(
                        t_screenshot, t_preprocess, t_prompt, t_inference, 0.0, t_step_total, token_usage
                    ),
                    "image_path": image_path,
                    "mode": "grid" if self._grid_on else "element",
                },
            )
        elif not self._grid_only:
            # after any non-grid action, switch back to element mode
            self._grid_on = False

        # 7. execute action
        t0 = time.perf_counter()
        if not is_done:
            try:
                if parsed_action["action"] == "tap_xy":
                    nx, ny = parsed_action["x"], parsed_action["y"]
                    x = max(0, min(SCREEN_W - 1, int(nx * SCREEN_W) if nx <= 1.0 else int(nx)))
                    y = max(0, min(SCREEN_H - 1, int(ny * SCREEN_H) if ny <= 1.0 else int(ny)))
                    print(f"[aw_adapter] tap_xy norm({nx},{ny}) -> px({x},{y})")
                    aw_action = json_action.JSONAction(action_type=json_action.CLICK, x=x, y=y)
                    self._env.execute_action(aw_action)
                elif parsed_action["action"] == "long_press_xy":
                    nx, ny = parsed_action["x"], parsed_action["y"]
                    x = max(0, min(SCREEN_W - 1, int(nx * SCREEN_W) if nx <= 1.0 else int(nx)))
                    y = max(0, min(SCREEN_H - 1, int(ny * SCREEN_H) if ny <= 1.0 else int(ny)))
                    print(f"[aw_adapter] long_press_xy norm({nx},{ny}) -> px({x},{y})")
                    aw_action = json_action.JSONAction(action_type=json_action.LONG_PRESS, x=x, y=y)
                    self._env.execute_action(aw_action)
                elif parsed_action["action"] == "swipe_xy":
                    nx, ny = parsed_action["x"], parsed_action["y"]
                    x = max(0, min(SCREEN_W - 1, int(nx * SCREEN_W) if nx <= 1.0 else int(nx)))
                    y = max(0, min(SCREEN_H - 1, int(ny * SCREEN_H) if ny <= 1.0 else int(ny)))
                    direction = parsed_action["direction"]
                    dist_name = parsed_action.get("dist", "medium")
                    dist_px = int(SCREEN_H * _SWIPE_DIST_FRAC.get(dist_name, 0.40))
                    duration = _SWIPE_DURATION.get(dist_name, 600)
                    dx = {"left": -dist_px, "right": dist_px, "up": 0, "down": 0}.get(direction, 0)
                    dy = {"left": 0, "right": 0, "up": -dist_px, "down": dist_px}.get(direction, 0)
                    x2 = max(0, min(SCREEN_W - 1, x + dx))
                    y2 = max(0, min(SCREEN_H - 1, y + dy))
                    print(f"[aw_adapter] swipe_xy norm({nx},{ny}) -> px({x},{y})\u2192({x2},{y2}) {direction} {dist_name}")
                    self._adb_shell("input", "swipe", str(x), str(y), str(x2), str(y2), str(duration), timeout=10)
                elif parsed_action["action"] == "swipe":
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
                    self._adb_shell("input", "swipe", str(x), str(y), str(x2), str(y2), str(duration), timeout=10)
                elif parsed_action["action"] == "clear_text":
                    print("[aw_adapter] clear_text: keycombination 113 29 + keyevent 67")
                    self._adb_shell("input", "keycombination", "113 29")
                    self._adb_shell("input", "keyevent", "67")
                elif parsed_action["action"] == "enter":
                    print("[aw_adapter] enter: KEYCODE_ENTER")
                    self._adb_shell("input", "keyevent", "KEYCODE_ENTER")
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
                    self._adb_shell("input", "swipe", str(cx), str(cy), str(cx), str(y2), str(duration), timeout=10)
                else:
                    aw_action = _action_to_aw(
                        parsed_action,
                        elem_list=self._elem_list,
                        grid_cols=self._grid_cols,
                        cell_w=self._cell_w,
                        cell_h=self._cell_h,
                    )
                    self._env.execute_action(aw_action)
            except Exception as e:
                print(f"[step {self._step_count}] WARNING executing (continuing): {e}")
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
                "latency": self._build_latency_dict(
                    t_screenshot, t_preprocess, t_prompt, t_inference, t_action, t_step_total, token_usage
                ),
                "image_path": image_path,
                "mode": "grid" if self._grid_on else "element",
                "n_elements": len(self._elem_list),
            },
        )
