import re

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


def parse_raw_response(rsp: str) -> dict | None:
    """Same as parse_element_response but for raw normalized coordinates."""
    try:
        observation = re.findall(r"Observation:\s*(.*?)$", rsp, re.MULTILINE)[0]
        thought = re.findall(r"Thought:\s*(.*?)$", rsp, re.MULTILINE)[0]
        act_str = re.findall(r"Action:\s*(.*?)$", rsp, re.MULTILINE)[0]
        summary = re.findall(r"Summary:\s*(.*?)$", rsp, re.MULTILINE)[0]
    except IndexError:
        return None

    parsed = _parse_raw_action_string(act_str)
    if parsed is None:
        return None

    return {
        "observation": observation,
        "thought": thought,
        "action_raw": act_str,
        "summary": summary,
        "parsed_action": parsed,
    }


def _parse_raw_action_string(act_str: str) -> dict | None:
    """Parser for raw normalized coordinate function calls."""
    act_str = act_str.strip()
    if "task_complete" in act_str.lower() or "task_impossible" in act_str.lower() or "finish" in act_str.lower():
        return {"action": "done"}

    if "(" not in act_str:
        return None

    act_name = act_str.split("(")[0].strip().lower()

    try:
        inner = re.findall(r'\((.*)\)', act_str)[0]
        parts = [p.strip().strip('"').strip("'") for p in inner.split(",")]

        if act_name in ("tap", "click"):
            return {"action": "tap_raw", "x": float(parts[0]), "y": float(parts[1])}

        elif act_name == "swipe":
            return {
                "action": "swipe_raw",
                "x1": float(parts[0]),
                "y1": float(parts[1]),
                "x2": float(parts[2]),
                "y2": float(parts[3]),
            }

        elif act_name in ("text", "type", "input"):
            text_val = inner.strip().strip('"').strip("'")
            return {"action": "text", "text": text_val}

        elif act_name == "press_back":
            return {"action": "back"}

        elif act_name == "press_home":
            return {"action": "home"}

        elif act_name == "press_enter":
            return {"action": "enter"}

        else:
            return None

    except (IndexError, ValueError):
        return None


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
    OPEN_ALIASES  = {"open", "launch", "open_app", "launch_app"}

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

        elif act_name in OPEN_ALIASES:
            inner = re.findall(r'\((.*)\)', act_str)[0]
            app_name = inner.strip().strip('"').strip("'")
            return {"action": "open", "app": app_name}

        elif act_name == "scroll":
            inner = re.findall(r'\((.*)\)', act_str)[0]
            direction = inner.strip().strip('"').strip("'").lower()
            return {"action": "scroll", "direction": direction}

        elif act_name == "answer":
            inner = re.findall(r'\((.*)\)', act_str)[0]
            text_val = inner.strip().strip('"').strip("'")
            return {"action": "answer", "text": text_val}

        elif act_name == "wait":
            inner = re.findall(r'\((.*)\)', act_str)[0]
            sec = int(inner.strip()) if inner.strip() else 2
            return {"action": "wait", "time": sec}

        elif act_name == "enter":
            return {"action": "enter"}

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
