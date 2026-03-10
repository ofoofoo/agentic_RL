import json
import os


def build_element_prompt(screen_width: int, screen_height: int) -> str:
    """System prompt for UI-hierarchy (element) mode — primary mode."""
    return f"""\
You are an agent controlling an Android phone via a screen-reading loop.

At each step you receive:
  1. A screenshot of the current screen with interactive UI elements labeled by numbers
  2. A text list describing each labeled element (its type, text, and content description)
  3. The overall task you are trying to complete
  4. A brief history of the actions you have already taken

Your job is to decide the SINGLE best next action to make progress on the task.

Your response MUST follow this exact format (four sections, each on its own line):
  Observation: <Describe what you see on the current screen>
  Thought: <To complete the given task, what is the next step I should do>
  Action: <The function call with correct parameters, OR FINISH if done>
  Summary: <Summarize your past actions along with your latest action in one sentence>

The action must follow the exact format of the function calls, as this is crucial to parsing and execution.

Available actions (use exactly one per step):

  tap(element)
    Tap the UI element labeled with the given number.
    Example: tap(5)

  text(text_input)
    Type text into the currently focused input field. Use when a keyboard is visible.
    Example: text("Hello, world!")

  clear_text()
    Clear all text in the currently focused input field (select-all then delete).
    Use this before typing new text if the field already has content you want to replace.
    Example: clear_text()

  long_press(element)
    Long press the UI element labeled with the given number.
    Example: long_press(5)

  swipe(element, direction, dist)
    Swipe starting from a labeled element number (just the integer, not a variable name).
    direction: "up", "down", "left", or "right"
    dist: "short", "medium", or "long"
    Example: swipe(3, "up", "medium")   ← correct
    WRONG:   swipe(element_3, "up", "medium")  ← do NOT use variable names

  grid()
    Call this ONLY if the target element is NOT visible as a labeled number. This
    switches to grid overlay mode where you can target any screen area.

  back()
    Press the Android back button.

  home()
    Press the Android home button.

  FINISH
    Output this when the task has been successfully completed.

CRITICAL: The Action line must use ONLY the function names listed above, with plain integer
arguments (e.g. tap(6), swipe(3, "up", "medium")). Do NOT use variable names, natural language,
or any other format. The system will fail to execute any action that doesn't match exactly.

The screen dimensions are {screen_width}x{screen_height}.
"""


def build_grid_prompt(screen_width: int, screen_height: int, cell_w: int, cell_h: int) -> str:
    """System prompt for grid-overlay mode — fallback when elements aren't labeled."""
    return f"""\
You are an agent controlling an Android phone. The screen is overlaid with a numbered grid.
Each grid area is labeled with an integer in the top-left corner.

Your response MUST follow this exact format:
  Observation: <Describe what you see on the current screen>
  Thought: <To complete the given task, what is the next step I should do>
  Action: <The function call with correct parameters, OR FINISH if done>
  Summary: <Summarize your past actions along with your latest action in one sentence>

Available actions:

  tap(area, subarea)
    Tap a grid area. "subarea" is one of: center, top-left, top, top-right,
    left, right, bottom-left, bottom, bottom-right.
    Example: tap(5, "center")

  long_press(area, subarea)
    Long press a grid area. Same subarea options as tap.
    Example: long_press(7, "top-left")

  swipe(start_area, start_subarea, end_area, end_subarea)
    Swipe from one grid area to another.
    Example: swipe(21, "center", 25, "right")

  text(text_input)
    Type text into the currently focused input field.
    Example: text("Hello")

  clear_text()
    Clear all text in the currently focused input field (select-all then delete).
    Example: clear_text()

  back()
    Press the Android back button.

  home()
    Press the Android home button.

  FINISH
    Output this when the task has been successfully completed.

The screen dimensions are {screen_width}x{screen_height}. Each grid cell is {cell_w}x{cell_h}.
"""


def build_element_text_list(elem_list) -> str:
    """
    Build a text description of labeled elements to include in the prompt,
    giving the model both visual AND textual grounding.
    """
    if not elem_list:
        return "No interactive elements detected on screen."
    lines = ["Interactive elements on screen:"]
    for idx, elem in enumerate(elem_list, 1):
        parts = [f"  {idx}."]
        # element type from attrib
        parts.append(f"[{elem.attrib}]")
        if elem.text:
            parts.append(f'text="{elem.text}"')
        if elem.content_desc:
            parts.append(f'desc="{elem.content_desc}"')
        lines.append(" ".join(parts))
    return "\n".join(lines)


def load_examples(examples_dir: str) -> list[dict]:
    """
    Load ICL examples from *examples_dir*.

    Each example is a pair of files with the same numeric prefix:
      NNN_screenshot.png  — grid-annotated screenshot (what the model sees)
      NNN_meta.json       — {"task": str, "reasoning": str, "action": dict}

    Returns a list of dicts:
      {"task": str, "screenshot": str, "reasoning": str, "action": dict}
    sorted by prefix.
    """
    if not os.path.isdir(examples_dir):
        return []

    examples = {}
    for fname in os.listdir(examples_dir):
        prefix, _, rest = fname.partition("_")
        if not prefix.isdigit():
            continue
        idx = int(prefix)
        examples.setdefault(idx, {})
        full_path = os.path.join(examples_dir, fname)

        if rest == "screenshot.png":
            examples[idx]["screenshot"] = full_path
        elif rest == "meta.json":
            with open(full_path) as f:
                meta = json.load(f)
            examples[idx].update(meta)

    # Only return complete examples (have both files)
    complete = [
        v for v in examples.values()
        if "screenshot" in v and "task" in v and "action" in v
    ]
    complete.sort(key=lambda e: e["screenshot"])
    return complete
