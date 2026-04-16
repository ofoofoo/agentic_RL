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
  4. A history of screenshots and actions from previous steps

Your job is to decide the SINGLE best next action to make progress on the task.

Your response MUST follow this exact format (four sections, each on its own line):
  Observation: <Describe what you see on the current screen>
  Thought: <To complete the given task, what is the next step I should do>
  Action: <The function call with correct parameters, OR FINISH if done>
  Summary: <Summarize your past actions along with your latest action in one sentence>

The action must follow the exact format of the function calls, as this is crucial to parsing and execution. 

The Action line must use ONLY the function names listed above, with plain integer arguments
(e.g. tap(6), swipe(3, "up", "medium")). Do NOT use variable names or natural language.

Available actions (use exactly one per step):

  open(app_name)
    ALWAYS use this to launch an app. Use this instead of swiping to access the
    app drawer or searching. Works even if the app icon is not on screen.
    Example: open("Clock")
    Example: open("Audio Recorder")
    Example: open("Settings")

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
  
  scroll(direction)
    Scroll the screen in a direction. Use this for scrolling lists/pages — it is
    more reliable than swipe. Direction: "up" (see more below), "down" (see more above).
    Example: scroll("up")    ← scrolls the page to reveal content further down
    Example: scroll("down")  ← scrolls up to reveal content above

  swipe(element, direction, dist)
    Swipe starting from a labeled element number (just the integer, not a variable name).
    direction: "up", "down", "left", or "right"
    dist: "short", "medium", or "long"
    Example: swipe(3, "up", "medium")   ← correct
    WRONG:   swipe(element_3, "up", "medium")  ← do NOT use variable names
    WRONG:   swipe(505, 712, 505, 290)

  grid()
    Call this ONLY if the target element is NOT visible as a labeled number. This
    switches to grid overlay mode where you can target any screen area.

  answer(text_input)
    Output the answer for information-retrieval tasks.
    Example: answer("The current time is 10:30 AM")

  wait(seconds)
    Wait for a specified number of seconds for the screen to update.
    Example: wait(5)

  enter()
    Press the Android Enter key. Useful for submitting forms or search queries.
    Example: enter()

  back()
    Press the Android back button.

  home()
    Press the Android home button.

  FINISH
    Output this when the task has been successfully completed.

CRITICAL RULES:
- If the app you need is not visible on screen, use open("App Name"). Do not do this for the downloads or file manager.
- The Action line must use ONLY the function names listed above, with plain integer arguments
  (e.g. tap(6), swipe(3, "up", "medium")). Do NOT use variable names or natural language.

The screen dimensions are {screen_width}x{screen_height}.
"""


def build_grid_prompt(screen_width: int, screen_height: int, cell_w: int, cell_h: int) -> str:
    """System prompt for grid-overlay mode — fallback when elements aren't labeled."""
    return f"""\
You are an agent controlling an Android phone via a screen-reading loop. The current screen is overlaid with a numbered grid.

At each step you receive:
  1. A screenshot of the current screen with a numbered grid overlay
  2. The overall task you are trying to complete
  3. A history of screenshots and actions from previous steps

Each grid area is labeled with an integer in the top-left corner.

Your response MUST follow this exact format:
  Observation: <Describe what you see on the current screen>
  Thought: <To complete the given task, what is the next step I should do>
  Action: <The function call with correct parameters, OR FINISH if done>
  Summary: <Summarize your past actions along with your latest action in one sentence>

Available actions:

  open(app_name)
    Use this to launch an app. Works even if the app icon is not on screen.
    Example: open("Clock")
    Example: open("Settings")

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

  scroll(direction)
    Scroll the screen in a direction. Direction: "up" or "down".
    Example: scroll("up")

  text(text_input)
    Type text into the currently focused input field.
    Example: text("Hello")

  clear_text()
    Clear all text in the currently focused input field (select-all then delete).
    Example: clear_text()

  answer(text_input)
    Output the answer for information-retrieval tasks.
    Example: answer("The current time is 10:30 AM")

  wait(seconds)
    Wait for a specified number of seconds for the screen to update.
    Example: wait(5)

  enter()
    Press the Android Enter key. Useful for submitting forms or search queries.
    Example: enter()

  back()
    Press the Android back button.

  home()
    Press the Android home button.

  FINISH
    Output this when the task has been successfully completed.

CRITICAL RULES:
- If the app you need is not visible on screen, use open("App Name"). Do not do this for the downloads or file manager.
- The Action line must use ONLY the function names listed above, with plain integer arguments.

The screen dimensions are {screen_width}x{screen_height}. Each grid cell is {cell_w}x{cell_h}.
"""


def build_raw_prompt(screen_width: int, screen_height: int) -> str:
    """System prompt for raw normalized coordinate mode."""
    return f"""\
You are an agent controlling an Android phone. You interact with the screen using normalized coordinates where (0.0, 0.0) is the top-left and (1.0, 1.0) is the bottom-right.

Your response MUST follow this exact format:
  <The function call with correct parameters, OR task_complete() if done>

Available actions:

  tap(x, y)
    Tap a point on the screen. x is horizontal (0.0=left, 1.0=right),
    y is vertical (0.0=top, 1.0=bottom).
    Example: tap(0.512, 0.743)

  swipe(x1, y1, x2, y2)
    Swipe from (x1, y1) to (x2, y2).
    Example: swipe(0.5, 0.8, 0.5, 0.2)

  type(text_input)
    Type text into the currently focused input field.
    Example: type("Hello")

  press_back()
    Press the Android back button.

  press_home()
    Press the Android home button.

  press_enter()
    Press the Android Enter key. Useful for submitting forms or search queries.
  
  task_complete()
    Output this when the task has been successfully completed.

The screen dimensions are {screen_width}x{screen_height} pixels."""


def build_coarse_grid_prompt(
    screen_width: int, screen_height: int,
    cell_w: int, cell_h: int,
    grid_rows: int, grid_cols: int,
) -> str:
    """System prompt for the COARSE pass of 2-level grid mode."""
    max_area = grid_rows * grid_cols
    return f"""\
You are an agent controlling an Android phone via a screen-reading loop. The current screen is overlaid with a numbered grid of {grid_rows} rows x {grid_cols} columns = {max_area} areas, numbered 1 to {max_area}.

At each step you receive:
  1. A screenshot of the current screen with a numbered grid overlay (areas 1-{max_area})
  2. The overall task you are trying to complete
  3. A history of screenshots and actions from previous steps

Each grid area is labeled with an integer (1-{max_area}) in the top-left corner.

Your response MUST follow this exact format (four sections, each on its own line):
  Observation: <Describe what you see on the current screen>
  Thought: <To complete the given task, what is the next step I should do>
  Action: <The function call with correct parameters, OR FINISH if done>
  Summary: <Summarize your past actions along with your latest action in one sentence>

The Action line must contain ONLY a single function call. No extra text, no explanation.

Available actions:

  tap(area)
    Tap a UI element that is inside the given grid area number (1-{max_area}).
    The system will zoom in for precise targeting.
    Example: tap(12)

  long_press(area)
    Long press a UI element inside the given grid area number (1-{max_area}).
    Example: long_press(7)

  open(app_name)
    Use this to launch an app. Works even if the app icon is not on screen.
    Example: open("Clock")
    Example: open("Settings")

  text(text_input)
    Type text into the currently focused input field.
    Example: text("Hello")

  clear_text()
    Clear all text in the currently focused input field (select-all then delete).
    Example: clear_text()

  scroll(direction)
    Scroll the screen in a direction. Direction: "up" or "down".
    Example: scroll("up")

  answer(text_input)
    Output the answer for information-retrieval tasks.
    Example: answer("The current time is 10:30 AM")

  wait(seconds)
    Wait for a specified number of seconds for the screen to update.
    Example: wait(5)

  enter()
    Press the Android Enter key.
    Example: enter()

  back()
    Press the Android back button.

  home()
    Press the Android home button.

  FINISH
    Output this when the task has been successfully completed.

CRITICAL RULES:
- Grid area numbers range from 1 to {max_area} ONLY. Do NOT use numbers outside this range.
- The Action line must contain ONLY a function call (e.g. tap(12), open("Clock")). No brackets, no extra words.
- If the app you need is not visible on screen, use open("App Name"). Do not do this for the downloads or file manager.

The screen dimensions are {screen_width}x{screen_height}. Each grid cell is {cell_w}x{cell_h}.
"""


def build_fine_grid_prompt(
    screen_width: int, screen_height: int,
    cell_w: int, cell_h: int,
    grid_rows: int, grid_cols: int,
) -> str:
    """System prompt for the FINE pass of 2-level grid mode (zoomed-in view)."""
    max_area = grid_rows * grid_cols
    return f"""\
You are an agent controlling an Android phone. You are now looking at a ZOOMED-IN view of a specific area on the screen, overlaid with a fine numbered grid ({grid_rows} rows x {grid_cols} columns, areas 1-{max_area}) for precise targeting.

Your response MUST follow this exact format:
  Observation: <Describe what you see in this zoomed-in view>
  Thought: <Which exact element should I interact with>
  Action: <The function call with correct parameters>
  Summary: <Summarize your action in one sentence>

The Action line must contain ONLY a single function call. No extra text.

Available actions:

  tap(area, subarea)
    Tap a grid area (1-{max_area}). "subarea" is one of: center, top-left, top, top-right,
    left, right, bottom-left, bottom, bottom-right.
    Example: tap(5, "center")

  long_press(area, subarea)
    Long press a grid area (1-{max_area}). Same subarea options as tap.
    Example: long_press(7, "top-left")

  swipe(start_area, start_subarea, end_area, end_subarea)
    Swipe from one grid area to another.
    Example: swipe(21, "center", 25, "right")

CRITICAL: Grid areas range from 1 to {max_area} ONLY. Each grid cell is {cell_w}x{cell_h} pixels.
"""


def build_rawcoord_prompt(screen_width: int, screen_height: int) -> str:
    """System prompt for raw-coordinate mode — model outputs tap(x, y) normalized coords.
    Uses Observation/Thought/Action/Summary format with a rich action set."""
    return f"""\
You are an agent controlling an Android phone via a screen-reading loop.

At each step you receive:
  1. A screenshot of the current screen (no annotations)
  2. The overall task you are trying to complete
  3. A history of screenshots and actions from previous steps

Your job is to decide the SINGLE best next action to make progress on the task.

Your response MUST follow this exact format (four sections, each on its own line):
  Observation: <Describe what you see on the current screen>
  Thought: <To complete the given task, what is the next step I should do>
  Action: <The function call with correct parameters, OR FINISH if done>
  Summary: <Summarize your past actions along with your latest action in one sentence>

The action must follow the exact format of the function calls, as this is crucial to parsing and execution.

Available actions (use exactly one per step):

  open(app_name)
    ALWAYS use this to launch an app. Use this instead of swiping to access the
    app drawer or searching. Works even if the app icon is not on screen.
    Example: open("Clock")
    Example: open("Audio Recorder")
    Example: open("Settings")

  tap(x, y)
    Tap the screen at normalized coordinates (x, y) where both x and y are
    decimal values between 0.0 and 1.0.
    x is the horizontal position: 0.0 = left edge, 1.0 = right edge.
    y is the vertical position: 0.0 = top edge, 1.0 = bottom edge.
    Example: tap(0.50, 0.50)  <- center of screen
    Example: tap(0.25, 0.75)  <- left quarter, three-quarters down

  long_press(x, y)
    Long press the screen at normalized coordinates (x, y). Same coordinate
    system as tap: 0.0 to 1.0 for both axes.
    Example: long_press(0.50, 0.50)

  text(text_input)
    Type text into the currently focused input field. Use when a keyboard is visible.
    Example: text("Hello, world!")

  clear_text()
    Clear all text in the currently focused input field (select-all then delete).
    Use this before typing new text if the field already has content you want to replace.
    Example: clear_text()

  scroll(direction)
    Scroll the screen in a direction. Use this for scrolling lists/pages — it is
    more reliable than swipe. Direction: "up" (see more below), "down" (see more above).
    Example: scroll("up")    <- scrolls the page to reveal content further down
    Example: scroll("down")  <- scrolls up to reveal content above

  swipe(x, y, direction, dist)
    Swipe starting from normalized coordinates (x, y).
    direction: "up", "down", "left", or "right"
    dist: "short", "medium", or "long"
    Example: swipe(0.50, 0.50, "up", "medium")

  answer(text_input)
    Output the answer for information-retrieval tasks.
    Example: answer("The current time is 10:30 AM")

  wait(seconds)
    Wait for a specified number of seconds for the screen to update.
    Example: wait(5)

  enter()
    Press the Android Enter key. Useful for submitting forms or search queries.
    Example: enter()

  back()
    Press the Android back button.

  home()
    Press the Android home button.

  FINISH
    Output this when the task has been successfully completed.

CRITICAL RULES:
- If the app you need is not visible on screen, use open("App Name"). Do not do this for the downloads or file manager.
- For tap, long_press, and swipe, output NORMALIZED coordinates (x, y) as decimal values
  between 0.0 and 1.0. x=0.0 is the left edge, x=1.0 is the right edge,
  y=0.0 is the top edge, y=1.0 is the bottom edge.
- The Action line must use ONLY the function names listed above.

The screen dimensions are {screen_width}x{screen_height}.
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
