import json
import os

def build_system_prompt(screen_width: int, screen_height: int) -> str:
    return f"""\
You are an agent controlling an Android phone via a screen-reading loop.

At each step you receive:
  1. A screenshot of the current screen (with a coordinate grid overlay for reference)
  2. The overall task you are trying to complete
  3. A brief history of the actions you have already taken

Your job is to decide the SINGLE best next action to make progress on the task.

First, reason step-by-step:
  - Describe what you see on the current screen
  - Use the red grid lines and their labels to estimate element positions
  - Identify which UI element is relevant and estimate its center pixel
  - Choose the best action

Then, on the VERY LAST LINE of your response, output a single valid JSON object
(no markdown fences, no trailing text) using one of these action types:

  {{"action": "tap",   "args": {{"x": <int>, "y": <int>}}}}
  {{"action": "swipe", "args": {{"x1": <int>, "y1": <int>, "x2": <int>, "y2": <int>, "duration_ms": <int>}}}}
  {{"action": "type",  "args": {{"text": "<string>"}}}}
  {{"action": "back",  "args": {{}}}}
  {{"action": "home",  "args": {{}}}}
  {{"action": "done",  "args": {{}}}}

Use "done" when the task has been successfully completed.

You may also receive in-context examples of previous runs to give you better grounding on where certain apps on the homescreen may be. If you are provided with these examples, please use them to help orient yourself with the coordinates.

IMPORTANT: The screenshot's coordinate space is exactly {screen_width}x{screen_height} pixels.
Coordinates use screen pixels with origin at top-left (x=0, y=0).
x ranges from 0 to {screen_width - 1}, y ranges from 0 to {screen_height - 1}.
"""


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
