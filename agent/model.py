import json
from pathlib import Path
import google.genai as genai
import google.genai.types as types


class GeminiModel:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        image_path: str = None,
        examples: list[dict] = None,
    ) -> str:
        """
        Send a prompt to Gemini and return the raw text response.
        """
        parts: list[types.Part] = []

        # load ICL examples
        if examples:
            for i, ex in enumerate(examples, 1):
                header = (
                    f"=== EXAMPLE {i} ===\n"
                    f"Task: {ex['task']}\n"
                    f"Screenshot (with coordinate grid):"
                )
                parts.append(types.Part.from_text(text=header))

                ex_img = Path(ex["screenshot"]).read_bytes()
                parts.append(types.Part.from_bytes(data=ex_img, mime_type="image/png"))

                reasoning = ex.get("reasoning", "")
                action_json = json.dumps(ex["action"])
                footer = (
                    f"{reasoning}\n{action_json}"
                    if reasoning
                    else action_json
                )
                parts.append(types.Part.from_text(text=footer))

            parts.append(types.Part.from_text(text="=== YOUR TURN ==="))

        # ── Current step ─────────────────────────────────────────────────
        parts.append(types.Part.from_text(text=prompt))

        if image_path is not None:
            img_bytes = Path(image_path).read_bytes()
            parts.append(
                types.Part.from_bytes(data=img_bytes, mime_type="image/png")
            )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=parts,
        )
        return response.text
