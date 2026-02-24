import json, base64
from pathlib import Path
import google.genai as genai
import google.genai.types as types
from openai import OpenAI


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

class VLLMModel:
    def __init__(self, api_key: str, model_name: str, base_url: str = "http://127.0.0.1:8000/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def generate(self, prompt: str, image_path: str = None, examples: list[dict] = None) -> str:
        messages = []

        # ICL examples
        if examples:
            for i, ex in enumerate(examples, 1):
                img_b64 = base64.b64encode(Path(ex["screenshot"]).read_bytes()).decode()
                messages.append({"role": "user", "content": [
                    {"type": "text", "text": f"=== EXAMPLE {i} ===\nTask: {ex['task']}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ]})
                reasoning = ex.get("reasoning", "")
                action_str = (f"{reasoning}\n" if reasoning else "") + json.dumps(ex["action"])
                messages.append({"role": "assistant", "content": action_str})
            messages.append({"role": "user", "content": "=== YOUR TURN ==="})

        # Current step
        content = [{"type": "text", "text": prompt}]
        if image_path:
            img_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
        messages.append({"role": "user", "content": content})

        response = self.client.chat.completions.create(model=self.model_name, messages=messages)
        return response.choices[0].message.content
