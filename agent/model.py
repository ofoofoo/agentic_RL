import time
from pathlib import Path
import google.genai as genai
import google.genai.types as types
from google.genai.errors import ClientError


class GeminiModel:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def generate(self, prompt: str, image_path=None) -> str:
        """
        Send a prompt to Gemini and return the raw text response.
        """
        parts: list[types.Part] = [types.Part.from_text(text=prompt)]

        if image_path is not None:
            img_bytes = Path(image_path).read_bytes()
            parts.append(
                types.Part.from_bytes(data=img_bytes, mime_type="image/png")
            )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=parts,
        )
        print(response.text)
        return response.text
