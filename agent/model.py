import json, base64, os, time as _time
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
        history: list[dict] = None,
        examples: list[dict] = None,
    ) -> tuple[str, dict]:
        """
        Send a prompt to Gemini and return (text, usage) where
        usage = {prompt_tokens, completion_tokens, total_tokens, ttft_s, decode_s, tpot_s}
        Note: Gemini SDK does not expose per-request TTFT/TPOT, so those are 0.
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

        # ── History ──────────────────────────────────────────────────────
        if history:
            parts.append(types.Part.from_text(text="History of previous steps:"))
            for i, h in enumerate(history):
                step_header = f"Step {i + 1}:"
                parts.append(types.Part.from_text(text=step_header))

                img_p = h.get("image_path")
                if img_p and os.path.exists(img_p):
                    img_bytes = Path(img_p).read_bytes()
                    parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))

                summary = h.get("summary", "")
                if summary:
                    parts.append(types.Part.from_text(text=f"Action taken: {summary}"))

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

        usage = {}
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            meta = response.usage_metadata
            prompt_tokens = getattr(meta, "prompt_token_count", 0) or 0
            completion_tokens = getattr(meta, "candidates_token_count", 0) or 0
            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                # Gemini SDK doesn't expose per-request TTFT/TPOT
                "ttft_s": 0.0,
                "decode_s": 0.0,
                "tpot_s": 0.0,
            }

        return response.text, usage


class VLLMModel:
    def __init__(self, api_key: str, model_name: str, base_url: str = "http://127.0.0.1:8000/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def generate(self, prompt: str, image_path: str = None, history: list[dict] = None, examples: list[dict] = None) -> tuple[str, dict]:
        """
        Returns (text, usage) where usage includes:
          prompt_tokens, completion_tokens, total_tokens,
          ttft_s  (Time To First Token  = ViT encode + LLM prefill),
          decode_s (time from first token to last token),
          tpot_s  (decode_s / completion_tokens, i.e. per-output-token latency)
        Measured via streaming so TTFT is accurate.
        """
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

        # History
        if history:
            history_content = [{"type": "text", "text": "History of previous steps:"}]
            for i, h in enumerate(history):
                history_content.append({"type": "text", "text": f"Step {i + 1}:"})
                img_p = h.get("image_path")
                if img_p and os.path.exists(img_p):
                    img_b64 = base64.b64encode(Path(img_p).read_bytes()).decode()
                    history_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
                summary = h.get("summary", "")
                if summary:
                    history_content.append({"type": "text", "text": f"Action taken: {summary}"})
            messages.append({"role": "user", "content": history_content})

        # Current step
        content = [{"type": "text", "text": prompt}]
        if image_path:
            img_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}})
        messages.append({"role": "user", "content": content})

        # ── Streaming call to capture TTFT ────────────────────────────────
        t_request_start = _time.perf_counter()
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

        full_text = ""
        t_first_token: float | None = None
        usage_data = None

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if t_first_token is None:
                    t_first_token = _time.perf_counter()
                full_text += chunk.choices[0].delta.content
            # usage comes in the last chunk when stream_options include_usage=True
            if hasattr(chunk, "usage") and chunk.usage is not None:
                usage_data = chunk.usage

        t_end = _time.perf_counter()

        ttft = (t_first_token - t_request_start) if t_first_token else 0.0
        decode_s = (t_end - t_first_token) if t_first_token else 0.0

        prompt_tokens = getattr(usage_data, "prompt_tokens", 0) or 0 if usage_data else 0
        completion_tokens = getattr(usage_data, "completion_tokens", 0) or 0 if usage_data else 0
        tpot = (decode_s / completion_tokens) if completion_tokens > 0 else 0.0

        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "ttft_s": round(ttft, 4),
            "decode_s": round(decode_s, 4),
            "tpot_s": round(tpot, 4),
        }

        return full_text, usage
