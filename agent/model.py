import json, base64, os, time as _time
from pathlib import Path
import google.genai as genai
import google.genai.types as types
from openai import OpenAI


def _image_part_gemini(image_path: str) -> types.Part:
    return types.Part.from_bytes(data=Path(image_path).read_bytes(), mime_type="image/png")

def _image_content(image_path: str) -> dict:
    img_b64 = base64.b64encode(Path(image_path).read_bytes()).decode() # encode image
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}

def _format_action(ex: dict) -> str:
    reasoning = ex.get("reasoning", "")
    action_json = json.dumps(ex["action"])
    return f"{reasoning}\n{action_json}" if reasoning else action_json


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
        temperature: float | None = None,
        enable_thinking: bool = False,
    ) -> tuple[str, dict]:
        """
        Send a prompt to Gemini and return (text, usage) where
        usage = {prompt_tokens, completion_tokens, total_tokens, ttft_s, decode_s, tpot_s}
        Note: Gemini SDK does not expose per-request TTFT/TPOT, so those are 0.
        """
        parts: list[types.Part] = []

        # ICL examples
        if examples:
            for i, ex in enumerate(examples, 1):
                header = (
                    f"=== EXAMPLE {i} ===\n"
                    f"Task: {ex['task']}\n"
                    f"Screenshot (with coordinate grid):"
                )
                parts.append(types.Part.from_text(text=header))
                parts.append(_image_part_gemini(ex["screenshot"]))
                parts.append(types.Part.from_text(text=_format_action(ex)))

            parts.append(types.Part.from_text(text="=== YOUR TURN ==="))

        # History
        if history:
            parts.append(types.Part.from_text(text="History of previous steps:"))
            for i, h in enumerate(history):
                parts.append(types.Part.from_text(text=f"Step {i + 1}:"))
                img_p = h.get("image_path")
                if img_p and os.path.exists(img_p):
                    parts.append(_image_part_gemini(img_p))

                summary = h.get("summary", "")
                if summary:
                    parts.append(types.Part.from_text(text=f"Action taken: {summary}"))

        # Current step
        parts.append(types.Part.from_text(text=prompt))
        if image_path is not None:
            parts.append(_image_part_gemini(image_path))

        gen_config = {}
        if temperature is not None:
            gen_config["temperature"] = temperature
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=parts,
            config=types.GenerateContentConfig(**gen_config) if gen_config else None,
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
                "ttft_s": 0.0,
                "decode_s": 0.0,
                "tpot_s": 0.0,
            }

        return response.text, usage


class VLLMModel:
    def __init__(self, api_key: str, model_name: str, base_url: str = "http://127.0.0.1:8000/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def generate(
        self,
        prompt: str,
        image_path: str = None,
        history: list[dict] = None,
        examples: list[dict] = None,
        temperature: float | None = None,
        enable_thinking: bool = False,
        stop: list[str] | None = None,
        model_override: str | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, dict]:
        """
        Returns (text, usage) where usage includes:
          prompt_tokens, completion_tokens, total_tokens,
          ttft_s  (Time To First Token  = ViT encode + LLM prefill),
          decode_s (time from first token to last token),
          tpot_s  (decode_s / completion_tokens, i.e. per-output-token latency)
        """
        messages = []

        # ICL examples
        if examples:
            for i, ex in enumerate(examples, 1):
                messages.append({"role": "user", "content": [
                    {"type": "text", "text": f"=== EXAMPLE {i} ===\nTask: {ex['task']}"},
                    _image_content(ex["screenshot"]),
                ]})
                messages.append({"role": "assistant", "content": _format_action(ex)})
            messages.append({"role": "user", "content": "=== YOUR TURN ==="})

        # History
        if history:
            history_content = [{"type": "text", "text": "History of previous steps:"}]
            for i, h in enumerate(history):
                history_content.append({"type": "text", "text": f"Step {i + 1}:"})
                img_p = h.get("image_path")
                if img_p and os.path.exists(img_p):
                    history_content.append(_image_content(img_p))
                summary = h.get("summary", "")
                if summary:
                    history_content.append({"type": "text", "text": f"Action taken: {summary}"})
            messages.append({"role": "user", "content": history_content})

        # Current step
        content = [{"type": "text", "text": prompt}]
        if image_path:
            content.append(_image_content(image_path))
        messages.append({"role": "user", "content": content})

        t_request_start = _time.perf_counter()
        kwargs = dict(
            model=(model_override or self.model_name),
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
        if stop:
            kwargs["stop"] = stop
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        stream = self.client.chat.completions.create(**kwargs)

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


class DynamicLoRAVLLMModel:
    """
    Two-pass inference helper for "base-think then LoRA-act" using a single vLLM OpenAI server.

    Assumptions:
    - Server is started with --enable-lora and --lora-modules <lora_name>=<lora_path>
    - You can select the adapter by setting the request "model" to <lora_name>
      while base requests use the original base model id.
    """

    def __init__(self, base: VLLMModel, lora_model_name: str):
        self.base = base
        self.lora_model_name = lora_model_name

    def generate(
        self,
        prompt: str,
        image_path: str = None,
        history: list[dict] = None,
        examples: list[dict] = None,
        temperature: float | None = None,
        enable_thinking: bool = False,
    ) -> tuple[str, dict]:
        import re

        def _rewrite_for_pass1_think(p: str) -> str:
            """
            The raw-mode system prompt often forces 'Action:' only.
            For pass 1 we rewrite that instruction to allow a think trace.
            """
            if "Your response MUST follow this exact format:" not in p:
                return p
            # Replace the strict Action-only contract with a think-only contract.
            p = re.sub(
                r"Your response MUST follow this exact format:\s*\n\s*Action:.*?\n\n",
                "Your response MUST follow this exact format:\n  <think>...</think>\n\n",
                p,
                flags=re.DOTALL,
            )
            return p

        # Pass 1: base-only thinking.
        # Make this robust by *forcing* the generation to begin inside a <think> block:
        # we end the prompt with "<think>\n" and stop at "</think>".
        prompt1 = _rewrite_for_pass1_think(prompt) + "\n\n<think>\n"

        # Stop right before </think> is emitted (stop token is not included).
        pass1_raw, usage1 = self.base.generate(
            prompt=prompt1,
            image_path=image_path,
            history=history,
            examples=examples,
            temperature=temperature,
            enable_thinking=True,
            stop=["</think>"],
            max_tokens=256,
        )

        # Pass 1 output is the INSIDE of the think block (because prompt already included "<think>\n").
        think_inner = (pass1_raw or "").strip()
        # If the base still tried to emit "Action:" inside think, drop it.
        think_inner = re.sub(r"(?im)^\s*Action:\s*.*$", "", think_inner).strip()
        think_text = f"<think>\n{think_inner}\n</think>" if think_inner else ""

        # Match the original SFT format where action follows immediately after the thinking trace.
        think_prefix = (think_text.rstrip() + "\n") if think_text else ""

        # Pass 2: LoRA generates a direct continuation (action) after </think>.
        # This is closer to the training distribution than adding new meta-instructions.
        # Pass 2: LoRA action-only.
        # We explicitly forbid <think> here and stop at the first newline so it outputs ONE line.
        prompt2 = (
            f"{prompt}\n\n"
            f"{think_prefix}"
            "Respond with exactly one line in the format:\n"
            "Action: <single action call>\n"
            "Do not output <think>.\n"
            "Action: "
        )
        pass2_raw, usage2 = self.base.generate(
            prompt=prompt2,
            image_path=image_path,
            history=history,
            examples=examples,
            temperature=temperature,
            enable_thinking=False,
            model_override=self.lora_model_name,
            stop=["\n", "<think>", "Observation:", "Thought:"],
            max_tokens=32,
        )
        action_text = (pass2_raw or "").strip()
        # If LoRA starts a think block anyway, strip it.
        action_text = re.sub(r"(?is)<think>.*$", "", action_text).strip()
        # Ensure we return an Action line (parser expects it in raw mode).
        if action_text and not re.match(r"(?i)^Action:\s*", action_text):
            action_text = "Action: " + action_text

        usage = {
            "prompt_tokens": usage1.get("prompt_tokens", 0) + usage2.get("prompt_tokens", 0),
            "completion_tokens": usage1.get("completion_tokens", 0) + usage2.get("completion_tokens", 0),
            "total_tokens": usage1.get("total_tokens", 0) + usage2.get("total_tokens", 0),
            "ttft_s": (usage1.get("ttft_s", 0.0) + usage2.get("ttft_s", 0.0)),
            "decode_s": (usage1.get("decode_s", 0.0) + usage2.get("decode_s", 0.0)),
            "tpot_s": 0.0,
            "dynamic_lora": True,
            "pass1_model": self.base.model_name,
            "pass2_model": self.lora_model_name,
            "pass1_raw": pass1_raw,
            "pass1_think": think_text,
            "pass2_raw": pass2_raw,
        }
        # Keep think trace visible (for debugging), then the action output.
        combined = (think_text + "\n" + action_text).strip() if think_text else action_text.strip()
        return combined, usage
