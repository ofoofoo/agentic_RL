import json, base64, os, time as _time
from pathlib import Path
import google.genai as genai
import google.genai.types as types
from openai import OpenAI


def _image_part_gemini(image_path: str) -> types.Part:
    return types.Part.from_bytes(data=Path(image_path).read_bytes(), mime_type="image/png")

def _image_content(image_path: str) -> dict:
    img_b64 = base64.b64encode(Path(image_path).read_bytes()).decode()
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
    Two-pass inference: base model reasons, LoRA model acts.

    Pass 1 (base model — NO action formatting):
      - A clean "just reason" system prompt is used. The full `prompt` string from
        _build_prompt() is NOT passed to the base model because it embeds the raw-mode
        system prompt ("Your response MUST follow this exact format: Action: ..."),
        which would cause the base model to output an action instead of reasoning.
      - Only the task line is extracted and passed. The assistant turn is prefilled
        with "<think>\n"; generation stops at "</think>".

    Pass 2 (LoRA model — exact SFT format):
      - The original prompt + screenshot are passed (matching training).
      - The assistant turn is prefilled with the <think>...</think> block from Pass 1.
      - enable_thinking=True is used so Qwen3's template does NOT auto-insert a
        "<think>\n\n</think>\n" prefix (which would stop generation immediately).
      - The LoRA generates "Action: tap(x, y)" as it was trained to do.
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

        # ── Pass 1: base model generates a reasoning trace ───────────────────
        # Extract the task description from the assembled prompt string.
        # We cannot pass the full prompt because it embeds the raw-mode system
        # prompt text which causes the base model to emit an action, not reasoning.
        task_match = re.search(r"Task:\s*(.+?)(?:\n|$)", prompt)
        task_str   = task_match.group(1).strip() if task_match else prompt.strip()

        pass1_system = (
            "You are a reasoning assistant for Android phone control. "
            "You will be shown a screenshot and a task. "
            "Think step-by-step about the current screen state and what single "
            "action should be taken next to make progress on the task. "
            "Output ONLY your chain-of-thought reasoning. "
            "Do NOT output any function call or action."
        )

        # Only pass the task + screenshot — no action-formatting instructions.
        pass1_user_content: list[dict] = [{"type": "text", "text": f"Task: {task_str}"}]
        if image_path:
            pass1_user_content.append(_image_content(image_path))

        pass1_messages = [
            {"role": "system",    "content": pass1_system},
            {"role": "user",      "content": pass1_user_content},
            # Partial assistant prefill — model continues generating inside <think>
            {"role": "assistant", "content": "<think>\n"},
        ]

        t_req1 = _time.perf_counter()
        stream1 = self.base.client.chat.completions.create(
            model=self.base.model_name,
            messages=pass1_messages,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            stop=["</think>"],
            max_tokens=512,
            **({"temperature": temperature} if temperature is not None else {}),
        )

        think_inner = ""
        t_first1: float | None = None
        usage1_data = None
        for chunk in stream1:
            if chunk.choices and chunk.choices[0].delta.content:
                if t_first1 is None:
                    t_first1 = _time.perf_counter()
                think_inner += chunk.choices[0].delta.content
            if hasattr(chunk, "usage") and chunk.usage is not None:
                usage1_data = chunk.usage

        t_end1  = _time.perf_counter()
        ttft1   = (t_first1 - t_req1) if t_first1 else 0.0
        decode1 = (t_end1 - t_first1) if t_first1 else 0.0
        p1_prompt     = getattr(usage1_data, "prompt_tokens",     0) or 0 if usage1_data else 0
        p1_completion = getattr(usage1_data, "completion_tokens", 0) or 0 if usage1_data else 0

        think_inner = (think_inner or "").strip()
        think_text  = f"<think>\n{think_inner}\n</think>" if think_inner else ""

        # ── Pass 2: LoRA model generates the action ───────────────────────────
        # SFT training format per assistant turn:
        #   <think>\n{reasoning}\n</think>\nAction: tap(x, y)
        #
        # We replicate it exactly:
        #   - user message    : full original prompt + screenshot (same as training)
        #   - assistant prefill: <think>...</think>\n
        # The LoRA then continues with "Action: tap(x, y)".
        #
        # enable_thinking=True is REQUIRED here: with enable_thinking=False, Qwen3's
        # chat template auto-injects "<think>\n\n</think>\n" before every generation.
        # That injected "<think>" token would immediately terminate generation
        # (or corrupt the output). With enable_thinking=True and our prefill already
        # containing a complete think block, the model skips straight to Action:.
        think_prefix = (think_text.rstrip() + "\n") if think_text else ""

        pass2_user_content: list[dict] = [{"type": "text", "text": prompt}]
        if image_path:
            pass2_user_content.append(_image_content(image_path))

        pass2_messages = [
            {"role": "user",      "content": pass2_user_content},
            {"role": "assistant", "content": think_prefix},
        ]

        t_req2 = _time.perf_counter()
        stream2 = self.base.client.chat.completions.create(
            model=self.lora_model_name,
            messages=pass2_messages,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            stop=["\n"],        # one action line only
            max_tokens=64,
            **({"temperature": temperature} if temperature is not None else {}),
        )

        pass2_raw = ""
        t_first2: float | None = None
        usage2_data = None
        for chunk in stream2:
            if chunk.choices and chunk.choices[0].delta.content:
                if t_first2 is None:
                    t_first2 = _time.perf_counter()
                pass2_raw += chunk.choices[0].delta.content
            if hasattr(chunk, "usage") and chunk.usage is not None:
                usage2_data = chunk.usage

        t_end2  = _time.perf_counter()
        ttft2   = (t_first2 - t_req2) if t_first2 else 0.0
        decode2 = (t_end2 - t_first2) if t_first2 else 0.0
        p2_prompt     = getattr(usage2_data, "prompt_tokens",     0) or 0 if usage2_data else 0
        p2_completion = getattr(usage2_data, "completion_tokens", 0) or 0 if usage2_data else 0

        action_text = (pass2_raw or "").strip()
        # Strip any stray <think> block the LoRA may have opened.
        action_text = re.sub(r"(?is)<think>.*$", "", action_text).strip()
        # Ensure the parser sees an "Action:" prefix.
        if action_text and not re.match(r"(?i)^Action:\s*", action_text):
            action_text = "Action: " + action_text

        usage = {
            "prompt_tokens":     p1_prompt + p2_prompt,
            "completion_tokens": p1_completion + p2_completion,
            "total_tokens":      p1_prompt + p2_prompt + p1_completion + p2_completion,
            "ttft_s":   round(ttft1 + ttft2, 4),
            "decode_s": round(decode1 + decode2, 4),
            "tpot_s":   0.0,
            "dynamic_lora": True,
            "pass1_model": self.base.model_name,
            "pass2_model": self.lora_model_name,
            "pass1_raw":   f"<think>\n{think_inner}\n</think>",
            "pass1_think": think_text,
            "pass2_raw":   pass2_raw,
        }
        combined = (think_text + "\n" + action_text).strip() if think_text else action_text.strip()
        return combined, usage
