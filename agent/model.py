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
        assistant_prefill: str | None = None,
        continue_final_message: bool = False,
    ) -> tuple[str, dict]:
        """
        Returns (text, usage) where usage includes:
          prompt_tokens, completion_tokens, total_tokens,
          ttft_s  (Time To First Token  = ViT encode + LLM prefill),
          decode_s (time from first token to last token),
          tpot_s  (decode_s / completion_tokens, i.e. per-output-token latency)

        If ``assistant_prefill`` is provided, a final assistant message with that
        content is appended and ``continue_final_message`` is auto-enabled (vLLM
        will continue from the prefill instead of opening a new assistant turn).
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

        if assistant_prefill is not None:
            messages.append({"role": "assistant", "content": assistant_prefill})
            continue_final_message = True

        extra_body = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
        if continue_final_message:
            extra_body["continue_final_message"] = True
            extra_body["add_generation_prompt"] = False

        t_request_start = _time.perf_counter()
        kwargs = dict(
            model=(model_override or self.model_name),
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=extra_body,
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

    @staticmethod
    def _strip_action_contract(agent_prompt: str) -> str:
        """Remove the agent's action-format contract from the user prompt for pass 1.

        The agent prompt (raw / element / grid) contains a "Your response MUST follow
        this exact format: ... Action: ..." section that pulls the base model toward
        emitting an Action line. Pass 1 only needs the situation + task; pass 2 still
        gets the full prompt unchanged.
        """
        cutoff = agent_prompt.find("Your response MUST follow this exact format")
        return agent_prompt[:cutoff].rstrip() if cutoff >= 0 else agent_prompt.rstrip()

    @staticmethod
    def _sanitize_think_inner(inner: str) -> str:
        """Drop trailing action leakage from pass-1 reasoning."""
        import re

        s = (inner or "").strip()
        if not s:
            return ""

        truncators = [
            re.compile(r"(?im)^\s*Action\s*:"),
            re.compile(r"(?im)^\s*(tap|swipe|type|press_|open|scroll|wait|task_|answer|long_press)\s*\("),
        ]
        cut = len(s)
        for pat in truncators:
            m = pat.search(s)
            if m:
                cut = min(cut, m.start())
        return s[:cut].rstrip()

    def generate(
        self,
        prompt: str,
        image_path: str = None,
        history: list[dict] = None,
        examples: list[dict] = None,
        temperature: float | None = None,
        enable_thinking: bool = False,
    ) -> tuple[str, dict]:
        # ── Pass 1 ────────────────────────────────────────────────────────────────
        # Base model writes the reasoning trace. We:
        #   - drop the action-format contract from the prompt (so it doesn't write Action: …)
        #   - drop ICL examples (they teach reasoning + action JSON, not <think>)
        #   - prefill the assistant turn with "<think>\n" via continue_final_message,
        #     so generation literally begins inside the think block
        #   - stop at "</think>"
        prompt1 = (
            self._strip_action_contract(prompt)
            + "\n\n---\n"
            + "You are writing an internal reasoning trace for a downstream model that will choose the UI action.\n"
            + "Look at the screenshot and the task. In first person, describe what you see and the next logical step is to complete the task. Nothing extra over what is needed.\n"
            + "Plain English only. End your reasoning naturally."
        )
        pass1_raw, usage1 = self.base.generate(
            prompt=prompt1,
            image_path=image_path,
            history=history,
            examples=None,
            temperature=temperature,
            enable_thinking=False,
            stop=["</think>"],
            max_tokens=512,
            assistant_prefill="<think>\n",
        )
        think_inner = self._sanitize_think_inner(pass1_raw)
        think_text = f"<think>\n{think_inner}\n</think>" if think_inner else ""

        # ── Pass 2 ────────────────────────────────────────────────────────────────
        # LoRA continues the assistant turn. Training labels look like:
        #   <|im_start|>assistant\n<think>\n…reasoning…\n</think>\nAction: …<|im_end|>
        # so we prefill exactly that structure (closed think block + newline) and let
        # the LoRA decode the single action line.
        if think_text:
            prefill2 = think_text + "\n"
        else:
            # Fallback: pass-1 produced nothing usable. Give the LoRA an empty think
            # block so it stays on-distribution and emits the action.
            prefill2 = "<think>\n\n</think>\n"

        pass2_raw, usage2 = self.base.generate(
            prompt=prompt,
            image_path=image_path,
            history=history,
            examples=examples,
            temperature=temperature,
            enable_thinking=enable_thinking,
            model_override=self.lora_model_name,
            stop=["\n", "<|im_end|>"],
            max_tokens=128,
            assistant_prefill=prefill2,
        )
        action_text = (pass2_raw or "").strip()

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
        combined = (
            (think_text + "\n" + action_text).strip()
            if think_text and action_text
            else (think_text or action_text)
        )
        return combined, usage
