import json, base64, os, re, time as _time
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
        thinking_budget: int | None = None,
        max_tokens: int | None = None,
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
        if max_tokens is not None:
            gen_config["max_output_tokens"] = max_tokens
        if thinking_budget is not None:
            gen_config["thinking_config"] = types.ThinkingConfig(
                thinking_budget=thinking_budget,
            )
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


def _build_vllm_messages(
    prompt: str,
    image_path: str | None,
    history: list[dict] | None,
    examples: list[dict] | None,
) -> list[dict]:
    """Construct the OpenAI-style multimodal messages list shared by all vLLM models."""
    messages: list[dict] = []

    if examples:
        for i, ex in enumerate(examples, 1):
            messages.append({"role": "user", "content": [
                {"type": "text", "text": f"=== EXAMPLE {i} ===\nTask: {ex['task']}"},
                _image_content(ex["screenshot"]),
            ]})
            messages.append({"role": "assistant", "content": _format_action(ex)})
        messages.append({"role": "user", "content": "=== YOUR TURN ==="})

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

    content = [{"type": "text", "text": prompt}]
    if image_path:
        content.append(_image_content(image_path))
    messages.append({"role": "user", "content": content})

    return messages


class VLLMModel:
    def __init__(self, api_key: str, model_name: str, base_url: str = "http://127.0.0.1:8000/v1"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def generate(self, prompt: str, image_path: str = None, history: list[dict] = None, examples: list[dict] = None, temperature: float | None = None, enable_thinking: bool = False, thinking_budget: int | None = None, max_tokens: int | None = None) -> tuple[str, dict]:
        """
        Returns (text, usage) where usage includes:
          prompt_tokens, completion_tokens, total_tokens,
          ttft_s  (Time To First Token  = ViT encode + LLM prefill),
          decode_s (time from first token to last token),
          tpot_s  (decode_s / completion_tokens, i.e. per-output-token latency)
        """
        messages = _build_vllm_messages(prompt, image_path, history, examples)

        t_request_start = _time.perf_counter()
        extra_body = {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
        if thinking_budget is not None:
            extra_body["thinking_token_budget"] = thinking_budget
        kwargs = dict(
            model=self.model_name,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            extra_body=extra_body,
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
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
    Two-pass "dynamic LoRA" inference against a single vLLM server that exposes
    both the base model and a LoRA adapter (started with --enable-lora --lora-modules).

    Pass 1 (reasoning):
        - target = `base_model` (no LoRA), `enable_thinking=True`
        - stops at `</think>` so the base model only emits the reasoning body
        - this is the original Qwen3-VL's "pure" thinking trace
    Pass 2 (action):
        - target = `lora_model` (LoRA adapter), `enable_thinking=False`
        - assistant prefill = the exact `<think>...</think>\n` block from pass 1,
          carried via vLLM's `continue_final_message=True`
        - LoRA continues directly into the trained action format

    Returned `text` is the concatenation `<think>\n{trace}\n</think>\n{action}`,
    so downstream parsers (which strip leading think blocks) see a normal action.

    Both passes hit the same vLLM server and share the same prompt prefix, so vLLM's
    prefix cache amortises the (expensive) ViT encode + prompt prefill across them.
    """

    DEFAULT_THINK_MAX_TOKENS = 512
    DEFAULT_ACTION_MAX_TOKENS = 256

    def __init__(
        self,
        api_key: str,
        base_model: str,
        lora_model: str,
        base_url: str = "http://127.0.0.1:8000/v1",
        think_max_tokens: int | None = None,
        action_max_tokens: int | None = None,
        lora_as_tool: bool = False,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.base_model = base_model
        self.lora_model = lora_model
        self.think_max_tokens = think_max_tokens or self.DEFAULT_THINK_MAX_TOKENS
        self.action_max_tokens = action_max_tokens or self.DEFAULT_ACTION_MAX_TOKENS
        self.lora_as_tool = lora_as_tool

    @staticmethod
    def _strip_think_wrapper(text: str) -> str:
        """If pass-1 emitted its own <think>...</think> wrapper, drop it; we re-wrap ourselves."""
        text = text.strip()
        text = re.sub(r"^<think>\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*</think>\s*$", "", text, flags=re.IGNORECASE)
        return text.strip()

    def _stream(self, kwargs: dict) -> tuple[str, dict]:
        """Run a streaming chat completion and collect (text, usage_dict)."""
        t_request_start = _time.perf_counter()
        stream = self.client.chat.completions.create(**kwargs)

        full_text = ""
        t_first_token: float | None = None
        usage_data = None

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if t_first_token is None:
                    t_first_token = _time.perf_counter()
                full_text += chunk.choices[0].delta.content
            if hasattr(chunk, "usage") and chunk.usage is not None:
                usage_data = chunk.usage

        t_end = _time.perf_counter()
        ttft = (t_first_token - t_request_start) if t_first_token else 0.0
        decode_s = (t_end - t_first_token) if t_first_token else 0.0

        prompt_tokens = getattr(usage_data, "prompt_tokens", 0) or 0 if usage_data else 0
        completion_tokens = getattr(usage_data, "completion_tokens", 0) or 0 if usage_data else 0
        tpot = (decode_s / completion_tokens) if completion_tokens > 0 else 0.0

        return full_text, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "ttft_s": ttft,
            "decode_s": decode_s,
            "tpot_s": tpot,
        }

    def generate(
        self,
        prompt: str,
        image_path: str | None = None,
        history: list[dict] | None = None,
        examples: list[dict] | None = None,
        temperature: float | None = None,
        enable_thinking: bool = False,  # accepted for API parity; ignored (this model is always 2-pass)
        pass1_prompt: str | None = None,
    ) -> tuple[str, dict]:
        """
        pass1_prompt: a SHORT, format-free prompt for Pass 1 — e.g. just
            "Task: <goal>\\n\\nWhat should happen next on this screen?"
            This is critical: if Pass 1 receives the full formatted agent prompt
            (with "Your response MUST follow this exact format: <action call>")
            the base model puts the action inside <think> instead of reasoning,
            which completely breaks Pass 2.

            When None, falls back to the full `prompt` (suboptimal but won't crash).
        """
        del enable_thinking  # always two-pass: pass 1 thinks, pass 2 acts

        # Pass 2 gets the full formatted agent prompt (matches LoRA training distribution)
        pass2_base_messages = _build_vllm_messages(prompt, image_path, history, examples)

        # Pass 1 gets a minimal prompt: just the task + screenshot, no format rules
        if pass1_prompt is not None:
            p1_content: list[dict] = [{"type": "text", "text": pass1_prompt}]
            if image_path:
                p1_content.append(_image_content(image_path))
            pass1_messages = [{"role": "user", "content": p1_content}]
        else:
            # Fallback: use the same messages as Pass 2 (suboptimal)
            pass1_messages = pass2_base_messages

        # ── Pass 1: base model generates the reasoning trace ─────────
        # The qwen3-vl chat template inserts `<think>\n` before the assistant
        # generation starts (enable_thinking=True).  We stop at </think> so we
        # only capture the reasoning body — the model never emits an action here.
        pass1_kwargs = dict(
            model=self.base_model,
            messages=pass1_messages,
            stream=True,
            stream_options={"include_usage": True},
            max_tokens=self.think_max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        )
        if not self.lora_as_tool:
            pass1_kwargs["stop"] = ["</think>"]
        if temperature is not None:
            pass1_kwargs["temperature"] = temperature
        thinking_body, p1_usage = self._stream(pass1_kwargs)
        thinking_body_stripped = self._strip_think_wrapper(thinking_body)

        print(
            f"\033[36m[PASS1/BASE  <think>]\033[0m\n"
            f"\033[36m{thinking_body}\033[0m"
        )

        clean_thinking_body = thinking_body_stripped
        if self.lora_as_tool:
            # Parse Is_Coordinate_Action from Pass 1
            is_coordinate = True # Default to True
            match = re.search(r"Is_Coordinate_Action:\s*(True|False)", thinking_body, re.IGNORECASE)
            if match:
                is_coordinate = match.group(1).lower() == "true"
            else:
                # Fallback heuristic
                if "task_complete" in thinking_body or "press_" in thinking_body:
                    is_coordinate = False

            reasoning_match = re.search(r"Reasoning:\s*(.*?)(?=\nAction:|$)", thinking_body_stripped, re.DOTALL | re.IGNORECASE)
            clean_thinking_body = reasoning_match.group(1).strip() if reasoning_match else thinking_body_stripped

            if not is_coordinate:
                print("\033[33m[PASS1/BASE] is_coordinate=False, bypassing Pass 2.\033[0m")
                # Construct a response that matches the expected format (action outside <think> block)
                action_match = re.search(r"Action:\s*(.*?)(?=\nIs_Coordinate_Action:|$)", thinking_body_stripped, re.DOTALL | re.IGNORECASE)
                action_text = f"Action: {action_match.group(1).strip()}" if action_match else thinking_body_stripped
                
                simulated_response = f"<think>\n{clean_thinking_body}\n</think>\n{action_text}"
                return simulated_response, p1_usage

        # ── Pass 2: LoRA generates the action with the trace prefilled ─
        # Extract just the reasoning part for the LoRA prefill so we don't confuse it
        # with the Action/Is_Coordinate_Action format which it wasn't trained on.

        # We pass the completed <think>...</think>\n block as the start of the
        # assistant turn.  `continue_final_message=True` tells vLLM to NOT close
        # the turn with <|im_end|> — the model decodes directly after </think>\n.
        #
        # IMPORTANT: use enable_thinking=False so the template does NOT prepend
        # another <think>\n before our prefill content (that would double the tag
        # and send the LoRA into a reasoning loop).
        prefill = f"<think>\n{clean_thinking_body}\n</think>\n"
        pass2_messages = pass2_base_messages + [{"role": "assistant", "content": prefill}]

        pass2_kwargs = dict(
            model=self.lora_model,
            messages=pass2_messages,
            stream=True,
            stream_options={"include_usage": True},
            max_tokens=self.action_max_tokens,
            extra_body={
                "continue_final_message": True,
                "add_generation_prompt": False,
                # The <think>...</think> block is already in our prefill content;
                # telling the template enable_thinking=False prevents it from
                # prepending a second <think> tag.
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        if temperature is not None:
            pass2_kwargs["temperature"] = temperature
        action_text, p2_usage = self._stream(pass2_kwargs)

        print(
            f"\033[32m[PASS2/LORA  action]\033[0m\n"
            f"\033[32m{action_text}\033[0m"
        )

        full_text = f"{prefill}{action_text}"
        usage = self._merge_usage(p1_usage, p2_usage)
        return full_text, usage

    @staticmethod
    def _merge_usage(p1: dict, p2: dict) -> dict:
        prompt_tokens = p1["prompt_tokens"] + p2["prompt_tokens"]
        completion_tokens = p1["completion_tokens"] + p2["completion_tokens"]
        total_tokens = prompt_tokens + completion_tokens
        # TTFT is dominated by pass-1 prefill (first user-visible latency before any token).
        # decode time is the sum of both passes' decode windows.
        ttft_s = p1["ttft_s"]
        decode_s = p1["decode_s"] + p2["decode_s"]
        tpot_s = (decode_s / completion_tokens) if completion_tokens > 0 else 0.0
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "ttft_s": round(ttft_s, 4),
            "decode_s": round(decode_s, 4),
            "tpot_s": round(tpot_s, 4),
            # extra split-out fields for visibility into the 2-pass breakdown
            "pass1_prompt_tokens": p1["prompt_tokens"],
            "pass1_completion_tokens": p1["completion_tokens"],
            "pass1_ttft_s": round(p1["ttft_s"], 4),
            "pass1_decode_s": round(p1["decode_s"], 4),
            "pass2_prompt_tokens": p2["prompt_tokens"],
            "pass2_completion_tokens": p2["completion_tokens"],
            "pass2_ttft_s": round(p2["ttft_s"], 4),
            "pass2_decode_s": round(p2["decode_s"], 4),
        }
