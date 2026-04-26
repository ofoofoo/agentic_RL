#!/usr/bin/env python3
"""
KL divergence check: Teacher-Forcing vs Natural-Inference reasoning traces.

For each SFT example (x, y_target):
  1. NI generation: modified prompt asks for reasoning → model reasons + acts freely
  2. TF generation: prompt includes y_target → model reasons about why it's correct
  3. Cross-evaluate: teacher-force R_TF through NI-prompted model via prompt_logprobs
  4. Compute per-position approximate KL(P_TF || P_NI) and average

If the TF reasoning is "in distribution" (similar to what the model would produce
naturally), the average KL should be low.

Usage:
  python kl_check.py \\
      --data data/aitw_general/aitw_general.json \\
      --image-dir data/aitw_general \\
      --model Qwen/Qwen3-VL-8B-Instruct \\
      --num-samples 100
"""

import argparse
import base64
import json
import math
import os
import random
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import io

import numpy as np
from PIL import Image
from openai import OpenAI
from tqdm import tqdm

MAX_RETRIES = 3
RETRY_BACKOFF = 5  # seconds, doubles each retry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REASONING_INSTRUCTION_NI = (
    "Your response MUST follow this exact format:\n"
    "  Reasoning: <describe what you see on the screen and reason step-by-step about what to do>\n"
    "  Action: <The function call with correct parameters, OR task_complete() if done>"
)

ORIGINAL_INSTRUCTION = (
    "Your response MUST follow this exact format:\n"
    "  Action: <The function call with correct parameters, OR task_complete() if done>"
)

# Tokens that form the generation prompt appended by vLLM after the last
# assistant message: \n <|im_start|> assistant \n
N_GEN_PROMPT_TOKENS = 4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_image_b64(path: str, max_pixels: int | None = None) -> str:
    """Load image as base64.  Optionally resize so total pixels ≤ max_pixels
    (reduces vision-token count and GPU memory for prompt_logprobs)."""
    if max_pixels is None:
        return base64.b64encode(Path(path).read_bytes()).decode()
    img = Image.open(path)
    w, h = img.size
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def _img_block(b64: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def _text_and_image_blocks(text: str, img_b64: str) -> list[dict]:
    """Split text on <image> placeholder and interleave with image blocks."""
    parts = text.split("<image>")
    blocks = []
    for i, seg in enumerate(parts):
        if seg:
            blocks.append({"type": "text", "text": seg})
        if i < len(parts) - 1:
            blocks.append(_img_block(img_b64))
    return blocks


def build_ni_messages(entry: dict, image_dir: str, max_pixels: int | None = None) -> list[dict]:
    """Natural-Inference prompt: asks the model to reason then act."""
    user_text = entry["messages"][0]["content"]
    img_b64 = load_image_b64(os.path.join(image_dir, entry["images"][0]), max_pixels)
    ni_text = user_text.replace(ORIGINAL_INSTRUCTION, REASONING_INSTRUCTION_NI)
    return [{"role": "user", "content": _text_and_image_blocks(ni_text, img_b64)}]


def build_tf_messages(entry: dict, image_dir: str, max_pixels: int | None = None) -> list[dict]:
    """Teacher-Forcing prompt: reveals the correct action, asks for reasoning."""
    user_text = entry["messages"][0]["content"]
    target_action = entry["messages"][1]["content"]
    img_b64 = load_image_b64(os.path.join(image_dir, entry["images"][0]), max_pixels)
    tf_instruction = (
        f"The correct next action is: {target_action}\n"
        "Your response MUST follow this exact format:\n"
        "  Reasoning: <explain why the above action is correct for the current screen>\n"
        "  Action: <repeat the correct action above>"
    )
    tf_text = user_text.replace(ORIGINAL_INSTRUCTION, tf_instruction)
    return [{"role": "user", "content": _text_and_image_blocks(tf_text, img_b64)}]


def decode_prompt_logprob_entry(entry: dict) -> tuple[str, float, dict]:
    """Parse one prompt_logprobs dict → (actual_token, actual_logprob, top_k_dict).

    vLLM format: {token_id_str: {logprob, rank, decoded_token}, ...}
    The first key is the actual prompt token at that position.
    """
    actual_id = next(iter(entry))
    info = entry[actual_id]
    actual_tok = info["decoded_token"]
    actual_lp = info["logprob"]
    top_k = {v["decoded_token"]: v["logprob"] for v in entry.values()}
    return actual_tok, actual_lp, top_k


# ---------------------------------------------------------------------------
# Generation + cross-evaluation
# ---------------------------------------------------------------------------

def _retry(fn, *args, **kwargs):
    """Call `fn` with retries on transient errors (connection, timeout)."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            is_transient = any(
                kw in str(e).lower()
                for kw in ("connection", "timeout", "unavailable", "502", "503", "reset")
            )
            if not is_transient or attempt == MAX_RETRIES - 1:
                raise
            wait = RETRY_BACKOFF * (2 ** attempt)
            tqdm.write(f"  [retry {attempt+1}/{MAX_RETRIES}] {e!r:.80s} — waiting {wait}s")
            time.sleep(wait)


def generate_with_logprobs(
    client: OpenAI,
    model: str,
    messages: list[dict],
    max_tokens: int = 512,
    temperature: float = 0.6,
    top_logprobs: int = 20,
) -> dict:
    """Generate a response and return token-level logprobs."""
    resp = _retry(
        client.chat.completions.create,
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        logprobs=True,
        top_logprobs=top_logprobs,
    )
    choice = resp.choices[0]
    gen_text = choice.message.content or ""
    token_data = []
    if choice.logprobs and choice.logprobs.content:
        for tl in choice.logprobs.content:
            top_k = {t.token: t.logprob for t in (tl.top_logprobs or [])}
            top_k[tl.token] = tl.logprob
            token_data.append({"token": tl.token, "logprob": tl.logprob, "top_logprobs": top_k})
    return {"text": gen_text, "tokens": token_data}


def cross_evaluate(
    client: OpenAI,
    model: str,
    user_messages: list[dict],
    assistant_text: str,
    n_gen_tokens: int,
    top_k: int = 20,
) -> list[dict] | None:
    """Teacher-force `assistant_text` through the model with `user_messages`
    and return per-token logprobs for the assistant response.

    Uses vLLM's prompt_logprobs feature via the chat completions API.
    """
    eval_messages = user_messages + [{"role": "assistant", "content": assistant_text}]
    resp = _retry(
        client.chat.completions.create,
        model=model,
        messages=eval_messages,
        max_tokens=1,
        temperature=0.0,
        extra_body={"prompt_logprobs": top_k},
    )

    plps = getattr(resp, "prompt_logprobs", None)
    if not plps:
        return None

    n_total = len(plps)

    # The prompt structure (Qwen ChatML):
    #   [user tokens]<|im_end|>\n<|im_start|>assistant\n[R tokens]<|im_end|>\n<|im_start|>assistant\n
    # The last N_GEN_PROMPT_TOKENS entries are the generation prompt: \n <|im_start|> assistant \n
    # Right before that is the <|im_end|> of the assistant message.
    # The R tokens (+ <|im_end|>) are at positions:
    #   start = n_total - N_GEN_PROMPT_TOKENS - n_gen_tokens
    #   end   = n_total - N_GEN_PROMPT_TOKENS
    start = n_total - N_GEN_PROMPT_TOKENS - n_gen_tokens
    end = n_total - N_GEN_PROMPT_TOKENS

    if start < 0:
        start = 0

    result = []
    for i in range(start, end):
        entry = plps[i]
        if entry is None:
            result.append({"token": "?", "logprob": float("nan"), "top_logprobs": {}})
            continue
        tok, lp, top_dict = decode_prompt_logprob_entry(entry)
        result.append({"token": tok, "logprob": lp, "top_logprobs": top_dict})

    return result


def verify_alignment(gen_tokens: list[dict], cross_tokens: list[dict], n_check: int = 5) -> float:
    """Check what fraction of the first n_check tokens align between
    the generation and the cross-evaluation."""
    matches = 0
    n = min(n_check, len(gen_tokens), len(cross_tokens))
    for i in range(n):
        if gen_tokens[i]["token"].strip() == cross_tokens[i]["token"].strip():
            matches += 1
    return matches / n if n > 0 else 0.0


# ---------------------------------------------------------------------------
# KL computation
# ---------------------------------------------------------------------------

def approx_kl_at_position(p_tf_topk: dict, p_ni_topk: dict) -> float:
    """Approximate KL(P_TF || P_NI) using top-K logprob distributions.

    For tokens in P_TF's support but not in P_NI's top-K, we use a
    conservative floor (min observed NI logprob - 3).
    """
    if not p_tf_topk or not p_ni_topk:
        return float("nan")

    ni_floor = min(p_ni_topk.values()) - 3.0

    kl = 0.0
    for token, tf_lp in p_tf_topk.items():
        p_tf = math.exp(tf_lp)
        if p_tf <= 1e-12:
            continue
        ni_lp = p_ni_topk.get(token, ni_floor)
        kl += p_tf * (tf_lp - ni_lp)

    return max(kl, 0.0)


def compute_kl_for_example(tf_tokens: list[dict], cross_tokens: list[dict]) -> dict:
    """Compute per-position KL and aggregate statistics.

    Skips the final <|im_end|> token if present.
    """
    n = min(len(tf_tokens), len(cross_tokens))
    if n == 0:
        return {"n_positions": 0}

    # Optionally skip <|im_end|> at the end
    if n > 1 and tf_tokens[n - 1]["token"].strip() in ("<|im_end|>", "</s>", "<|endoftext|>"):
        n -= 1

    position_kls = []
    position_logratios = []
    per_token_details = []

    for t in range(n):
        p_tf_topk = tf_tokens[t]["top_logprobs"]
        p_ni_topk = cross_tokens[t]["top_logprobs"]

        kl_t = approx_kl_at_position(p_tf_topk, p_ni_topk)
        lr_t = tf_tokens[t]["logprob"] - cross_tokens[t]["logprob"]

        position_kls.append(kl_t)
        position_logratios.append(lr_t)
        per_token_details.append({
            "pos": t,
            "token": tf_tokens[t]["token"],
            "lp_tf": round(tf_tokens[t]["logprob"], 4),
            "lp_ni": round(cross_tokens[t]["logprob"], 4),
            "kl": round(kl_t, 6),
        })

    valid_kls = [k for k in position_kls if not math.isnan(k)]
    valid_lrs = [k for k in position_logratios if not math.isnan(k)]

    return {
        "n_positions": n,
        "mean_kl": float(np.mean(valid_kls)) if valid_kls else None,
        "median_kl": float(np.median(valid_kls)) if valid_kls else None,
        "max_kl": float(np.max(valid_kls)) if valid_kls else None,
        "std_kl": float(np.std(valid_kls)) if valid_kls else None,
        "mean_logratio": float(np.mean(valid_lrs)) if valid_lrs else None,
        "tf_mean_lp": float(np.mean([t["logprob"] for t in tf_tokens[:n]])),
        "ni_cross_mean_lp": float(np.mean([t["logprob"] for t in cross_tokens[:n]])),
        "per_token": per_token_details,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_kl_check(args):
    print(f"Loading data from {args.data} ...")
    with open(args.data) as f:
        data = json.load(f)

    if args.num_samples and args.num_samples < len(data):
        random.seed(args.seed)
        data = random.sample(data, args.num_samples)
    print(f"  Using {len(data)} examples (seed={args.seed})")

    sample_img = os.path.join(args.image_dir, data[0]["images"][0])
    if not os.path.exists(sample_img):
        print(f"ERROR: Image not found: {sample_img}")
        sys.exit(1)

    client = OpenAI(api_key=args.api_key, base_url=args.vllm_url)
    print(f"  vLLM: {args.vllm_url}  model: {args.model}")
    print(f"  temperature: {args.temperature}  max_tokens: {args.max_tokens}  top_logprobs: {args.top_logprobs}")
    print()

    results = []
    alignment_rates = []

    for idx, entry in enumerate(tqdm(data, desc="KL check")):
        target_action = entry["messages"][1]["content"]
        rec = {"image": entry["images"][0], "target_action": target_action}

        try:
            # --- Step 1: NI generation ---
            ni_msgs = build_ni_messages(entry, args.image_dir)
            ni_gen = generate_with_logprobs(
                client, args.model, ni_msgs,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_logprobs=args.top_logprobs,
            )

            # --- Step 2: TF generation ---
            tf_msgs = build_tf_messages(entry, args.image_dir)
            tf_gen = generate_with_logprobs(
                client, args.model, tf_msgs,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_logprobs=args.top_logprobs,
            )

            n_tf_tokens = len(tf_gen["tokens"])
            if n_tf_tokens == 0:
                rec["error"] = "TF generation produced 0 tokens"
                results.append(rec)
                continue

            # --- Step 3: Cross-evaluate R_TF under NI prompt ---
            # Use resized image to reduce vision tokens and avoid OOM
            # during prompt_logprobs computation
            cross_max_px = args.cross_eval_max_pixels or None
            ni_msgs_small = build_ni_messages(
                entry, args.image_dir, max_pixels=cross_max_px,
            )
            cross_ni = cross_evaluate(
                client, args.model, ni_msgs_small,
                tf_gen["text"], n_tf_tokens,
                top_k=args.top_logprobs,
            )

            if cross_ni is None or len(cross_ni) == 0:
                rec["error"] = "cross-evaluation returned no logprobs"
                results.append(rec)
                continue

            # Verify alignment
            align_rate = verify_alignment(tf_gen["tokens"], cross_ni)
            alignment_rates.append(align_rate)
            if align_rate < 0.6:
                # Try to fix offset: shift cross_ni by ±1,2 and pick best alignment
                best_offset, best_rate = 0, align_rate
                for off in [-2, -1, 1, 2]:
                    shifted = cross_ni[off:] if off > 0 else cross_ni[:off] if off < 0 else cross_ni
                    if len(shifted) == 0:
                        continue
                    rate = verify_alignment(tf_gen["tokens"], shifted)
                    if rate > best_rate:
                        best_rate = rate
                        best_offset = off
                if best_offset != 0 and best_rate > align_rate:
                    cross_ni = cross_ni[best_offset:] if best_offset > 0 else cross_ni[:best_offset]
                    align_rate = best_rate

            # --- Step 4: Compute KL ---
            kl_result = compute_kl_for_example(tf_gen["tokens"], cross_ni)

            rec.update({
                "ni_text": ni_gen["text"][:300],
                "tf_text": tf_gen["text"][:300],
                "n_tf_tokens": n_tf_tokens,
                "n_cross_tokens": len(cross_ni),
                "alignment_rate": round(align_rate, 2),
                **{k: (round(v, 6) if isinstance(v, float) else v)
                   for k, v in kl_result.items() if k != "per_token"},
            })

            # Only save per-token details if requested
            if args.save_per_token:
                rec["per_token"] = kl_result.get("per_token", [])

        except Exception as e:
            rec["error"] = str(e)
            rec["traceback"] = traceback.format_exc()

        results.append(rec)

        # Periodic progress + checkpoint
        if (idx + 1) % 10 == 0:
            valid_so_far = [r for r in results if "mean_kl" in r and r["mean_kl"] is not None]
            if valid_so_far:
                running_kl = np.mean([r["mean_kl"] for r in valid_so_far])
                running_align = np.mean(alignment_rates) if alignment_rates else 0
                tqdm.write(
                    f"  [{idx+1}/{len(data)}] running mean KL={running_kl:.4f}  "
                    f"align={running_align:.2f}  errors={len(results)-len(valid_so_far)}"
                )
            # Checkpoint save every 50 examples
            if (idx + 1) % 50 == 0:
                _checkpoint = {"config": {}, "results": results}
                with open(args.output + ".checkpoint", "w") as _f:
                    json.dump(_checkpoint, _f, default=str)
                tqdm.write(f"  [checkpoint saved: {len(results)} results]")

    return results


# ---------------------------------------------------------------------------
# Aggregation & reporting
# ---------------------------------------------------------------------------

def aggregate_and_print(results: list[dict]) -> dict:
    valid = [r for r in results if "mean_kl" in r and r["mean_kl"] is not None]
    errors = [r for r in results if "error" in r]

    if not valid:
        print("\nNo valid results to aggregate.")
        return {"error": "no valid results", "n_errors": len(errors)}

    kls = [r["mean_kl"] for r in valid]
    median_kls = [r["median_kl"] for r in valid if r.get("median_kl") is not None]
    max_kls = [r["max_kl"] for r in valid if r.get("max_kl") is not None]
    logratios = [r["mean_logratio"] for r in valid if r.get("mean_logratio") is not None]
    tf_lps = [r["tf_mean_lp"] for r in valid if r.get("tf_mean_lp") is not None]
    ni_lps = [r["ni_cross_mean_lp"] for r in valid if r.get("ni_cross_mean_lp") is not None]
    aligns = [r["alignment_rate"] for r in valid if r.get("alignment_rate") is not None]

    summary = {
        "n_total": len(results),
        "n_valid": len(valid),
        "n_errors": len(errors),
        "mean_kl": round(float(np.mean(kls)), 6),
        "std_kl": round(float(np.std(kls)), 6),
        "median_kl": round(float(np.median(kls)), 6),
        "p90_kl": round(float(np.percentile(kls, 90)), 6),
        "max_example_kl": round(float(np.max(kls)), 6),
        "mean_max_position_kl": round(float(np.mean(max_kls)), 6) if max_kls else None,
        "mean_logratio": round(float(np.mean(logratios)), 6) if logratios else None,
        "tf_mean_logprob": round(float(np.mean(tf_lps)), 4) if tf_lps else None,
        "ni_cross_mean_logprob": round(float(np.mean(ni_lps)), 4) if ni_lps else None,
        "mean_alignment_rate": round(float(np.mean(aligns)), 4) if aligns else None,
    }

    print(f"\n{'='*65}")
    print(f"  KL Divergence Check: Teacher-Forcing vs Natural-Inference")
    print(f"{'='*65}")
    print(f"  Examples evaluated:     {summary['n_valid']} / {summary['n_total']}  ({summary['n_errors']} errors)")
    print(f"  Token alignment rate:   {summary['mean_alignment_rate']:.2%}" if summary["mean_alignment_rate"] else "")
    print()
    print(f"  Per-position KL(P_TF || P_NI):")
    print(f"    Mean:                 {summary['mean_kl']:.6f}")
    print(f"    Std:                  {summary['std_kl']:.6f}")
    print(f"    Median:               {summary['median_kl']:.6f}")
    print(f"    90th percentile:      {summary['p90_kl']:.6f}")
    print(f"    Max (across examples):{summary['max_example_kl']:.6f}")
    if summary["mean_max_position_kl"] is not None:
        print(f"    Mean max-position KL: {summary['mean_max_position_kl']:.6f}")
    print()
    print(f"  Log-probability stats:")
    print(f"    Mean log-ratio (TF-NI): {summary['mean_logratio']:.6f}" if summary["mean_logratio"] is not None else "")
    print(f"    TF mean logprob:        {summary['tf_mean_logprob']:.4f}" if summary["tf_mean_logprob"] is not None else "")
    print(f"    NI cross-eval logprob:  {summary['ni_cross_mean_logprob']:.4f}" if summary["ni_cross_mean_logprob"] is not None else "")
    print(f"{'='*65}")

    # Interpretation
    mean_kl = summary["mean_kl"]
    if mean_kl < 0.1:
        verdict = "PASS - TF reasoning is well in-distribution"
    elif mean_kl < 0.5:
        verdict = "MARGINAL - some divergence, review high-KL examples"
    elif mean_kl < 2.0:
        verdict = "WARNING - moderate divergence from natural generation"
    else:
        verdict = "FAIL - TF reasoning is significantly out of distribution"
    print(f"\n  Verdict: {verdict}  (mean KL = {mean_kl:.4f})")
    print()

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="KL divergence check: TF vs NI reasoning traces"
    )
    parser.add_argument("--data", required=True, help="Path to aitw_general.json")
    parser.add_argument("--image-dir", required=True, help="Parent dir of aitw_general_images/")
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max generation tokens for reasoning traces")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-logprobs", type=int, default=20)
    parser.add_argument("--output", default="kl_tf_ni_results.json")
    parser.add_argument("--cross-eval-max-pixels", type=int, default=120000,
                        help="Max image pixels for cross-eval (reduces GPU OOM risk). "
                             "Default 120000 (~346x346). Set 0 to disable resizing.")
    parser.add_argument("--save-per-token", action="store_true",
                        help="Save per-token KL details (large output)")
    args = parser.parse_args()

    results = run_kl_check(args)
    summary = aggregate_and_print(results)

    output = {
        "config": {k: v for k, v in vars(args).items() if k != "save_per_token"},
        "summary": summary,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
