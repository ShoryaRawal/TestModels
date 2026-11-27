#!/usr/bin/env python3
"""05 — Mechanistic interpretability lab + interactive instruction chat.

Runs entirely offline against the checkpoints produced by scripts 03 and 04.

Sections
--------
1. Logit Lens: project every layer's residual stream through the final
   LayerNorm and the untied output embedding.
2. Induction Head Scanner: feed repeated random token sequences to every
   saved pretraining checkpoint and measure the classic induction score.
3. Causal head ablation: zero (or mean) ablate the strongest heads and report
   KL divergence, logit shift and generation degradation.
4. Sample generations from the instruction-tuned model.

Use ``--chat`` to talk to ``instruct_final.safetensors`` interactively.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np
from tokenizers import Tokenizer

from model import (
    PythiaConfig,
    generate,
    format_chat_prompt,
    load_model_from_checkpoint,
)

SPECIAL_TOKENS = ["<|endoftext|>", "<|padding|>", "<|user|>", "<|assistant|>"]


def verify_tokenizer(tok: Tokenizer) -> Dict[str, int]:
    ids: Dict[str, int] = {}
    for special in SPECIAL_TOKENS:
        idx = tok.token_to_id(special)
        if idx is None:
            raise SystemExit(f"tokenizer is missing special token {special!r}")
        enc = tok.encode(special, add_special_tokens=False).ids
        if enc != [idx]:
            raise SystemExit(f"{special!r} is not a single token: id={idx}, encode={enc}")
        ids[special] = idx
    return ids


def stop_ids(tok: Tokenizer) -> List[int]:
    """Always stop on EOS; also stop if the model tries to start a new turn."""
    return [
        tok.token_to_id("<|endoftext|>"),
        tok.token_to_id("<|user|>"),
        tok.token_to_id("<|assistant|>"),
        tok.token_to_id("<|padding|>"),
    ]


class Reporter:
    def __init__(self, path: Optional[str]):
        self.path = path
        self.lines: List[str] = []

    def emit(self, text: str = "") -> None:
        print(text, flush=True)
        self.lines.append(text)

    def save(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")


def step_number(path: str) -> int:
    m = re.search(r"step_(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 10**9


def tok_str(tok: Tokenizer, idx: int) -> str:
    s = tok.decode([int(idx)])
    s = s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return s if s else f"<{int(idx)}>"


def topk_tokens(tok: Tokenizer, logits: mx.array, k: int = 5) -> List[Tuple[str, float]]:
    probs = mx.softmax(logits.astype(mx.float32), axis=-1)
    order = mx.argsort(logits)[::-1][:k]
    mx.eval(order, probs)
    out = []
    for idx in np.array(order).tolist():
        out.append((tok_str(tok, idx), float(probs[int(idx)])))
    return out


# ---------------------------------------------------------------------------
# 1. Logit Lens
# ---------------------------------------------------------------------------
def logit_lens_report(model, tok: Tokenizer, prompts: Sequence[str], rep: Reporter) -> None:
    rep.emit("=" * 78)
    rep.emit("1. LOGIT LENS — next-token predictions by layer")
    rep.emit("=" * 78)
    for prompt in prompts:
        ids = tok.encode(prompt, add_special_tokens=False).ids
        if not ids:
            continue
        x = mx.array([ids], dtype=mx.int32)
        lens = model.logit_lens(x)
        rep.emit("")
        rep.emit(f"prompt: {prompt!r}")
        rep.emit(f"{'stage':<8} {'top-1':<16} {'p':>8}   top-5")
        rep.emit("-" * 78)
        for layer_idx, logits in enumerate(lens):
            stage = "final" if layer_idx == len(lens) - 1 else f"L{layer_idx}"
            last = logits[0, -1, :]
            top = topk_tokens(tok, last, 5)
            top1, p1 = top[0]
            rest = ", ".join(f"{t}({p:.3f})" for t, p in top[1:])
            rep.emit(f"{stage:<8} {top1:<16} {p1:>8.4f}   {rest}")
        rep.emit("")


# ---------------------------------------------------------------------------
# 2. Induction head scanner
# ---------------------------------------------------------------------------
def induction_scores(model, ids: np.ndarray) -> np.ndarray:
    """Return (n_layers, n_heads) induction scores for a repeated sequence."""
    L = len(ids) // 2
    x = mx.array(ids[None, :], dtype=mx.int32)
    _, cache = model(x, return_cache=True)
    scores = []
    idx = mx.arange(L - 1)
    for layer in cache["layers"]:
        probs = layer["attn_probs"][0]  # (H, 2L, 2L)
        # Position L+j (second copy of token j) attending to position j+1
        # (the token that followed the first copy of token j).
        vals = probs[:, L + idx, idx + 1]
        scores.append(np.array(mx.mean(vals, axis=-1)))
    return np.stack(scores)  # (L, H)


def induction_report(
    pretrain_dir: str,
    tok: Tokenizer,
    seq_half: int,
    threshold: float,
    rep: Reporter,
    max_checkpoints: int = 0,
) -> Optional[List[Tuple[int, int, float]]]:
    rep.emit("=" * 78)
    rep.emit("2. INDUCTION HEAD SCANNER — developmental tracking")
    rep.emit("=" * 78)
    paths = sorted(glob.glob(os.path.join(pretrain_dir, "step_*.safetensors")), key=step_number)
    if max_checkpoints and len(paths) > max_checkpoints:
        keep = np.linspace(0, len(paths) - 1, max_checkpoints).astype(int)
        paths = [paths[i] for i in sorted(set(keep.tolist()))]
    final_path = os.path.join(pretrain_dir, "base_final.safetensors")
    entries: List[Tuple[str, int, np.ndarray]] = []
    if not paths:
        rep.emit("no pretraining checkpoints found")
        return None

    rng = np.random.default_rng(1234)
    vocab = tok.get_vocab_size()
    half = rng.integers(100, max(101, min(vocab - 1, 3000)), size=seq_half).astype(np.int32)
    ids = np.concatenate([half, half])
    rep.emit(f"repeated random sequence: {seq_half} tokens x 2 (common token ids); uniform baseline = {1/(2*seq_half):.3f}")
    rep.emit("")
    rep.emit(f"{'step':>8}  {'best':>7}  {'(layer,head)':>16}  top-3")
    rep.emit("-" * 78)

    config_path = os.path.join(pretrain_dir, "config.json")
    for path in paths:
        model, _ = load_model_from_checkpoint(path, config_path=config_path, dtype="float32")
        scores = induction_scores(model, ids)
        best = float(scores.max())
        l, h = np.unravel_index(int(scores.argmax()), scores.shape)
        flat = scores.reshape(-1)
        top_idx = np.argsort(flat)[::-1][:3]
        top_txt = ", ".join(
            f"L{int(i)//scores.shape[1]}H{int(i)%scores.shape[1]}={flat[i]:.2f}" for i in top_idx
        )
        step = step_number(path)
        rep.emit(f"{step:>8}  {best:>7.3f}  {f'L{int(l)}H{int(h)}':>16}  {top_txt}")
        entries.append((os.path.basename(path), step, scores))
        del model
        mx.clear_cache()

    if os.path.exists(final_path) and (not paths or os.path.basename(paths[-1]) != "base_final.safetensors"):
        model, _ = load_model_from_checkpoint(final_path, config_path=config_path, dtype="float32")
        scores = induction_scores(model, ids)
        best = float(scores.max())
        l, h = np.unravel_index(int(scores.argmax()), scores.shape)
        rep.emit(f"{'final':>8}  {best:>7.3f}  {f'L{int(l)}H{int(h)}':>16}  base_final.safetensors")
        entries.append(("base_final", 10**9, scores))
        del model
        mx.clear_cache()

    # Emergence table: for every (layer, head), final score and first step above
    # threshold.
    final_scores = entries[-1][2]
    n_layers, n_heads = final_scores.shape
    rows = []
    for l in range(n_layers):
        for h in range(n_heads):
            first = None
            first_score = None
            for _, step, sc in entries:
                if sc[l, h] >= threshold:
                    first = step
                    first_score = float(sc[l, h])
                    break
            rows.append((float(final_scores[l, h]), l, h, first, first_score))
    rows.sort(reverse=True)

    rep.emit("")
    rep.emit(f"emergence (threshold={threshold}):")
    rep.emit(f"{'rank':>4}  {'layer':>5}  {'head':>4}  {'final':>7}  {'emerged_at':>10}  {'score_then':>10}")
    rep.emit("-" * 78)
    for rank, (score, l, h, first, first_score) in enumerate(rows[:16], 1):
        emerged = "never" if first is None else ("final" if first == 10**9 else str(first))
        then = "-" if first_score is None else f"{first_score:.3f}"
        rep.emit(f"{rank:>4}  {l:>5}  {h:>4}  {score:>7.3f}  {emerged:>10}  {then:>10}")

    top_heads = [(l, h, score) for score, l, h, _, _ in rows[:6]]
    rep.emit("")
    return top_heads


# ---------------------------------------------------------------------------
# 3. Causal head ablation
# ---------------------------------------------------------------------------
def kl_divergence(logits_a: mx.array, logits_b: mx.array) -> float:
    p = mx.softmax(logits_a.astype(mx.float32), axis=-1)
    q = mx.softmax(logits_b.astype(mx.float32), axis=-1)
    kl = mx.sum(p * (mx.log(p + 1e-9) - mx.log(q + 1e-9)))
    mx.eval(kl)
    return float(kl)


def ablation_report(
    model,
    tok: Tokenizer,
    top_heads: Sequence[Tuple[int, int, float]],
    prompt: str,
    rep: Reporter,
    max_new_tokens: int = 80,
    stops: Sequence[int] = (),
) -> None:
    rep.emit("=" * 78)
    rep.emit("3. CAUSAL HEAD ABLATION")
    rep.emit("=" * 78)
    if not top_heads:
        rep.emit("no heads available")
        return
    ids = tok.encode(prompt, add_special_tokens=False).ids
    x = mx.array([ids], dtype=mx.int32)
    base_logits = model(x)[0, -1, :]
    mx.eval(base_logits)
    base_top = topk_tokens(tok, base_logits, 3)
    rep.emit(f"prompt: {prompt!r}")
    rep.emit(f"clean top-3: " + ", ".join(f"{t}({p:.3f})" for t, p in base_top))
    rep.emit("")
    rep.emit(f"{'ablation':<24} {'KL':>10} {'max|dlogit|':>12}  ablated top-1")
    rep.emit("-" * 78)

    for l, h, score in top_heads[:4]:
        ab = {int(l): [int(h)]}
        ablated = model(x, ablate_heads=ab)[0, -1, :]
        mx.eval(ablated)
        kl = kl_divergence(base_logits, ablated)
        shift = float(mx.max(mx.abs(base_logits - ablated)))
        top1 = topk_tokens(tok, ablated, 1)[0][0]
        rep.emit(f"L{l}H{h} (ind={score:.2f})".ljust(24) + f" {kl:>10.4f} {shift:>12.3f}  {top1}")

    combined = {int(l): [] for l, _, _ in top_heads[:3]}
    for l, h, _ in top_heads[:3]:
        combined[int(l)].append(int(h))
    ablated = model(x, ablate_heads=combined)[0, -1, :]
    mx.eval(ablated)
    kl = kl_divergence(base_logits, ablated)
    shift = float(mx.max(mx.abs(base_logits - ablated)))
    top1 = topk_tokens(tok, ablated, 1)[0][0]
    rep.emit(f"top-3 combined".ljust(24) + f" {kl:>10.4f} {shift:>12.3f}  {top1}")

    # Generation degradation (instruct model only; may be slow on base).
    try:
        clean = generate(
            model, tok, prompt, max_new_tokens=max_new_tokens, temperature=0.7, top_k=40,
            seed=7, stop_token_ids=stops,
        )
        ablated_text = generate(
            model, tok, prompt, max_new_tokens=max_new_tokens, temperature=0.7, top_k=40,
            seed=7, ablate_heads=combined, stop_token_ids=stops,
        )
        rep.emit("")
        rep.emit("clean generation:")
        rep.emit(clean.strip() or "<empty>")
        rep.emit("")
        rep.emit("top-3-combined ablation generation:")
        rep.emit(ablated_text.strip() or "<empty>")
    except Exception as exc:  # pragma: no cover - diagnostic path
        rep.emit(f"generation comparison skipped: {exc}")
    rep.emit("")


# ---------------------------------------------------------------------------
# 4. Sample generations
# ---------------------------------------------------------------------------
def generation_report(model, tok: Tokenizer, rep: Reporter, max_new_tokens: int, instruct: bool) -> None:
    rep.emit("=" * 78)
    rep.emit("4. SAMPLE GENERATIONS")
    rep.emit("=" * 78)
    stops = stop_ids(tok)
    if instruct:
        items = [
            ("hello", format_chat_prompt("hello"), min(50, max_new_tokens), 0.7),
            ("who are you?", format_chat_prompt("who are you?"), min(60, max_new_tokens), 0.7),
            ("what", format_chat_prompt("what"), min(50, max_new_tokens), 0.7),
            ("write a short story about a brave cat", format_chat_prompt("write a short story about a brave cat"), max_new_tokens, 0.8),
            ("moral of the slow turtle story", format_chat_prompt("What lesson does the story of the slow turtle teach?"), min(60, max_new_tokens), 0.7),
            ("one-sentence summary", format_chat_prompt("Summarize this story in one sentence: A tiny bird was afraid to fly, but her friends cheered her on until she tried."), min(60, max_new_tokens), 0.7),
        ]
    else:
        items = [
            ("story continuation", "Once upon a time, there was a little girl named Lily who", max_new_tokens, 0.8),
            ("story continuation", "The small dragon looked at the sky and", max_new_tokens, 0.8),
            ("story continuation", "One day, a clever fox found a", max_new_tokens, 0.8),
        ]
    for label, prompt, n_tokens, temp in items:
        text = generate(
            model,
            tok,
            prompt,
            max_new_tokens=n_tokens,
            temperature=temp,
            top_k=40,
            seed=11,
            stop_token_ids=stops,
        )
        rep.emit("")
        rep.emit(f"prompt: {label!r}")
        rep.emit(f"output: {text.strip() or '<empty>'}")
    rep.emit("")


# ---------------------------------------------------------------------------
# Interactive chat
# ---------------------------------------------------------------------------
def build_chat_prompt(
    history: Sequence[Tuple[str, str]],
    user: str,
    tok: Tokenizer,
    max_seq_len: int,
    max_new_tokens: int,
) -> Tuple[str, Dict[str, int | bool]]:
    """Format the next turn with the exact training template.

    Old turns are dropped from the front until the prompt fits, and an oversized
    current user message is token-truncated.  Returns ``(prompt, stats)``.
    """
    user = user.strip()
    reserve = max(16, min(max_new_tokens, max_seq_len - 64))
    budget = max(32, max_seq_len - reserve)

    current = f"<|user|>\n{user}\n<|assistant|>\n"
    was_trimmed = False
    if len(tok.encode(current, add_special_tokens=False).ids) > budget:
        overhead = len(tok.encode("<|user|>\n\n<|assistant|>\n", add_special_tokens=False).ids)
        room = max(1, budget - overhead)
        user_ids = tok.encode(user, add_special_tokens=False).ids
        # Byte-level BPE boundaries can change the count after decoding, so
        # shrink the user slice until the rebuilt prompt actually fits.
        while room > 1:
            user = tok.decode(user_ids[:room]).strip()
            current = f"<|user|>\n{user}\n<|assistant|>\n"
            if len(tok.encode(current, add_special_tokens=False).ids) <= budget:
                break
            room -= 1
        was_trimmed = True

    kept: List[Tuple[str, str]] = []
    for u, a in reversed(history):
        block = f"<|user|>\n{u.strip()}\n<|assistant|>\n{a.strip()}<|endoftext|>"
        prefix = "".join(
            f"<|user|>\n{uu.strip()}\n<|assistant|>\n{aa.strip()}<|endoftext|>"
            for uu, aa in reversed(kept)
        )
        candidate = prefix + block + current
        if len(tok.encode(candidate, add_special_tokens=False).ids) <= budget:
            kept.append((u, a))
        else:
            break

    parts = [
        f"<|user|>\n{u.strip()}\n<|assistant|>\n{a.strip()}<|endoftext|>"
        for u, a in reversed(kept)
    ]
    prompt = "".join(parts) + current
    prompt_tokens = len(tok.encode(prompt, add_special_tokens=False).ids)
    dropped = len(history) - len(kept)
    stats: Dict[str, int | bool] = {
        "prompt_tokens": prompt_tokens,
        "budget": budget,
        "max_seq_len": max_seq_len,
        "reserve": reserve,
        "kept_turns": len(kept),
        "dropped_turns": dropped,
        "trimmed": was_trimmed or dropped > 0,
        "user_truncated": was_trimmed,
    }
    return prompt, stats


def parse_ablation(text: str) -> Dict[int, List[int]]:
    ab: Dict[int, List[int]] = {}
    for part in text.replace(",", " ").split():
        if ":" not in part:
            continue
        layer_s, heads_s = part.split(":", 1)
        try:
            layer = int(layer_s)
        except ValueError:
            continue
        heads = []
        for h in heads_s.replace("|", ",").split(","):
            if h.strip():
                try:
                    heads.append(int(h))
                except ValueError:
                    pass
        if heads:
            ab[layer] = heads
    return ab


def chat(model, tok: Tokenizer, args, max_seq_len: int) -> None:
    print(f"\nInteractive instruction model (context window: {max_seq_len} tokens). Commands:")
    print("  /quit              exit")
    print("  /reset             clear conversation history")
    print("  /ctx               show context window usage and kept turns")
    print("  /maxnew 120        set max new tokens reserved for replies")
    print("  /ablate 0:1,2 3:0  zero-ablate heads")
    print("  /clearablate       remove ablation")
    print("  /temp 0.7          set temperature")
    print("  /topk 40           set top-k\n")
    history: List[Tuple[str, str]] = []
    ablation: Optional[Dict[int, List[int]]] = None
    temperature = args.temperature
    top_k = args.top_k
    max_new_tokens = args.max_new_tokens
    stops = stop_ids(tok)
    last_stats: Optional[Dict[str, int | bool]] = None
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit"):
            break
        if user == "/reset":
            history = []
            last_stats = None
            print("(history cleared)")
            continue
        if user == "/ctx":
            reserve = max(16, min(max_new_tokens, max_seq_len - 64))
            budget = max(32, max_seq_len - reserve)
            hist_text = "".join(
                f"<|user|>\n{u.strip()}\n<|assistant|>\n{a.strip()}<|endoftext|>"
                for u, a in history
            )
            hist_tok = len(tok.encode(hist_text, add_special_tokens=False).ids) if hist_text else 0
            print(
                f"(window={max_seq_len} tok | prompt_budget={budget} | max_new={reserve} | "
                f"history={len(history)} turns, {hist_tok} tok)"
            )
            for idx, (u, a) in enumerate(history, 1):
                print(f"  [{idx}] you: {u[:60]!r} -> model: {a[:60]!r}")
            continue
        if user.startswith("/maxnew "):
            try:
                max_new_tokens = max(16, min(int(user.split()[1]), max_seq_len - 32))
                print(f"(max_new_tokens = {max_new_tokens})")
            except (IndexError, ValueError):
                print("usage: /maxnew 120")
            continue
        if user == "/clearablate":
            ablation = None
            print("(ablation cleared)")
            continue
        if user.startswith("/ablate "):
            ablation = parse_ablation(user[len("/ablate "):])
            print(f"(ablation = {ablation})")
            continue
        if user.startswith("/temp "):
            try:
                temperature = float(user.split()[1])
                print(f"(temperature = {temperature})")
            except (IndexError, ValueError):
                print("usage: /temp 0.7")
            continue
        if user.startswith("/topk "):
            try:
                top_k = int(user.split()[1])
                print(f"(top_k = {top_k})")
            except (IndexError, ValueError):
                print("usage: /topk 40")
            continue

        prompt, stats = build_chat_prompt(
            history, user, tok, max_seq_len, max_new_tokens
        )
        if stats["user_truncated"]:
            print("(current message truncated to fit context budget)")
        elif stats["dropped_turns"]:
            print(f"(trimmed {stats['dropped_turns']} oldest turn(s) to fit {stats['budget']}-token prompt budget)")
        t0 = time.time()
        reply = generate(
            model,
            tok,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            ablate_heads=ablation,
            seed=None,
            stop_token_ids=stops,
        ).strip()
        dt = time.time() - t0
        if not reply:
            reply = "(no reply)"
        reply_tok = len(tok.encode(reply, add_special_tokens=False).ids)
        total_tok = int(stats["prompt_tokens"]) + reply_tok
        print(
            f"model> {reply}\n"
            f"        [{dt:.1f}s | ctx {total_tok}/{max_seq_len} tok "
            f"(in {stats['prompt_tokens']}/{stats['budget']}, out {reply_tok}) | "
            f"turns kept {int(stats['kept_turns']) + 1}]\n"
        )
        history.append((user, reply))
        last_stats = stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-dir", default="checkpoints/pretrain")
    parser.add_argument("--instruct-checkpoint", default="checkpoints/instruct/instruct_final.safetensors")
    parser.add_argument("--rl-checkpoint", default="checkpoints/rl/rl_final.safetensors")
    parser.add_argument("--use-sft", action="store_true", help="ignore rl_final.safetensors and load SFT checkpoint")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--report", default="artifacts/interpretability_report.txt")
    parser.add_argument("--chat", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--context-len", type=int, default=0, help="override model max_seq_len for chat (e.g. 1024)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--induction-len", type=int, default=32)
    parser.add_argument("--max-checkpoints", type=int, default=0)
    args = parser.parse_args()

    if not os.path.exists(args.tokenizer):
        raise SystemExit(f"missing tokenizer: {args.tokenizer}")
    tok = Tokenizer.from_file(args.tokenizer)
    special_ids = verify_tokenizer(tok)
    print(f"special token ids: {special_ids}")
    stops = stop_ids(tok)

    active_ckpt = args.instruct_checkpoint
    if (
        not args.use_sft
        and args.instruct_checkpoint == "checkpoints/instruct/instruct_final.safetensors"
        and os.path.exists(args.rl_checkpoint)
    ):
        active_ckpt = args.rl_checkpoint

    if args.chat:
        if not os.path.exists(active_ckpt):
            raise SystemExit(f"missing instruct checkpoint: {active_ckpt}")
        model, config = load_model_from_checkpoint(active_ckpt, dtype="float32")
        ctx_len = args.context_len if args.context_len > 0 else config.max_seq_len
        config.max_seq_len = ctx_len
        model.config.max_seq_len = ctx_len
        print(f"loaded model ({active_ckpt}): {config.num_hidden_layers} layers, {config.hidden_size} hidden, ctx={ctx_len}")
        chat(model, tok, args, ctx_len)
        return

    rep = Reporter(args.report)
    rep.emit("=" * 78)
    rep.emit("SMALL PYTHIA / GPT-NEOX — MECHANISTIC INTERPRETABILITY REPORT")
    rep.emit(f"generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    rep.emit("=" * 78)

    base_final = os.path.join(args.pretrain_dir, "base_final.safetensors")
    if not os.path.exists(base_final):
        raise SystemExit(f"missing base checkpoint: {base_final}")
    base_model, config = load_model_from_checkpoint(base_final, dtype="float32")
    rep.emit(
        f"model: {config.num_hidden_layers} layers, hidden={config.hidden_size}, "
        f"heads={config.num_attention_heads}, head_dim={config.head_dim}, "
        f"rotary_dim={config.rotary_dim}, vocab={config.vocab_size}"
    )
    rep.emit("")

    prompts = [
        "Once upon a time, there was a little girl named Lily who",
        "The capital of France is",
        "1, 2, 3, 4, 5,",
    ]
    logit_lens_report(base_model, tok, prompts, rep)

    top_heads = induction_report(
        args.pretrain_dir,
        tok,
        args.induction_len,
        args.threshold,
        rep,
        args.max_checkpoints,
    )

    # Use the RL / instruction model for the ablation demonstration when available.
    if os.path.exists(active_ckpt):
        instr_model, _ = load_model_from_checkpoint(active_ckpt, dtype="float32")
        ablation_model = instr_model
        instruct = True
    else:
        ablation_model = base_model
        instruct = False

    ablation_prompt = format_chat_prompt("Write a short story about a brave little robot.")
    ablation_report(
        ablation_model,
        tok,
        top_heads or [],
        ablation_prompt,
        rep,
        max_new_tokens=min(80, args.max_new_tokens),
        stops=stops,
    )
    generation_report(ablation_model, tok, rep, min(120, args.max_new_tokens), instruct)
    rep.save()
    print(f"\nreport saved to {args.report}")


if __name__ == "__main__":
    main()
