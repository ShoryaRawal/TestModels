#!/usr/bin/env python3
"""08 — Continual learning & volatility stress test.

Runs multi-cycle continual learning across four mixed streams:
  1. Old pretraining text replay (data/train_tokens.npy — TinyStories)
  2. New open-web text (data/open_web_train_tokens.npy — Wikipedia/SimpleWiki)
  3. Instruction SFT replay (data/sft_train_v2.jsonl)
  4. Online GRPO / RLVR updates with break-test failure mining

Optionally depth-grows the model (e.g. 6 -> 12 layers via ``grow_model_depth``)
before continual training, and evaluates stability/forgetting after every cycle:
old-text val loss, new-text val loss, SFT val loss, KL drift from initial
weights, gradient norms, smoke-test pass rate, break-test pass rate, and
degeneracy/looping rate.

Outputs
-------
checkpoints/continual/continual_final.safetensors
checkpoints/continual/config.json
artifacts/continual_metrics.jsonl
artifacts/volatility_report.txt
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mlx_utils
import numpy as np
from tokenizers import Tokenizer

from model import (
    _masked_cross_entropy,
    batched_generate,
    format_chat_prompt,
    generate,
    grow_model_depth,
    load_model_from_checkpoint,
    sequence_logprobs,
)

rl_mod = importlib.import_module("06_rl_loop")


def log(msg: str) -> None:
    print(f"[08 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def grad_norm(grads) -> mx.array:
    leaves = [g for _, g in mlx_utils.tree_flatten(grads)]
    if not leaves:
        return mx.array(0.0, dtype=mx.float32)
    return mx.sqrt(sum(mx.sum(g.astype(mx.float32) ** 2) for g in leaves))


def clip_grads_with_norm(grads, max_norm: float):
    norm = grad_norm(grads)
    scale = mx.minimum(
        mx.array(1.0, dtype=mx.float32),
        mx.array(max_norm, dtype=mx.float32) / (norm + 1e-6),
    )
    clipped = mlx_utils.tree_map(lambda g: g * scale.astype(g.dtype), grads)
    return clipped, norm


def eval_token_loss(model, tokens: Optional[np.ndarray], seq_len: int = 256, n_batches: int = 8, batch_size: int = 8) -> float:
    if tokens is None or len(tokens) <= seq_len + 2:
        return float("nan")
    max_start = len(tokens) - seq_len - 1
    starts = np.linspace(0, max_start, n_batches * batch_size).astype(np.int64).reshape(n_batches, batch_size)
    losses = []
    for row in starts:
        x = np.stack([np.asarray(tokens[s : s + seq_len]) for s in row]).astype(np.int32)
        y = np.stack([np.asarray(tokens[s + 1 : s + seq_len + 1]) for s in row]).astype(np.int32)
        loss = mx.mean(nn.losses.cross_entropy(model(mx.array(x)).astype(mx.float32), mx.array(y)))
        mx.eval(loss)
        losses.append(float(loss))
    return float(sum(losses) / max(len(losses), 1))


def eval_sft_loss(model, sft_rows: Sequence[Dict[str, np.ndarray]], batch_size: int = 8, n_batches: int = 8) -> float:
    if not sft_rows:
        return float("nan")
    idx = np.linspace(0, len(sft_rows) - 1, min(len(sft_rows), batch_size * n_batches)).astype(np.int64)
    losses = []
    for start in range(0, len(idx), batch_size):
        b = idx[start : start + batch_size]
        x = mx.array(np.stack([sft_rows[int(i)]["input_ids"] for i in b]))
        y = mx.array(np.stack([sft_rows[int(i)]["labels"] for i in b]))
        # Align logits[:, :-1, :] with y[:, 1:].
        logits = model(x).astype(mx.float32)
        loss = _masked_cross_entropy(logits[:, :-1, :], y[:, 1:], ignore_index=-100)
        mx.eval(loss)
        losses.append(float(loss))
    return float(sum(losses) / max(len(losses), 1))


def eval_kl_from_ref(model, ref_model, probe_ids: mx.array) -> float:
    logits_p = model(probe_ids).astype(mx.float32)
    logits_q = ref_model(probe_ids).astype(mx.float32)
    p = mx.softmax(logits_p, axis=-1)
    kl = mx.mean(mx.sum(p * (mx.log(p + 1e-8) - mx.log(mx.softmax(logits_q, axis=-1) + 1e-8)), axis=-1))
    mx.eval(kl)
    return float(kl)


def eval_core_smoke(model, tok: Tokenizer, stops: Sequence[int]) -> Tuple[int, int]:
    cases = [
        ("hello", lambda low, r: bool(r) and not low.startswith("once upon a time") and (low.startswith(("hi", "hello", "hey")) or "help" in low)),
        ("who are you?", lambda low, r: bool(r) and not low.startswith("once upon a time") and any(w in low for w in ("model", "pythia", "language", "macbook"))),
        ("what", lambda low, r: bool(r) and not low.startswith("once upon a time") and any(w in low for w in ("mean", "more", "explain", "context", "tell me"))),
        ("write a short story about a brave cat", lambda low, r: len(r) >= 40 and low.startswith("once upon a time")),
    ]
    passed = 0
    for q, fn in cases:
        reply = generate(
            model, tok, format_chat_prompt(q), max_new_tokens=96,
            temperature=0.0, top_k=0, seed=7, stop_token_ids=stops,
        ).strip()
        if fn(reply.lower(), reply):
            passed += 1
    return passed, len(cases)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/rl/rl_final.safetensors")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--checkpoints-dir", default="checkpoints/continual")
    parser.add_argument("--grow-layers", type=int, default=0, help="if > current layers, depth-grow model before continual run")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--steps-per-cycle", type=int, default=32)
    parser.add_argument("--lm-seq-len", type=int, default=256)
    parser.add_argument("--sft-seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--kl-beta", type=float, default=0.03)
    parser.add_argument("--rl-group-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    tok = Tokenizer.from_file(args.tokenizer)
    rl_mod.verify_tokenizer(tok)
    pad_id = int(tok.token_to_id("<|padding|>"))
    stops = rl_mod.stop_ids(tok)

    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        ckpt_path = "checkpoints/instruct/instruct_final.safetensors"
    model, config = load_model_from_checkpoint(ckpt_path, dtype="float32")
    model.set_dtype(mx.float32)

    if args.grow_layers and args.grow_layers > config.num_hidden_layers:
        probe_x = mx.array([tok.encode(format_chat_prompt("Once upon a time"), add_special_tokens=False).ids], dtype=mx.int32)
        base_logits = model(probe_x)
        mx.eval(base_logits)
        old_l = config.num_hidden_layers
        model, config = grow_model_depth(model, args.grow_layers)
        grown_logits = model(probe_x)
        mx.eval(grown_logits)
        grow_diff = float(mx.max(mx.abs(base_logits - grown_logits)))
        log(f"depth-grew {ckpt_path}: {old_l} -> {config.num_hidden_layers} layers (step-0 max|dlogit|={grow_diff:.6f})")

    n_params = sum(v.size for _, v in mlx_utils.tree_flatten(model.parameters()))
    log(f"continual model: {n_params/1e6:.2f}M params, {config.num_hidden_layers}L, hidden={config.hidden_size}, ctx={config.max_seq_len}")

    # Snapshot initial weights & reference model for KL/drift tracking.
    init_weights = {k: mx.array(v) for k, v in mlx_utils.tree_flatten(model.parameters())}
    ref_model, _ = load_model_from_checkpoint(ckpt_path, dtype="float32")
    if args.grow_layers and args.grow_layers > ref_model.config.num_hidden_layers:
        ref_model, _ = grow_model_depth(ref_model, args.grow_layers)
    ref_model.set_dtype(mx.float32)
    mx.eval(init_weights, ref_model.parameters())

    # Load old pretraining texts, new open-web texts, and SFT replay.
    old_train = np.load(os.path.join(args.data_dir, "train_tokens.npy"), mmap_mode="r")
    old_valid = np.load(os.path.join(args.data_dir, "valid_tokens.npy"), mmap_mode="r")
    ow_train_path = os.path.join(args.data_dir, "open_web_train_tokens.npy")
    ow_valid_path = os.path.join(args.data_dir, "open_web_valid_tokens.npy")
    ow_train = np.load(ow_train_path, mmap_mode="r") if os.path.exists(ow_train_path) else None
    ow_valid = np.load(ow_valid_path, mmap_mode="r") if os.path.exists(ow_valid_path) else None
    log(
        f"corpora: old_train={len(old_train):,} tok | "
        f"open_web_train={(len(ow_train) if ow_train is not None else 0):,} tok"
    )

    sft_train = rl_mod.load_sft_anchor_rows(os.path.join(args.data_dir, "sft_train_v2.jsonl"), tok, args.sft_seq_len)
    sft_valid = rl_mod.load_sft_anchor_rows(os.path.join(args.data_dir, "sft_valid_v2.jsonl"), tok, args.sft_seq_len)
    log(f"SFT replay: train={len(sft_train):,}, valid={len(sft_valid):,}")

    suite = rl_mod.build_break_suite()
    probe_texts = [
        "Once upon a time, there was a little girl named Lily who",
        format_chat_prompt("hello"),
        format_chat_prompt("What do you think the capital of Mongolia is?"),
        format_chat_prompt("what is 9047 + 837"),
    ]
    probe_max = max(len(tok.encode(t, add_special_tokens=False).ids) for t in probe_texts)
    probe_mat = np.full((len(probe_texts), probe_max), pad_id, dtype=np.int32)
    for i, t in enumerate(probe_texts):
        ids = tok.encode(t, add_special_tokens=False).ids
        probe_mat[i, : len(ids)] = ids
    probe_ids = mx.array(probe_mat)

    optimizer = optim.AdamW(learning_rate=args.lr, betas=[0.9, 0.95], eps=1e-8, weight_decay=0.01)

    def lm_loss_fn(m, x, y):
        logits = m(x).astype(mx.float32)
        return mx.mean(nn.losses.cross_entropy(logits, y))

    def sft_loss_fn(m, x, y):
        logits = m(x).astype(mx.float32)
        return _masked_cross_entropy(logits[:, :-1, :], y[:, 1:], ignore_index=-100)

    def rl_loss_fn(m, seq_ids, gen_mask, advs, old_lp, ref_lp):
        new_lp = sequence_logprobs(m, seq_ids)
        log_ratio = new_lp - old_lp
        ratio = mx.exp(mx.clip(log_ratio, -10.0, 10.0))
        adv_b = advs[:, None]
        pg_loss = -mx.sum(mx.minimum(ratio * adv_b, mx.clip(ratio, 0.8, 1.2) * adv_b) * gen_mask) / mx.maximum(mx.sum(gen_mask), mx.array(1.0, dtype=mx.float32))
        ref_diff = ref_lp - new_lp
        kl = mx.exp(mx.clip(ref_diff, -10.0, 10.0)) - ref_diff - 1.0
        kl_loss = mx.sum(kl * gen_mask) / mx.maximum(mx.sum(gen_mask), mx.array(1.0, dtype=mx.float32))
        return pg_loss + args.kl_beta * kl_loss

    lm_vg = nn.value_and_grad(model, lm_loss_fn)
    sft_vg = nn.value_and_grad(model, sft_loss_fn)
    rl_vg = nn.value_and_grad(model, rl_loss_fn)
    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step_lm(x, y, lr):
        optimizer.learning_rate = lr
        loss, grads = lm_vg(model, x, y)
        grads, gnorm = clip_grads_with_norm(grads, 1.0)
        optimizer.update(model, grads)
        return loss, gnorm

    @partial(mx.compile, inputs=state, outputs=state)
    def step_sft(x, y, lr):
        optimizer.learning_rate = lr
        loss, grads = sft_vg(model, x, y)
        grads, gnorm = clip_grads_with_norm(grads, 1.0)
        optimizer.update(model, grads)
        return loss, gnorm

    @partial(mx.compile, inputs=state, outputs=state)
    def step_rl(seq_ids, gen_mask, advs, old_lp, ref_lp, lr):
        optimizer.learning_rate = lr
        loss, grads = rl_vg(model, seq_ids, gen_mask, advs, old_lp, ref_lp)
        grads, gnorm = clip_grads_with_norm(grads, 1.0)
        optimizer.update(model, grads)
        return loss, gnorm

    def sample_lm_batch(tokens: np.ndarray) -> Tuple[mx.array, mx.array]:
        max_s = len(tokens) - args.lm_seq_len - 1
        starts = rng.integers(0, max_s, size=args.batch_size)
        x = np.stack([np.asarray(tokens[s : s + args.lm_seq_len]) for s in starts]).astype(np.int32)
        y = np.stack([np.asarray(tokens[s + 1 : s + args.lm_seq_len + 1]) for s in starts]).astype(np.int32)
        return mx.array(x), mx.array(y)

    def measure_cycle(cycle: int, max_gn: float) -> Dict[str, Any]:
        old_vl = eval_token_loss(model, old_valid, args.lm_seq_len)
        ow_vl = eval_token_loss(model, ow_valid, args.lm_seq_len)
        sft_vl = eval_sft_loss(model, sft_valid)
        kl = eval_kl_from_ref(model, ref_model, probe_ids)
        cur = dict(mlx_utils.tree_flatten(model.parameters()))
        p_delta = max(float(mx.max(mx.abs(cur[k] - init_weights[k]))) for k in cur)
        smoke_ok, smoke_tot = eval_core_smoke(model, tok, stops)
        brk = rl_mod.evaluate_break_suite(model, tok, suite, stops, max_new_tokens=96)
        degen = sum(1 for d in brk["details"] if d["reason"] in ("ngram_loop", "empty")) / max(len(brk["details"]), 1)
        row = {
            "cycle": cycle,
            "old_pretrain_val": round(old_vl, 4),
            "open_web_val": round(ow_vl, 4) if not np.isnan(ow_vl) else None,
            "sft_val": round(sft_vl, 4) if not np.isnan(sft_vl) else None,
            "kl_from_init": round(kl, 4),
            "max_grad_norm": round(max_gn, 3),
            "max_param_delta": round(p_delta, 4),
            "smoke_pass": f"{smoke_ok}/{smoke_tot}",
            "smoke_ok": smoke_ok,
            "break_pass": f"{brk['passed']}/{brk['total']}",
            "break_overall": round(brk["overall"], 4),
            "degeneracy_rate": round(degen, 4),
        }
        return row, brk["failures"]

    metrics_path = os.path.join(args.artifacts_dir, "continual_metrics.jsonl")
    mf = open(metrics_path, "w", encoding="utf-8")

    row0, hard_pool = measure_cycle(0, 0.0)
    mf.write(json.dumps(row0) + "\n")
    mf.flush()
    log(f"cycle 0 (baseline): {row0}")

    volatile_reasons: List[str] = []
    best_cycles: List[Dict[str, Any]] = [row0]

    for cyc in range(1, args.cycles + 1):
        max_gn = 0.0
        for step in range(args.steps_per_cycle):
            slot = step % 4
            lr_arr = mx.array(args.lr, dtype=mx.float32)

            if slot == 0 or (slot == 1 and ow_train is None):
                x, y = sample_lm_batch(old_train)
                loss, gn = step_lm(x, y, lr_arr)
                mx.eval(loss, gn, state)
            elif slot == 1 and ow_train is not None:
                x, y = sample_lm_batch(ow_train)
                loss, gn = step_lm(x, y, lr_arr)
                mx.eval(loss, gn, state)
            elif slot == 2 and sft_train:
                idx = rng.integers(0, len(sft_train), size=args.batch_size)
                x = mx.array(np.stack([sft_train[int(i)]["input_ids"] for i in idx]))
                y = mx.array(np.stack([sft_train[int(i)]["labels"] for i in idx]))
                loss, gn = step_sft(x, y, lr_arr)
                mx.eval(loss, gn, state)
            else:
                prompts = rl_mod.sample_procedural_prompts(rng, 2, hard_pool)
                packed = []
                for item in prompts:
                    roll = batched_generate(
                        model, tok, item.prompt,
                        num_samples=args.rl_group_size, max_new_tokens=96,
                        temperature=0.85, top_k=40, stop_token_ids=stops,
                    )
                    rews = np.asarray([rl_mod.score_completion(item, t)[0] for t in roll["texts"]], dtype=np.float32)
                    std = float(rews.std())
                    advs = np.zeros_like(rews) if std < 1e-4 else (rews - float(rews.mean())) / (std + 1e-4)
                    for g_ids, adv in zip(roll["gen_ids"], advs.tolist()):
                        if g_ids:
                            packed.append((roll["prompt_ids"], g_ids, adv))
                if not packed:
                    continue
                seq_ids, gen_mask, advs_arr = rl_mod.pack_rollouts(packed, pad_id, args.sft_seq_len)
                old_lp = mx.stop_gradient(sequence_logprobs(model, seq_ids))
                ref_lp = mx.stop_gradient(sequence_logprobs(ref_model, seq_ids))
                mx.eval(old_lp, ref_lp)
                loss, gn = step_rl(seq_ids, gen_mask, advs_arr, old_lp, ref_lp, mx.array(args.lr * 0.5, dtype=mx.float32))
                mx.eval(loss, gn, state)

            gn_val = float(gn)
            l_val = float(loss)
            max_gn = max(max_gn, gn_val)
            if not np.isfinite(l_val) or not np.isfinite(gn_val):
                volatile_reasons.append(f"cycle {cyc} step {step}: NaN/Inf loss={l_val} gnorm={gn_val}")
                break

        row, hard_pool = measure_cycle(cyc, max_gn)
        mf.write(json.dumps(row) + "\n")
        mf.flush()
        best_cycles.append(row)
        log(f"cycle {cyc}/{args.cycles}: {row}")

        # Volatility tripwires
        if row["max_grad_norm"] > 10.0:
            volatile_reasons.append(f"cycle {cyc}: grad norm spike ({row['max_grad_norm']:.2f} > 10.0)")
        if row["kl_from_init"] > 2.5:
            volatile_reasons.append(f"cycle {cyc}: KL drift ({row['kl_from_init']:.3f} > 2.5)")
        if row["old_pretrain_val"] - row0["old_pretrain_val"] > 0.35:
            volatile_reasons.append(
                f"cycle {cyc}: old-text forgetting ({row0['old_pretrain_val']:.3f} -> {row['old_pretrain_val']:.3f})"
            )
        if row["smoke_ok"] < 3:
            volatile_reasons.append(f"cycle {cyc}: core smoke collapse ({row['smoke_pass']})")
        if row["degeneracy_rate"] > 0.25:
            volatile_reasons.append(f"cycle {cyc}: high loop degeneracy ({row['degeneracy_rate']:.2%})")

    mf.close()
    verdict = "VOLATILE" if volatile_reasons else "STABLE"
    report_lines = [
        "=" * 78,
        "CONTINUAL LEARNING & VOLATILITY REPORT",
        f"verdict: {verdict}",
        f"checkpoint: {ckpt_path} ({n_params/1e6:.2f}M params, {config.num_hidden_layers} layers)",
        "=" * 78,
    ]
    for r in best_cycles:
        report_lines.append(
            f"cycle {r['cycle']}: old_val={r['old_pretrain_val']} open_web_val={r['open_web_val']} "
            f"sft_val={r['sft_val']} kl={r['kl_from_init']} max_gn={r['max_grad_norm']} "
            f"smoke={r['smoke_pass']} break={r['break_pass']} degen={r['degeneracy_rate']}"
        )
    if volatile_reasons:
        report_lines.append("")
        report_lines.append("tripwire triggers:")
        report_lines.extend(f"  - {reason}" for reason in volatile_reasons)
    report_txt = "\n".join(report_lines) + "\n"
    report_path = os.path.join(args.artifacts_dir, "volatility_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_txt)

    final_ckpt = os.path.join(args.checkpoints_dir, "continual_final.safetensors")
    model.save_weights_bf16(final_ckpt)
    config.save(os.path.join(args.checkpoints_dir, "config.json"))
    log(f"verdict={verdict}; wrote {report_path} and {final_ckpt}")
    print("\n" + report_txt)


if __name__ == "__main__":
    main()
