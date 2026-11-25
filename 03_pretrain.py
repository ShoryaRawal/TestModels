#!/usr/bin/env python3
"""03 — Pretrain the small Pythia / GPT-NeoX model from scratch.

Features
--------
* GPT-NeoX architecture from ``model.py`` (parallel residual, partial RoPE,
  untied embeddings).
* ``mx.compile`` fused training step with global gradient clipping.
* AdamW + linear warmup + cosine decay.
* Throughput calibration: after 50 real steps the step count is chosen so the
  run fills the requested wall-clock budget.
* Pythia-style log-spaced checkpoints (powers of two up to 512) plus linear
  checkpoints every 1000 steps.

Outputs
-------
checkpoints/pretrain/step_XXXX.safetensors
checkpoints/pretrain/base_final.safetensors
checkpoints/pretrain/config.json
artifacts/pretrain_metrics.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from functools import partial
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mlx_utils
import numpy as np

from model import PythiaConfig, build_model


def log(msg: str) -> None:
    print(f"[03 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def lr_at(step: int, total_steps: int, peak_lr: float, warmup: int, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return peak_lr * (step + 1) / max(warmup, 1)
    progress = min(1.0, (step - warmup) / max(total_steps - warmup, 1))
    min_lr = peak_lr * min_ratio
    return min_lr + 0.5 * (peak_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def clip_grads(grads, max_norm: float):
    leaves = [g for _, g in mlx_utils.tree_flatten(grads)]
    if not leaves:
        return grads
    norm_sq = sum(mx.sum(g.astype(mx.float32) ** 2) for g in leaves)
    norm = mx.sqrt(norm_sq)
    scale = mx.minimum(mx.array(1.0, dtype=mx.float32), mx.array(max_norm, dtype=mx.float32) / (norm + 1e-6))
    return mlx_utils.tree_map(lambda g: g * scale.astype(g.dtype), grads)


def checkpoint_steps(total_steps: int) -> List[int]:
    """Pythia-style log-spaced + linear checkpoint schedule."""
    steps = {0}
    for i in range(0, 20):
        s = 1 << i
        if s <= total_steps:
            steps.add(s)
    for s in range(1000, total_steps + 1, 1000):
        steps.add(s)
    # A few extra evenly-spaced snapshots for developmental analysis.
    for frac in (4, 2, 3):
        s = (total_steps * frac) // 4
        if 0 < s <= total_steps:
            steps.add(s)
    steps.add(total_steps)
    return sorted(steps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--checkpoints-dir", default="checkpoints/pretrain")
    parser.add_argument("--minutes", type=float, default=25.0)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-frac", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-steps", type=int, default=0, help="override calibration (smoke tests)")
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--intermediate-size", type=int, default=1536)
    args = parser.parse_args()

    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)

    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    with open(os.path.join(args.artifacts_dir, "tokenizer_meta.json"), "r", encoding="utf-8") as f:
        tok_meta = json.load(f)
    vocab_size = int(tok_meta["vocab_size"])

    train_tokens = np.load(os.path.join(args.data_dir, "train_tokens.npy"), mmap_mode="r")
    valid_tokens = np.load(os.path.join(args.data_dir, "valid_tokens.npy"), mmap_mode="r")
    log(f"tokens: train={len(train_tokens):,} valid={len(valid_tokens):,} vocab={vocab_size}")

    config = PythiaConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        max_seq_len=512,
        dtype="float32",
    )
    model = build_model(config, dtype="float32")
    mx.eval(model.parameters())
    n_params = sum(v.size for _, v in mlx_utils.tree_flatten(model.parameters()))
    log(
        f"model: {n_params/1e6:.2f}M params, hidden={config.hidden_size}, "
        f"layers={config.num_hidden_layers}, heads={config.num_attention_heads}, "
        f"head_dim={config.head_dim}, rotary_dim={config.rotary_dim}"
    )

    config.save(os.path.join(args.checkpoints_dir, "config.json"))
    model.save_weights_bf16(os.path.join(args.checkpoints_dir, "step_0000.safetensors"))
    log("saved step_0000 (random init)")

    optimizer = optim.AdamW(
        learning_rate=args.lr,
        betas=[0.9, 0.95],
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    def loss_fn(model, x, y):
        logits = model(x)
        return mx.mean(nn.losses.cross_entropy(logits, y))

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(x, y, lr):
        optimizer.learning_rate = lr
        loss, grads = loss_and_grad(model, x, y)
        grads = clip_grads(grads, args.grad_clip)
        optimizer.update(model, grads)
        return loss

    def get_batch(tokens, batch_size: int, seq_len: int) -> Tuple[mx.array, mx.array]:
        max_start = len(tokens) - seq_len - 1
        starts = rng.integers(0, max_start, size=batch_size)
        x = np.stack([np.asarray(tokens[s : s + seq_len]) for s in starts]).astype(np.int32)
        y = np.stack([np.asarray(tokens[s + 1 : s + seq_len + 1]) for s in starts]).astype(np.int32)
        return mx.array(x), mx.array(y)

    # Fixed validation windows so the metric is comparable across steps.
    max_start = len(valid_tokens) - args.seq_len - 1
    val_starts = np.linspace(0, max_start, args.val_batches * args.batch_size).astype(np.int64).reshape(
        args.val_batches, args.batch_size
    )

    def evaluate() -> float:
        losses = []
        for starts in val_starts:
            x = np.stack([np.asarray(valid_tokens[s : s + args.seq_len]) for s in starts]).astype(np.int32)
            y = np.stack([np.asarray(valid_tokens[s + 1 : s + args.seq_len + 1]) for s in starts]).astype(np.int32)
            loss = mx.mean(nn.losses.cross_entropy(model(mx.array(x)), mx.array(y)))
            mx.eval(loss)
            losses.append(float(loss))
        return float(sum(losses) / len(losses))

    metrics_path = os.path.join(args.artifacts_dir, "pretrain_metrics.jsonl")
    metrics_file = open(metrics_path, "w", encoding="utf-8")

    def log_metrics(row: Dict) -> None:
        metrics_file.write(json.dumps(row) + "\n")
        metrics_file.flush()

    # ------------------------------------------------------------------
    # Throughput calibration: 50 real steps.
    # ------------------------------------------------------------------
    bench_steps = 50 if args.max_steps == 0 else min(args.max_steps, 50)
    log(f"calibrating with {bench_steps} steps ...")
    t_bench = time.time()
    last_loss = float("nan")
    for step in range(bench_steps):
        x, y = get_batch(train_tokens, args.batch_size, args.seq_len)
        lr = args.lr * (step + 1) / max(bench_steps, 1)
        loss = train_step(x, y, mx.array(lr, dtype=mx.float32))
        mx.eval(loss, state)
        last_loss = float(loss)
        # Preserve the early Pythia log-spaced snapshots (steps 1, 2, 4, ...)
        # that would otherwise be consumed by the calibration phase.
        done = step + 1
        if done & (done - 1) == 0:
            path = os.path.join(args.checkpoints_dir, f"step_{done:04d}.safetensors")
            model.save_weights_bf16(path)
            log(f"  checkpoint {os.path.basename(path)} (calibration)")
    bench_dt = max(time.time() - t_bench, 1e-6)
    tok_per_sec = bench_steps * args.batch_size * args.seq_len / bench_dt

    if args.max_steps:
        total_steps = args.max_steps
    else:
        total_steps = max(bench_steps, int(args.minutes * 60.0 / bench_dt * bench_steps))
        total_steps = min(total_steps, 40000)
    warmup = max(10, int(args.warmup_frac * total_steps))
    log(
        f"calibration: {bench_dt/bench_steps*1000:.0f} ms/step, {tok_per_sec:,.0f} tok/s "
        f"-> total_steps={total_steps:,} (warmup={warmup})"
    )

    ckpt_schedule = set(checkpoint_steps(total_steps))
    log(f"checkpoint schedule: {sorted(ckpt_schedule)}")

    start_step = bench_steps
    tokens_processed = start_step * args.batch_size * args.seq_len
    t_start = time.time() - bench_dt
    last_log_t = time.time()
    last_log_tokens = tokens_processed
    best_val = float("inf")

    for step in range(start_step, total_steps):
        lr = lr_at(step, total_steps, args.lr, warmup, args.min_lr_ratio)
        x, y = get_batch(train_tokens, args.batch_size, args.seq_len)
        loss = train_step(x, y, mx.array(lr, dtype=mx.float32))
        mx.eval(loss, state)
        tokens_processed += args.batch_size * args.seq_len

        done_step = step + 1
        if done_step % args.log_every == 0 or done_step == total_steps:
            now = time.time()
            interval_tok = tokens_processed - last_log_tokens
            interval_dt = max(now - last_log_t, 1e-6)
            cum_tok_s = tokens_processed / max(now - t_start, 1e-6)
            row = {
                "step": done_step,
                "loss": float(loss),
                "lr": lr,
                "tok_per_sec": interval_tok / interval_dt,
                "tok_per_sec_cum": cum_tok_s,
                "tokens": tokens_processed,
                "elapsed_sec": now - t_start,
            }
            val_loss = None
            if args.val_every > 0 and (done_step % args.val_every == 0 or done_step == total_steps):
                val_loss = evaluate()
                row["val_loss"] = val_loss
                best_val = min(best_val, val_loss)
            log_metrics(row)
            log(
                f"step {done_step:5d}/{total_steps} loss {float(loss):.4f}"
                + (f" val {val_loss:.4f}" if val_loss is not None else "")
                + f" lr {lr:.2e} {cum_tok_s:,.0f} tok/s"
            )
            last_log_t = now
            last_log_tokens = tokens_processed

        if done_step in ckpt_schedule:
            path = os.path.join(args.checkpoints_dir, f"step_{done_step:04d}.safetensors")
            model.save_weights_bf16(path)
            log(f"  checkpoint {os.path.basename(path)}")

    final_path = os.path.join(args.checkpoints_dir, "base_final.safetensors")
    model.save_weights_bf16(final_path)
    config.save(os.path.join(args.checkpoints_dir, "config.json"))
    final_val = evaluate()
    log_metrics(
        {
            "step": total_steps,
            "loss": float(loss),
            "val_loss": final_val,
            "lr": lr_at(total_steps, total_steps, args.lr, warmup, args.min_lr_ratio),
            "tokens": tokens_processed,
            "elapsed_sec": time.time() - t_start,
            "final": True,
        }
    )
    metrics_file.close()

    epochs = tokens_processed / max(len(train_tokens), 1)
    log(
        f"done: {total_steps:,} steps, {tokens_processed/1e6:.1f}M tokens, "
        f"{epochs:.2f} epochs, final val loss {final_val:.4f}"
    )
    log(f"saved {final_path}")
    log(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
