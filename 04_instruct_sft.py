#!/usr/bin/env python3
"""04 — Instruction-tune the pretrained checkpoint with prompt-masked loss.

The model is trained on ``<|user|> ... <|assistant|> ... <|endoftext|>``
examples.  Cross entropy is computed only on the assistant response (and the
terminating ``<|endoftext|>``); prompt tokens and padding are labelled -100.

Outputs
-------
checkpoints/instruct/instruct_final.safetensors
checkpoints/instruct/config.json
artifacts/sft_metrics.jsonl
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
from tokenizers import Tokenizer

from model import PythiaConfig, build_model, load_model_from_checkpoint, _masked_cross_entropy


def log(msg: str) -> None:
    print(f"[04 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def load_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_dataset(
    rows: List[dict],
    tok: Tokenizer,
    seq_len: int,
) -> List[Dict[str, np.ndarray]]:
    pad_id = tok.token_to_id("<|padding|>")
    data: List[Dict[str, np.ndarray]] = []
    skipped = 0
    for row in rows:
        prompt_ids = tok.encode(row["prompt"], add_special_tokens=False).ids
        resp_ids = tok.encode(row["response"], add_special_tokens=False).ids
        if len(resp_ids) < 2:
            skipped += 1
            continue
        if len(prompt_ids) >= seq_len:
            prompt_ids = prompt_ids[: seq_len - 1]
        resp_ids = resp_ids[: seq_len - len(prompt_ids)]
        if not resp_ids:
            skipped += 1
            continue
        input_ids = prompt_ids + resp_ids
        labels = [-100] * len(prompt_ids) + resp_ids
        pad = seq_len - len(input_ids)
        if pad > 0:
            input_ids = input_ids + [pad_id] * pad
            labels = labels + [-100] * pad
        data.append(
            {
                "input_ids": np.asarray(input_ids, dtype=np.int32),
                "labels": np.asarray(labels, dtype=np.int32),
            }
        )
    log(f"tokenized {len(data):,} examples (skipped {skipped})")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--checkpoints-dir", default="checkpoints/instruct")
    parser.add_argument("--base-checkpoint", default="checkpoints/pretrain/base_final.safetensors")
    parser.add_argument("--config", default="checkpoints/pretrain/config.json")
    parser.add_argument("--minutes", type=float, default=6.0)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-frac", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-steps", type=int, default=0, help="override calibration (smoke tests)")
    args = parser.parse_args()

    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    tok = Tokenizer.from_file(os.path.join(args.artifacts_dir, "tokenizer.json"))
    config = PythiaConfig.load(args.config)
    model = build_model(config, dtype="float32")
    model.load_weights_from(args.base_checkpoint)
    # Checkpoints are stored as bf16.  Cast back to fp32 before creating the
    # optimizer: tiny bf16 AdamW updates round to zero and silently freeze SFT.
    model.set_dtype(mx.float32)
    mx.eval(model.parameters())
    log(f"loaded base checkpoint {args.base_checkpoint} (fp32 training)")

    train_rows = load_jsonl(os.path.join(args.data_dir, "instruct_train.jsonl"))
    valid_rows = load_jsonl(os.path.join(args.data_dir, "instruct_valid.jsonl"))
    train_data = build_dataset(train_rows, tok, args.seq_len)
    valid_data = build_dataset(valid_rows, tok, args.seq_len)
    if not train_data:
        raise SystemExit("no training examples")

    optimizer = optim.AdamW(
        learning_rate=args.lr,
        betas=[0.9, 0.95],
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    def loss_fn(model, x, labels):
        logits = model(x)
        # Predict token t from positions < t; mask prompt/padding labels.
        return _masked_cross_entropy(logits[:, :-1, :], labels[:, 1:], ignore_index=-100)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def train_step(x, labels, lr):
        optimizer.learning_rate = lr
        loss, grads = loss_and_grad(model, x, labels)
        grads = clip_grads(grads, args.grad_clip)
        optimizer.update(model, grads)
        return loss

    def get_batch(data: List[Dict[str, np.ndarray]], batch_size: int, indices: Optional[np.ndarray] = None):
        if indices is None:
            indices = rng.integers(0, len(data), size=batch_size)
        x = np.stack([data[int(i)]["input_ids"] for i in indices])
        y = np.stack([data[int(i)]["labels"] for i in indices])
        return mx.array(x), mx.array(y)

    def evaluate() -> float:
        if not valid_data:
            return float("nan")
        idx = np.linspace(0, len(valid_data) - 1, min(args.val_batches * args.batch_size, len(valid_data))).astype(np.int64)
        losses = []
        for start in range(0, len(idx), args.batch_size):
            batch_idx = idx[start : start + args.batch_size]
            if len(batch_idx) == 0:
                continue
            x, y = get_batch(valid_data, len(batch_idx), batch_idx)
            loss = loss_fn(model, x, y)
            mx.eval(loss)
            losses.append(float(loss))
        return float(sum(losses) / max(len(losses), 1))

    metrics_path = os.path.join(args.artifacts_dir, "sft_metrics.jsonl")
    metrics_file = open(metrics_path, "w", encoding="utf-8")

    def log_metrics(row: Dict) -> None:
        metrics_file.write(json.dumps(row) + "\n")
        metrics_file.flush()

    # Calibrate on the real SFT workload.
    bench_steps = 30 if args.max_steps == 0 else min(args.max_steps, 30)
    log(f"calibrating with {bench_steps} steps ...")
    t_bench = time.time()
    loss = mx.array(0.0)
    for _ in range(bench_steps):
        x, y = get_batch(train_data, args.batch_size)
        loss = train_step(x, y, mx.array(args.lr * 0.5, dtype=mx.float32))
        mx.eval(loss, state)
    bench_dt = max(time.time() - t_bench, 1e-6)
    tok_per_sec = bench_steps * args.batch_size * args.seq_len / bench_dt

    if args.max_steps:
        total_steps = args.max_steps
    else:
        total_steps = max(bench_steps, int(args.minutes * 60.0 / bench_dt * bench_steps))
        total_steps = min(total_steps, 20000)
    warmup = max(5, int(args.warmup_frac * total_steps))
    log(
        f"calibration: {bench_dt/bench_steps*1000:.0f} ms/step, {tok_per_sec:,.0f} tok/s "
        f"-> total_steps={total_steps:,}"
    )

    t_start = time.time() - bench_dt
    tokens_processed = bench_steps * args.batch_size * args.seq_len
    for step in range(bench_steps, total_steps):
        lr = lr_at(step, total_steps, args.lr, warmup, args.min_lr_ratio)
        x, y = get_batch(train_data, args.batch_size)
        loss = train_step(x, y, mx.array(lr, dtype=mx.float32))
        mx.eval(loss, state)
        tokens_processed += args.batch_size * args.seq_len
        done = step + 1
        if done % args.log_every == 0 or done == total_steps:
            val_loss = None
            if args.val_every > 0 and (done % args.val_every == 0 or done == total_steps):
                val_loss = evaluate()
            row = {
                "step": done,
                "loss": float(loss),
                "lr": lr,
                "tok_per_sec": tokens_processed / max(time.time() - t_start, 1e-6),
                "tokens": tokens_processed,
                "elapsed_sec": time.time() - t_start,
            }
            if val_loss is not None:
                row["val_loss"] = val_loss
            log_metrics(row)
            log(
                f"step {done:5d}/{total_steps} loss {float(loss):.4f}"
                + (f" val {val_loss:.4f}" if val_loss is not None else "")
                + f" lr {lr:.2e}"
            )

    final_path = os.path.join(args.checkpoints_dir, "instruct_final.safetensors")
    model.save_weights_bf16(final_path)
    config.save(os.path.join(args.checkpoints_dir, "config.json"))
    final_val = evaluate()
    log_metrics(
        {
            "step": total_steps,
            "loss": float(loss),
            "val_loss": final_val,
            "tokens": tokens_processed,
            "elapsed_sec": time.time() - t_start,
            "final": True,
        }
    )
    metrics_file.close()
    log(f"done: {total_steps:,} steps, {tokens_processed/1e6:.1f}M tokens, final val loss {final_val:.4f}")
    log(f"saved {final_path}")
    log(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
