#!/usr/bin/env python3
"""02 — Train a byte-level BPE tokenizer and pre-tokenize the corpus.

The standalone Rust ``tokenizers`` package is used; no Hugging Face model or
dataset code is involved.

Outputs
-------
artifacts/tokenizer.json      8192-token byte-level BPE tokenizer
artifacts/tokenizer_meta.json vocab size / special-token ids / stats
data/train_tokens.npy         uint16 token ids for pretraining
data/valid_tokens.npy         uint16 token ids for validation
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Iterator, List

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIAL_TOKENS = ["<|endoftext|>", "<|padding|>", "<|user|>", "<|assistant|>"]


def log(msg: str) -> None:
    print(f"[02 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def iter_text_chunks(path: str, chunk_bytes: int) -> Iterator[str]:
    buf: List[str] = []
    size = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            buf.append(line)
            size += len(line)
            if size >= chunk_bytes:
                yield "".join(buf)
                buf = []
                size = 0
        if buf:
            yield "".join(buf)


def build_tokenizer(vocab_size: int, sample_mb: int, train_path: str) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
        min_frequency=2,
    )
    log(f"training BPE (vocab={vocab_size}) on first {sample_mb} MiB ...")
    t0 = time.time()
    tok.train_from_iterator(
        iter_text_chunks(train_path, 1 << 20),
        trainer=trainer,
        length=min(sample_mb, max(1, os.path.getsize(train_path) >> 20)),
    )
    log(f"tokenizer trained in {time.time() - t0:.1f}s, actual vocab={tok.get_vocab_size()}")
    return tok


def encode_to_npy(
    tok: Tokenizer,
    text_path: str,
    out_path: str,
    max_mb: int,
    batch_chunks: int = 16,
    chunk_bytes: int = 1 << 20,
) -> int:
    """Encode a text file into a uint16 ``.npy`` array."""
    parts: List[np.ndarray] = []
    pending: List[str] = []
    n_tokens = 0
    n_bytes = 0
    t0 = time.time()
    limit = max_mb << 20 if max_mb > 0 else None

    def flush() -> None:
        nonlocal n_tokens
        if not pending:
            return
        encodings = tok.encode_batch(pending, add_special_tokens=False)
        for enc in encodings:
            arr = np.asarray(enc.ids, dtype=np.uint16)
            parts.append(arr)
            n_tokens += arr.size

    with open(text_path, "r", encoding="utf-8", errors="ignore") as f:
        while True:
            block = f.read(chunk_bytes)
            if not block:
                break
            pending.append(block)
            n_bytes += len(block)
            if len(pending) >= batch_chunks:
                flush()
                pending = []
                log(f"  {out_path}: {n_tokens / 1e6:.2f}M tokens ({n_bytes / (1 << 20):.0f} MiB)")
            if limit is not None and n_bytes >= limit:
                break
    flush()

    if not parts:
        raise RuntimeError(f"no tokens produced from {text_path}")
    arr = np.concatenate(parts).astype(np.uint16, copy=False)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.save(out_path, arr)
    dt = max(time.time() - t0, 1e-9)
    log(
        f"wrote {out_path}: {arr.size / 1e6:.2f}M tokens "
        f"({arr.nbytes / (1 << 20):.0f} MiB, {arr.size / dt / 1e3:.1f}k tok/s, "
        f"{n_bytes / max(arr.size, 1):.2f} bytes/token)"
    )
    return int(arr.size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--tokenizer-sample-mb", type=int, default=60)
    parser.add_argument("--max-train-mb", type=int, default=0, help="0 = full file")
    parser.add_argument("--max-valid-mb", type=int, default=0, help="0 = full file")
    args = parser.parse_args()

    os.makedirs(args.artifacts_dir, exist_ok=True)
    train_path = os.path.join(args.data_dir, "pretrain_train.txt")
    valid_path = os.path.join(args.data_dir, "pretrain_valid.txt")
    if not os.path.exists(train_path):
        raise SystemExit(f"missing {train_path}; run 01_fetch_data.py first")

    tok = build_tokenizer(args.vocab_size, args.tokenizer_sample_mb, train_path)
    tokenizer_path = os.path.join(args.artifacts_dir, "tokenizer.json")
    tok.save(tokenizer_path)
    log(f"saved {tokenizer_path}")

    train_tokens = encode_to_npy(
        tok,
        train_path,
        os.path.join(args.data_dir, "train_tokens.npy"),
        max_mb=args.max_train_mb,
    )
    valid_tokens = 0
    if os.path.exists(valid_path):
        valid_tokens = encode_to_npy(
            tok,
            valid_path,
            os.path.join(args.data_dir, "valid_tokens.npy"),
            max_mb=args.max_valid_mb,
        )

    meta = {
        "vocab_size": tok.get_vocab_size(),
        "special_tokens": {name: tok.token_to_id(name) for name in SPECIAL_TOKENS},
        "train_tokens": train_tokens,
        "valid_tokens": valid_tokens,
        "train_bytes": os.path.getsize(train_path),
        "bytes_per_token": os.path.getsize(train_path) / max(train_tokens, 1),
    }
    with open(os.path.join(args.artifacts_dir, "tokenizer_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log(f"meta: {json.dumps(meta, indent=2)}")
    log("done.")


if __name__ == "__main__":
    main()
