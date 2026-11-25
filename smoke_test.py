#!/usr/bin/env python3
"""Fast 5-step smoke test for the architecture, training step and utilities.

This does not touch the real data or checkpoints.  Run it before a long job:

    .venv/bin/python smoke_test.py
"""

from __future__ import annotations

import os
import tempfile
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from model import (
    PythiaConfig,
    build_model,
    generate,
    load_model_from_checkpoint,
    _masked_cross_entropy,
)

SPECIALS = ["<|endoftext|>", "<|padding|>", "<|user|>", "<|assistant|>"]


def main() -> None:
    mx.random.seed(0)
    np.random.seed(0)
    cfg = PythiaConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_seq_len=64,
        dtype="float32",
    )
    model = build_model(cfg)
    print(f"model: {cfg.num_parameters/1e6:.3f}M params, rotary_dim={cfg.rotary_dim}")

    B, T = 4, 16
    x = mx.random.randint(0, cfg.vocab_size, (B, T), dtype=mx.int32)
    y = mx.random.randint(0, cfg.vocab_size, (B, T), dtype=mx.int32)

    # Transparent forward.
    logits, cache = model(x, return_cache=True)
    assert logits.shape == (B, T, cfg.vocab_size)
    assert len(cache["layers"]) == cfg.num_hidden_layers
    for layer in cache["layers"]:
        assert layer["attn_probs"].shape == (B, cfg.num_attention_heads, T, T)
        assert layer["resid_pre"].shape == (B, T, cfg.hidden_size)
        assert layer["resid_post"].shape == (B, T, cfg.hidden_size)
        assert layer["mlp_out"].shape == (B, T, cfg.hidden_size)
    lens = model.logit_lens(x)
    assert len(lens) == cfg.num_hidden_layers + 1
    print("transparent forward + logit lens OK")

    # Ablation.
    base = model(x)
    zero = model(x, ablate_heads={0: [0]}, ablation_mode="zero")
    mean = model(x, ablate_heads={0: [0]}, ablation_mode="mean")
    assert float(mx.max(mx.abs(base - zero))) > 0.0
    assert float(mx.max(mx.abs(base - mean))) > 0.0
    print("zero/mean head ablation OK")

    # Five compiled optimizer steps.
    optimizer = optim.AdamW(learning_rate=1e-3)
    def loss_fn(model, x, y):
        return mx.mean(nn.losses.cross_entropy(model(x), y))
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x, y, lr):
        optimizer.learning_rate = lr
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        return loss

    losses = []
    for i in range(5):
        loss = step(x, y, mx.array(1e-3, dtype=mx.float32))
        mx.eval(loss, state)
        losses.append(float(loss))
    assert all(np.isfinite(losses)), losses
    print(f"5 compiled steps OK: loss {losses[0]:.4f} -> {losses[-1]:.4f}")

    # Masked cross entropy.
    labels = mx.array(np.random.randint(-100, cfg.vocab_size, (B, T)), dtype=mx.int32)
    mce = _masked_cross_entropy(model(x), labels)
    mx.eval(mce)
    assert np.isfinite(float(mce))
    print(f"masked cross entropy OK: {float(mce):.4f}")

    # Safetensors round trip (bf16 on disk, fp32 in memory).
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "w.safetensors")
        cfg_path = os.path.join(d, "config.json")
        model.save_weights_bf16(path)
        cfg.save(cfg_path)
        loaded, _ = load_model_from_checkpoint(path, config_path=cfg_path, dtype="float32")
        assert loaded.embed_out.weight.dtype == mx.float32
        diff = float(mx.max(mx.abs(model.embed_out.weight - loaded.embed_out.weight)))
        assert diff < 0.05, diff
        print(f"safetensors round trip OK (max diff {diff:.4f})")

    # Tiny tokenizer + generation.
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        ["once upon a time there was a little cat. " * 200] * 2,
        trainers.BpeTrainer(
            vocab_size=128,
            special_tokens=SPECIALS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    text = generate(model, tok, "<|user|>\nhello\n<|assistant|>\n", max_new_tokens=5, seed=1)
    assert isinstance(text, str)
    print(f"tokenizer + KV-cached generation OK: {text!r}")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
