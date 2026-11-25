"""A from-scratch GPT-NeoX / Pythia implementation in Apple MLX.

This module deliberately mirrors the architecture used by EleutherAI's Pythia
models (GPT-NeoX):

  * parallel attention / MLP residual branch
  * partial rotary position embeddings (default 25% of each head)
  * untied input/output embeddings
  * LayerNorm with bias, bias-free linear projections
  * exact GELU

It also implements a "transparent" forward mode used by the interpretability
lab.  When ``return_cache=True`` every block returns its residual stream before
and after the block, the explicit causal attention probabilities, the attention
output and the MLP output.  Attention is always computed with an explicit
``q @ k.T * scale + causal_mask -> softmax`` path so the probabilities are
available rather than hidden inside fused SDPA.

No Hugging Face code is used anywhere.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.utils as mlx_utils
import numpy as np

DTYPE_MAP = {
    "float32": mx.float32,
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
}


@dataclass
class PythiaConfig:
    """Configuration for a small Pythia / GPT-NeoX model."""

    vocab_size: int = 8192
    hidden_size: int = 384
    num_hidden_layers: int = 6
    num_attention_heads: int = 6
    intermediate_size: int = 1536
    max_seq_len: int = 512
    rotary_pct: float = 0.25
    rotary_base: float = 10000.0
    layer_norm_eps: float = 1e-5
    hidden_act: str = "gelu"
    use_parallel_residual: bool = True
    tie_word_embeddings: bool = False
    initializer_range: float = 0.02
    dtype: str = "float32"

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        return self.hidden_size // self.num_attention_heads

    @property
    def rotary_dim(self) -> int:
        # GPT-NeoX rotates int(head_dim * rotary_pct) dimensions and requires
        # an even number of rotated dimensions.
        d = int(self.head_dim * self.rotary_pct)
        if d % 2 != 0:
            d -= 1
        return max(2, d)

    @property
    def num_parameters(self) -> int:
        # Parameter count for documentation / logging (biases in LayerNorm
        # included, linear biases absent).
        h = self.hidden_size
        i = self.intermediate_size
        v = self.vocab_size
        per_layer = (
            4 * h * h  # q, k, v, dense
            + 2 * h * i  # mlp up/down
            + 4 * h  # two LayerNorms (weight + bias)
        )
        return v * h + self.num_hidden_layers * per_layer + 2 * h + v * h

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PythiaConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PythiaConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def _masked_cross_entropy(logits: mx.array, targets: mx.array, ignore_index: int = -100) -> mx.array:
    """Cross entropy that ignores ``ignore_index`` targets.

    ``logits`` has shape (..., vocab) and ``targets`` has shape (...).
    """
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    flat_log_probs = log_probs.reshape(-1, log_probs.shape[-1])
    flat_targets = targets.reshape(-1).astype(mx.int32)
    valid = flat_targets != ignore_index
    safe = mx.where(valid, flat_targets, mx.zeros_like(flat_targets))
    picked = mx.take_along_axis(flat_log_probs, safe[:, None], axis=-1).squeeze(-1)
    token_loss = -mx.where(valid, picked, mx.zeros_like(picked))
    denom = mx.maximum(valid.astype(mx.float32).sum(), mx.array(1.0))
    return token_loss.astype(mx.float32).sum() / denom


class NeoXAttention(nn.Module):
    """Multi-head causal attention with partial RoPE."""

    def __init__(self, config: PythiaConfig):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        self.scale = self.head_dim ** -0.5

        # GPT-NeoX uses bias-free linear projections.
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    # ------------------------------------------------------------------
    # Partial rotary embeddings
    # ------------------------------------------------------------------
    def _rope_tables(self, seq_len: int, offset: int, dtype: mx.Dtype) -> Tuple[mx.array, mx.array]:
        half = self.rotary_dim // 2
        inv_freq = 1.0 / (
            self.config.rotary_base
            ** (mx.arange(0, half, dtype=mx.float32) / float(half))
        )
        positions = mx.arange(offset, offset + seq_len, dtype=mx.float32)
        freqs = mx.outer(positions, inv_freq)  # (T, rotary_dim/2)
        cos = mx.concatenate([mx.cos(freqs), mx.cos(freqs)], axis=-1)
        sin = mx.concatenate([mx.sin(freqs), mx.sin(freqs)], axis=-1)
        return cos.astype(dtype), sin.astype(dtype)

    def _apply_rope(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        # x: (B, H, T, head_dim)
        x_rot = x[..., : self.rotary_dim]
        x_pass = x[..., self.rotary_dim :]
        half = self.rotary_dim // 2
        x1 = x_rot[..., :half]
        x2 = x_rot[..., half:]
        rotated = mx.concatenate([-x2, x1], axis=-1)
        out = x_rot * cos + rotated * sin
        if x_pass.shape[-1]:
            out = mx.concatenate([out, x_pass], axis=-1)
        return out

    # ------------------------------------------------------------------
    # Head ablation
    # ------------------------------------------------------------------
    def _apply_head_ablation(
        self,
        head_out: mx.array,
        ablate_heads: Optional[Sequence[int]],
        mode: str = "zero",
    ) -> mx.array:
        if not ablate_heads:
            return head_out
        mask = np.zeros((1, self.num_heads, 1, 1), dtype=np.float32)
        for h in ablate_heads:
            if 0 <= int(h) < self.num_heads:
                mask[0, int(h), 0, 0] = 1.0
        mask = mx.array(mask, dtype=head_out.dtype)
        if mode == "zero":
            return head_out * (1.0 - mask)
        if mode == "mean":
            mean = mx.mean(head_out, axis=2, keepdims=True)
            return head_out * (1.0 - mask) + mean * mask
        raise ValueError(f"unknown ablation mode: {mode}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def __call__(
        self,
        x: mx.array,
        past_kv: Optional[Dict[str, mx.array]] = None,
        use_kv_cache: bool = False,
        ablate_heads: Optional[Sequence[int]] = None,
        ablation_mode: str = "zero",
    ):
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        offset = past_kv["k"].shape[2] if past_kv is not None else 0
        cos, sin = self._rope_tables(T, offset, x.dtype)
        q = self._apply_rope(q, cos, sin)
        k = self._apply_rope(k, cos, sin)

        if past_kv is not None:
            k = mx.concatenate([past_kv["k"], k], axis=2)
            v = mx.concatenate([past_kv["v"], v], axis=2)

        new_kv = {"k": k, "v": v} if use_kv_cache else None

        # Explicit attention: (B, H, T, S) probabilities.
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        total_len = k.shape[2]
        if T > 1:
            q_pos = mx.arange(offset, offset + T)[:, None]
            k_pos = mx.arange(0, total_len)[None, :]
            causal_mask = mx.where(k_pos <= q_pos, 0.0, -1e9).astype(attn.dtype)
            attn = attn + causal_mask[None, None, :, :]
        probs = mx.softmax(attn, axis=-1)

        head_out = probs @ v  # (B, H, T, head_dim)
        head_out = self._apply_head_ablation(head_out, ablate_heads, ablation_mode)

        out = head_out.transpose(0, 2, 1, 3).reshape(B, T, self.num_heads * self.head_dim)
        out = self.dense(out)

        cache = {
            "attn_probs": probs,
            "attn_heads": head_out,
            "attn_out": out,
        }
        return out, cache, new_kv


class NeoXMLP(nn.Module):
    def __init__(self, config: PythiaConfig):
        super().__init__()
        self.config = config
        self.dense_h_to_4h = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dense_4h_to_h = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.dense_h_to_4h(x)
        if self.config.hidden_act == "gelu":
            h = nn.gelu(h)
        elif self.config.hidden_act == "gelu_new":
            h = nn.gelu_approx(h)
        elif self.config.hidden_act == "relu":
            h = nn.relu(h)
        else:
            raise ValueError(f"unsupported activation: {self.config.hidden_act}")
        return self.dense_4h_to_h(h)


class NeoXBlock(nn.Module):
    """A GPT-NeoX block with parallel attention and MLP residuals."""

    def __init__(self, config: PythiaConfig):
        super().__init__()
        self.config = config
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = NeoXAttention(config)
        self.mlp = NeoXMLP(config)

    def __call__(
        self,
        x: mx.array,
        past_kv: Optional[Dict[str, mx.array]] = None,
        use_kv_cache: bool = False,
        ablate_heads: Optional[Sequence[int]] = None,
        ablation_mode: str = "zero",
    ):
        resid_pre = x
        attn_out, attn_cache, new_kv = self.attention(
            self.input_layernorm(x),
            past_kv=past_kv,
            use_kv_cache=use_kv_cache,
            ablate_heads=ablate_heads,
            ablation_mode=ablation_mode,
        )

        if self.config.use_parallel_residual:
            # The defining GPT-NeoX residual:
            # x = x + attn(ln1(x)) + mlp(ln2(x))
            mlp_out = self.mlp(self.post_attention_layernorm(x))
            x = x + attn_out + mlp_out
        else:
            x = x + attn_out
            mlp_out = self.mlp(self.post_attention_layernorm(x))
            x = x + mlp_out

        cache = {
            "resid_pre": resid_pre,
            "resid_post": x,
            "mlp_out": mlp_out,
            **attn_cache,
        }
        return x, cache, new_kv


class PythiaForCausalLM(nn.Module):
    """Small GPT-NeoX causal language model with transparent internals."""

    def __init__(self, config: PythiaConfig):
        super().__init__()
        self.config = config
        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [NeoXBlock(config) for _ in range(config.num_hidden_layers)]
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Untied output embedding, bias-free for clean Logit Lens.
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.embed_out.weight = self.embed_in.weight
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight = mx.random.normal(
                    module.weight.shape,
                    scale=self.config.initializer_range,
                    dtype=module.weight.dtype,
                )
            elif isinstance(module, nn.Embedding):
                module.weight = mx.random.normal(
                    module.weight.shape,
                    scale=self.config.initializer_range,
                    dtype=module.weight.dtype,
                )
            elif isinstance(module, nn.LayerNorm):
                module.weight = mx.ones(module.weight.shape, dtype=module.weight.dtype)
                if module.bias is not None:
                    module.bias = mx.zeros(module.bias.shape, dtype=module.bias.dtype)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def __call__(
        self,
        input_ids: mx.array,
        return_cache: bool = False,
        ablate_heads: Optional[Dict[int, Sequence[int]]] = None,
        ablation_mode: str = "zero",
        past_kv: Optional[List[Dict[str, mx.array]]] = None,
        use_kv_cache: bool = False,
    ):
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]
        x = self.embed_in(input_ids)

        layer_caches: List[Dict[str, mx.array]] = []
        new_past: List[Dict[str, mx.array]] = []

        for idx, layer in enumerate(self.layers):
            heads = None
            if ablate_heads is not None:
                heads = ablate_heads.get(idx, None)
            layer_past = past_kv[idx] if past_kv is not None else None
            x, cache, kv = layer(
                x,
                past_kv=layer_past,
                use_kv_cache=use_kv_cache,
                ablate_heads=heads,
                ablation_mode=ablation_mode,
            )
            layer_caches.append(cache)
            new_past.append(kv)

        x = self.final_layer_norm(x)
        logits = self.embed_out(x)

        if use_kv_cache:
            return logits, new_past
        if return_cache:
            return logits, {
                "layers": layer_caches,
                "resid_final": x,
                "logits": logits,
            }
        return logits

    # ------------------------------------------------------------------
    # Logit Lens helper
    # ------------------------------------------------------------------
    def logit_lens(self, input_ids: mx.array) -> List[mx.array]:
        """Return next-token logits from every layer's residual stream.

        The returned list has one array per layer plus the final normalized
        residual, each of shape (batch, seq, vocab).
        """
        _, cache = self(input_ids, return_cache=True)
        outs = []
        for layer_cache in cache["layers"]:
            resid = layer_cache["resid_post"]
            outs.append(self.embed_out(self.final_layer_norm(resid)))
        outs.append(self.embed_out(cache["resid_final"]))
        return outs

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_weights_bf16(self, path: str) -> None:
        """Save a flattened, bf16 copy of the weights to safetensors."""
        flat = dict(mlx_utils.tree_flatten(self.parameters()))
        flat = {k: v.astype(mx.bfloat16) for k, v in flat.items()}
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        mx.save_safetensors(path, flat)

    def save_weights_fp32(self, path: str) -> None:
        flat = dict(mlx_utils.tree_flatten(self.parameters()))
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        mx.save_safetensors(path, flat)

    def load_weights_from(self, path: str) -> None:
        weights = mx.load(path)
        self.load_weights(list(weights.items()))
        mx.eval(self.parameters())


# ----------------------------------------------------------------------
# Construction / loading helpers
# ----------------------------------------------------------------------
def build_model(config: PythiaConfig, dtype: Optional[str] = None) -> PythiaForCausalLM:
    model = PythiaForCausalLM(config)
    dt = dtype or config.dtype
    if dt != "float32":
        model.set_dtype(DTYPE_MAP[dt])
    return model


def load_model_from_checkpoint(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    dtype: Optional[str] = None,
) -> Tuple[PythiaForCausalLM, PythiaConfig]:
    """Load a model and its config from a checkpoint directory or file."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(checkpoint_path)), "config.json")
    config = PythiaConfig.load(config_path)
    model = build_model(config, dtype=dtype or "float32")
    model.load_weights_from(checkpoint_path)
    # Saved checkpoints are bf16; cast to the requested compute dtype.
    model.set_dtype(DTYPE_MAP[dtype or "float32"])
    return model, config


# ----------------------------------------------------------------------
# Sampling / generation
# ----------------------------------------------------------------------
def generate(
    model: PythiaForCausalLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    ablate_heads: Optional[Dict[int, Sequence[int]]] = None,
    seed: Optional[int] = None,
    repetition_penalty: float = 1.0,
) -> str:
    """Simple KV-cached sampler for the interactive instruction model."""
    if seed is not None:
        mx.random.seed(seed)

    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    eos_id = tokenizer.token_to_id("<|endoftext|>")
    if not ids:
        ids = [eos_id]

    logits, past = model(
        mx.array([ids], dtype=mx.int32),
        use_kv_cache=True,
        ablate_heads=ablate_heads,
    )
    mx.eval(logits)

    out_ids: List[int] = []
    recent: List[int] = []
    for _ in range(max_new_tokens):
        step_logits = logits[0, -1, :].astype(mx.float32)

        if repetition_penalty and repetition_penalty != 1.0 and recent:
            penalty_ids = mx.array(sorted(set(recent[-64:])), dtype=mx.int32)
            selected = step_logits[penalty_ids]
            penalized = mx.where(selected > 0, selected / repetition_penalty, selected * repetition_penalty)
            step_logits = step_logits.at[penalty_ids].add(penalized - selected)

        if temperature <= 1e-5:
            next_id = mx.argmax(step_logits)
        else:
            step_logits = step_logits / temperature
            if top_k and top_k > 0:
                k = min(int(top_k), step_logits.shape[-1])
                threshold = mx.topk(step_logits, k)[-1]
                step_logits = mx.where(step_logits < threshold, -1e9, step_logits)
            next_id = mx.random.categorical(step_logits)

        mx.eval(next_id)
        nid = int(next_id)
        if nid == eos_id:
            break
        out_ids.append(nid)
        recent.append(nid)
        logits, past = model(
            mx.array([[nid]], dtype=mx.int32),
            past_kv=past,
            use_kv_cache=True,
            ablate_heads=ablate_heads,
        )
        mx.eval(logits)

    return tokenizer.decode(out_ids)


def format_chat_prompt(user_message: str) -> str:
    return f"<|user|>\n{user_message.strip()}\n<|assistant|>\n"


def format_chat_example(instruction: str, response: str) -> str:
    return (
        f"<|user|>\n{instruction.strip()}\n"
        f"<|assistant|>\n{response.strip()}<|endoftext|>"
    )
