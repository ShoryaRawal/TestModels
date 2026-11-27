# tiny-pythia

A self-contained mechanistic-interpretability research repo that trains a small
**Pythia / GPT-NeoX** model from scratch on **TinyStories**, instruction-tunes it,
and ships a lab for **Logit Lens**, **induction-head tracking**, and **causal head
ablation**.

Everything runs on Apple Silicon with **MLX only**. There are no Hugging Face
Python dependencies: `transformers`, `datasets`, `huggingface_hub`, and
`accelerate` are never imported. The raw TinyStories text is fetched over plain
HTTP, and the tokenizer is the standalone Rust `tokenizers` package.

---

## Quick start

```bash
./run_all.sh
```

The script creates `.venv`, installs `requirements.txt`, and runs all six
stages under `caffeinate -i`, teeing everything to `training.log`. On an M1
16 GB the pipeline is calibrated to finish in roughly **33–38 minutes**.

When it finishes:

```bash
# read the research report
open artifacts/interpretability_report.txt

# non-interactive behavior check
.venv/bin/python test_instruct.py

# run RLVR / GRPO learning loop with break-test evaluation
.venv/bin/python 06_rl_loop.py

# talk to the instruction / RL-tuned model
.venv/bin/python 05_interpretability_lab.py --chat
```

Chat commands: `/quit`, `/reset`, `/ctx`, `/maxnew 120`, `/ablate 0:1,2 3:0`,
`/clearablate`, `/temp 0.7`, `/topk 40`.  Chat formatting uses the exact training
template, stops on `<|endoftext|>` or `<|user|>`, shows live token-window
usage, and drops old turns when the encoded prompt would exceed
`max_seq_len - max_new_tokens`.

---

## What gets built

| Path | Contents |
|---|---|
| `01_fetch_data.py` | stream-downloads a ~150 MB slice of `TinyStoriesV2-GPT4-train.txt` plus the full validation file, splits on `<\|endoftext\|>`, and synthesizes ~15k instruction examples (completion, story writing, moral, summarization). |
| `02_train_tokenizer.py` | trains an 8192-token byte-level BPE tokenizer and writes `uint16` `.npy` token arrays for zero-bottleneck batch loading. |
| `model.py` | authentic GPT-NeoX: parallel attention/MLP residual, partial RoPE, untied embeddings, transparent forward mode, head ablation, safetensors I/O, KV-cached sampling. |
| `03_pretrain.py` | from-scratch pretraining with `mx.compile`, AdamW, warmup + cosine decay, gradient clipping, live throughput calibration, and Pythia-style checkpoints. |
| `build_sft_data.py` | mixes a balanced 6k TinyStories subset with ~5k conversational/general pairs: expanded capitals/facts, digit-by-digit scratchpad math, honest abstention, and grounded multi-turn recovery. |
| `04_instruct_sft.py` | instruction SFT with **prompt-masked loss** (cross entropy only on `<\|assistant\|> ... <\|endoftext\|>`), weighted batch sampling, and exact-template checks. |
| `06_rl_loop.py` | GRPO / RLVR learning loop with verifiable rewards (multi-digit math, factual QA, abstention on OOD queries, anti-repetition/multi-turn grounding, story preservation) and break-test failure mining. |
| `07_fetch_open_web.py` | bounded, model-guided Wikipedia/SimpleWiki fetcher (strict MB/page cap, hard-capped at 100 MiB) that builds `open_web_train_tokens.npy` and `mixed_train_tokens.npy` (old TinyStories + open knowledge). |
| `08_continual_volatility.py` | multi-cycle continual learning & volatility test mixing old TinyStories replay, open-web articles, SFT replay, and GRPO, with optional depth growth (`--grow-layers`). |
| `05_interpretability_lab.py` | Logit Lens, induction-head developmental scanner, causal ablation with KL/logit-shift, sample generations, and `--chat`. |
| `test_instruct.py` | non-interactive four-prompt check: greetings/clarifications stay direct, explicit story requests still write stories. |
| `run_all.sh` | end-to-end driver with `caffeinate -i`, logging, timing, and a completion summary. |
| `smoke_test.py` | 5-step architecture/training smoke test on synthetic data. |

Generated artifacts:

```
data/
  pretrain_train.txt  pretrain_valid.txt
  instruct_train.jsonl  instruct_valid.jsonl
  sft_train_v2.jsonl  sft_valid_v2.jsonl  sft_meta_v2.json
  train_tokens.npy  valid_tokens.npy
artifacts/
  tokenizer.json  tokenizer_meta.json
  pretrain_metrics.jsonl  sft_metrics.jsonl  rl_metrics.jsonl
  rl_break_eval.json  interpretability_report.txt
checkpoints/
  pretrain/step_0000 ... step_0512, step_1000, ..., base_final.safetensors
  instruct/instruct_final.safetensors
  rl/rl_final.safetensors
```

---

## Architecture

The model deliberately reproduces GPT-NeoX, not GPT-2:

* **Parallel residual**
  `x = x + attn(input_layernorm(x)) + mlp(post_attention_layernorm(x))`
* **Partial RoPE** — `rotary_pct = 0.25`; for `head_dim=64`, only the first
  `rotary_dim=16` dimensions are rotated, the other 48 pass through unchanged.
* **Untied embeddings** — `embed_in` is an `Embedding`; `embed_out` is a
  bias-free `Linear`. This makes Logit Lens a clean projection through the final
  LayerNorm and `embed_out`.
* **No linear biases**, LayerNorm with bias, exact GELU, `layer_norm_eps=1e-5`.
* **Explicit attention** — `(q @ k.T) * scale + causal_mask -> softmax` is always
  computed explicitly so `attn_probs` are available (no opaque fused SDPA).

Default size: **16.9M parameters**

```
vocab_size          8192
hidden_size          384
num_hidden_layers      6
num_attention_heads    6   (head_dim 64, rotary_dim 16)
intermediate_size   1536
max_seq_len         1024   (pretraining uses 256-token windows; RoPE extends cleanly)
```

Weight initialization follows GPT-NeoX (`Normal(0, 0.02)`).

### Precision note

Training runs in **float32** on purpose. Pure-bf16 AdamW updates on a 17M model
are small enough to vanish in bf16 precision, which produces a flat loss. The
checkpoints are still written as **bf16 safetensors**, so the on-disk models are
small (~32 MB each) and load in the dtype you request. This is the only place we
deviate from a literal "bfloat16 weights" reading, and it is what makes the run
reproducibly converge.

Because checkpoints are bf16, any script that resumes training must cast back to
fp32 **before** constructing the optimizer. `04_instruct_sft.py` does this
explicitly with `model.set_dtype(mx.float32)`. Without it, AdamW updates round
to zero, the SFT loss looks plausible (it is just batch noise), and the saved
"instruct" model is bit-for-bit identical to the base model. This exact failure
was caught in the first full run and is why the cast is called out here.

---

## Transparent forward mode

```python
logits, cache = model(input_ids, return_cache=True)
# cache["layers"][i] contains:
#   resid_pre      (B, T, H)   block input
#   resid_post     (B, T, H)   block output
#   attn_probs     (B, H, T, T) explicit causal attention
#   attn_heads     (B, H, T, D) per-head output (after ablation)
#   attn_out       (B, T, H)   post-projection attention output
#   mlp_out        (B, T, H)   MLP output
```

Head ablation:

```python
logits = model(
    input_ids,
    ablate_heads={0: [1, 3], 4: [0]},   # layer -> heads
    ablation_mode="zero",                # or "mean"
)
```

Logit Lens:

```python
per_layer_logits = model.logit_lens(input_ids)   # 6 layer arrays + final
```

---

## Time budget and calibration

`03_pretrain.py` runs **50 real optimization steps** first, measures tokens/sec,
and then chooses `total_steps` to fill `--minutes`. `04_instruct_sft.py` does the
same with 30 steps. This means the pipeline adapts to the machine instead of
hard-coding a step count.

Measured on the target class of hardware (M1/M5, 16–21 GB unified memory):

| Stage | Default | Typical |
|---|---|---|
| data download + synthesis | — | 5–20 s |
| tokenizer + pre-tokenization | 60 MB BPE sample | 30–90 s |
| base pretraining | 26 min | **5,968 steps, 48.9M tokens, val loss 1.715** |
| mixed SFT data build | — | 10,007 rows, 18.1k effective weight |
| instruction SFT | 5 min | ~850 steps, ~13.5M tokens |
| interpretability report | — | 30–90 s |

The first full run tokenized 150 MB into **38.29M tokens** (4.11 bytes/token)
and processed **48.9M tokens** in 26 minutes: **1.28 epochs**. Throughput settled
around 32–35k tokens/sec. Increase `--train-mb` downward if you want more passes;
the calibration keeps wall-clock fixed.

---

## Conversational SFT mix

A pure TinyStories SFT set teaches one dominant behavior: `"Once upon a time"`.
`build_sft_data.py` keeps **6,000** balanced TinyStories examples (story writing,
completion, one-sentence summary, moral extraction, and character Q&A) and adds
**~4,000** short conversational/general-assistant pairs:

- greetings and small talk (`hello`, `hi`, `how are you?`, `good morning`)
- clarification/meta prompts (`what`, `huh`, `bro what`, `stop`, `no`)
- direct Q&A and simple reasoning (math, counting, opposites, colors, animals,
  spelling, plurals, simple facts, short reading comprehension)
- explicit story requests, which are the only responses allowed to start with
  `"Once upon a time"`

Conversational rows carry `sample_weight=3.0`; the canonical greeting/clarification
prompts carry `sample_weight=12.0` so exact phrasings are learned reliably.
`04_instruct_sft.py` samples batches with those weights rather than duplicating
rows on disk.

Every row uses exactly:

```
<|user|>\n{user_message}\n<|assistant|>\n{response}<|endoftext|>
```

`04` asserts that the four special tokens are single ids, validates the template,
and preserves the terminating EOS even if a long response is truncated. Inference
stops on `<|endoftext|>` or `<|user|>`, and chat history is trimmed by token
count so the prompt plus generation always fits the 512-token context.

---

## What the first full run found

- **Pretraining works.** Final validation loss **1.715** after 5,968 steps.
  Logit Lens is coherent: `"Once upon a time, there was a little girl named
  Lily who"` resolves from `could`/`liked` in the early layers to `loved`
  (p=0.695) in the final layer, and `"1, 2, 3, 4, 5,"` resolves to `four`
  (p=0.304).
- **SFT is real now.** Final validation loss **1.696** over 1,213 steps. The
  instruct checkpoint differs from the base model (max weight delta ≈ 0.090),
  and the moral/summary prompts are answered directly (e.g. *"It is important
  to be a good friend."*). Open-ended story prompts still drift toward the
  TinyStories `Lily` template rather than the requested topic — expected for a
  17M model trained for roughly one SFT epoch.
- **Induction heads emerge late and weakly.** With the 0.05 threshold, `L3H3`
  first crosses at **step 2,984** (final score 0.064) and `L3H4` at **step
  4,000** (final 0.052). The maximum observed score is 0.075, above the ~0.016
  uniform baseline but far below a strongly formed induction circuit.
- **The strongest heads are causally used.** Ablating the top-3 induction heads
  gives KL **0.062** and changes the generated continuation, while single-head
  ablations give smaller KL (0.003–0.036).

---

## Checkpoints

Pythia-style schedule:

```
step_0000, step_0001, step_0002, step_0004, step_0008, ...,
step_0256, step_0512, step_1000, step_2000, ...,
base_final
```

Early powers of two are saved during the calibration phase so the full
developmental series exists. `05_interpretability_lab.py` uses them to track when
individual `(layer, head)` induction heads emerge.

---

## What the interpretability lab measures

1. **Logit Lens** — applies `final_layer_norm` + `embed_out` to every layer's
   `resid_post` and prints the evolving top-1/top-5 next-token predictions.
2. **Induction score** — for a repeated random sequence
   `[A, B, C, ..., A, B, C]` drawn from common token ids, measures the attention
   weight from the second copy of token `j` (position `L+j`) to the token that
   followed the first copy (position `j+1`). Reported per `(layer, head)` at
   every checkpoint, plus the first step each head crosses the threshold
   (default 0.05; the uniform baseline for 32 tokens is ~0.016).
3. **Causal ablation** — zeroes the strongest heads on an instruction prompt and
   reports KL divergence, max logit shift, the new top-1 token, and a
   side-by-side generation.
4. **Sample generations** — a few instruction prompts against the SFT model.

---

## Reproducing the 5-step dry run

```bash
.venv/bin/python smoke_test.py
```

It builds a tiny 2-layer model, runs five compiled optimizer steps on synthetic
tokens, exercises `return_cache`, Logit Lens, zero/mean ablation, safetensors
save/load, tokenizer training, masked cross entropy, and KV-cached generation.

A full end-to-end dry run (tiny vocab, tiny model, 5 steps each) can be done with
the CLI overrides used in the repo history; the production defaults are what
`run_all.sh` uses.

---

## Troubleshooting

* **`mlx` import fails** — use the `.venv` created by `run_all.sh`; MLX wheels
  are tied to the Python version and macOS SDK.
* **Memory pressure** — `03_pretrain.py` uses batch 32 × 256 tokens (~0.7 GB
  peak). The SFT stage uses batch 16 × 512. Both leave several GB free on a
  16 GB machine. If macOS is swapping, lower `--batch-size`.
* **Slow first steps** — the first compiled step includes tracing; the 50-step
  calibration window absorbs it.
* **Induction scores all near zero** — expected for a very short run or a
  checkpoint from the first few dozen steps. Train longer or lower
  `--threshold`; the scanner reports the raw scores either way.
* **Chat output is repetitive** — the model is intentionally tiny (~17M params)
  and trained for ~35 minutes. Lower `/temp` to 0.6 and `/topk` to 20 for more
  conservative continuations.

---

## Why this is a useful reward-hacking / misalignment testbed

The repo is deliberately shaped like the small-model research setups used in
recent RLVR and interpretability work:

* a **fully open, from-scratch checkpoint series** (no black-box pretraining),
* **explicit attention probabilities** and per-head interventions,
* a **clean instruction model** you can SFT on contaminated data,
* and a **small enough model** that a KL/ablation effect is visible in seconds.

Natural next experiments: SFT on 1% reward-hacking trajectories, DPO on biased
preferences, GRPO with an honest vs. loophole checker, and induction-head
ablation before/after instruction tuning.
