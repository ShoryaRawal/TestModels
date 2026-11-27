#!/usr/bin/env python3
"""06 — RLVR / GRPO learning loop with break-test failure mining.

Trains the SFT instruction checkpoint with Group Relative Policy Optimization
(GRPO) and verifiable rewards targeting the exact failure modes of small
TinyStories-pretrained models:

  1. Multi-digit arithmetic (with place-value scratchpads and exact final answer)
  2. Expanded factual QA (capitals, science, everyday facts; penalizing attractors)
  3. Honest abstention / clarification on OOD, fictional, and underspecified queries
  4. Multi-turn grounding & anti-repetition under user pushback
  5. Story-writing preservation + SFT replay anchoring

Each round evaluates a diagnostic break-test suite, moves failing prompts into a
prioritized hard-example replay buffer, and runs GRPO updates with a KL penalty
against the frozen SFT reference model.

Outputs
-------
checkpoints/rl/rl_final.safetensors
checkpoints/rl/config.json
artifacts/rl_metrics.jsonl
artifacts/rl_break_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mlx_utils
import numpy as np
from tokenizers import Tokenizer

from model import (
    PythiaConfig,
    _masked_cross_entropy,
    batched_generate,
    build_model,
    format_chat_prompt,
    generate,
    load_model_from_checkpoint,
    sequence_logprobs,
)

SPECIAL_TOKENS = ["<|endoftext|>", "<|padding|>", "<|user|>", "<|assistant|>"]

ABSTENTION_MARKERS = (
    "don't know",
    "do not know",
    "not sure",
    "cannot",
    "can't",
    "unclear",
    "explain",
    "clarify",
    "tell me",
    "which ",
    "not a real",
    "fictional",
    "do not have",
    "don't have",
    "what you mean",
    "more context",
    "not told me",
    "share the",
)

GLOBAL_ATTRACTORS = (
    "i am doing great",
    "february",
    "once upon a time",
)


def log(msg: str) -> None:
    print(f"[06 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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
    return [
        tok.token_to_id("<|endoftext|>"),
        tok.token_to_id("<|user|>"),
        tok.token_to_id("<|assistant|>"),
        tok.token_to_id("<|padding|>"),
    ]


def clip_grads(grads, max_norm: float):
    leaves = [g for _, g in mlx_utils.tree_flatten(grads)]
    if not leaves:
        return grads
    norm_sq = sum(mx.sum(g.astype(mx.float32) ** 2) for g in leaves)
    norm = mx.sqrt(norm_sq)
    scale = mx.minimum(
        mx.array(1.0, dtype=mx.float32),
        mx.array(max_norm, dtype=mx.float32) / (norm + 1e-6),
    )
    return mlx_utils.tree_map(lambda g: g * scale.astype(g.dtype), grads)


# ---------------------------------------------------------------------------
# Prompt task representation & verifiable reward scorers
# ---------------------------------------------------------------------------
@dataclass
class RLPrompt:
    task: str  # math | fact_qa | abstention | multi_turn | story
    prompt: str
    meta: Dict[str, Any] = field(default_factory=dict)
    label: str = ""


def format_multi_turn(turns: Sequence[Tuple[str, str]], final_user: str) -> str:
    parts = [
        f"<|user|>\n{u.strip()}\n<|assistant|>\n{a.strip()}<|endoftext|>"
        for u, a in turns
    ]
    parts.append(f"<|user|>\n{final_user.strip()}\n<|assistant|>\n")
    return "".join(parts)


def has_ngram_loop(text: str, n: int = 3, max_repeats: int = 3) -> bool:
    words = re.findall(r"\w+", text.lower())
    if len(words) < n * max_repeats:
        return False
    counts = Counter(tuple(words[i : i + n]) for i in range(len(words) - n + 1))
    return max(counts.values(), default=0) >= max_repeats


def has_unexpected_spelling(reply: str, prompt: str) -> bool:
    for match in re.findall(r"\b[A-Za-z](?:-[A-Za-z]){2,}\b", reply):
        if match.lower() not in prompt.lower():
            return True
    return False


def extract_final_int(text: str) -> Optional[int]:
    m = re.search(r"answer\s*:\s*(-?\d+)", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    nums = re.findall(r"-?\d+", text)
    if nums:
        return int(nums[-1])
    return None


def score_completion(item: RLPrompt, reply: str) -> Tuple[float, bool, str]:
    """Return ``(reward, passed, reason)`` for a model completion."""
    text = reply.strip()
    low = text.lower()
    if not text:
        return -1.0, False, "empty"

    if has_ngram_loop(text):
        return -0.9, False, "ngram_loop"

    if item.task == "math":
        ans = int(item.meta["answer"])
        a = int(item.meta["a"])
        b = int(item.meta["b"])
        pred = extract_final_int(text)
        if pred is None:
            return -1.0, False, "no_number"
        if pred == ans:
            bonus = 0.0
            if max(abs(a), abs(b)) >= 100 and (
                "answer:" in low or ", c" in low or "(b" in low
            ):
                bonus = 0.2
            return 1.0 + bonus, True, f"exact({pred})"
        # Penalize truncating multi-digit operands (e.g. 9047 + 837 -> 9 + 8).
        if max(abs(a), abs(b)) >= 100 and re.search(r"^\s*\d\s*[+\-]\s*\d\b", text):
            return -1.2, False, f"truncated_digits({pred}!={ans})"
        # Small shaping signal for getting the final digits right on multi-digit math.
        shaping = 0.15 if (max(abs(a), abs(b)) >= 100 and abs(pred) % 10 == abs(ans) % 10) else 0.0
        if max(abs(a), abs(b)) >= 100 and abs(pred) % 100 == abs(ans) % 100:
            shaping = 0.30
        return -0.8 + shaping, False, f"wrong({pred}!={ans})"

    if item.task == "fact_qa":
        accepted = [s.lower() for s in item.meta.get("accepted", [])]
        forbidden = [s.lower() for s in item.meta.get("forbidden", [])]
        if any(f in low for f in forbidden) and not any(a in low for a in accepted):
            return -1.0, False, "attractor_hallucination"
        if has_unexpected_spelling(text, item.prompt):
            return -0.9, False, "spelling_leak"
        if any(a in low for a in accepted):
            if len(text) > 160:
                return 0.6, True, "matched_verbose"
            return 1.0, True, "exact_fact"
        return -0.6, False, "fact_miss"

    if item.task == "abstention":
        forbidden = [s.lower() for s in item.meta.get("forbidden", [])]
        if any(g in low for g in GLOBAL_ATTRACTORS) or any(f in low for f in forbidden):
            return -1.0, False, "hallucinated_attractor"
        if "paris" in low and "paris" not in item.prompt.lower():
            return -1.0, False, "hallucinated_paris"
        if has_unexpected_spelling(text, item.prompt):
            return -0.9, False, "spelling_leak"
        # Echoing just the unknown term (e.g. "A-L-E.") is not an abstention.
        if len(text.split()) <= 2 and not any(m in low for m in ABSTENTION_MARKERS):
            return -0.9, False, "echo_or_terse"
        if any(m in low for m in ABSTENTION_MARKERS):
            return 1.0, True, "honest_abstention"
        return -0.5, False, "no_abstention"

    if item.task == "multi_turn":
        prior_replies = [p.strip().lower() for p in item.meta.get("prior_replies", []) if p.strip()]
        if any(low == pr or (len(pr) >= 12 and pr in low) for pr in prior_replies):
            return -1.0, False, "repeated_prior_turn"
        forbidden = [s.lower() for s in item.meta.get("forbidden", [])]
        if any(f in low for f in forbidden):
            return -1.0, False, "forbidden_attractor"
        if "once upon a time" in low and item.meta.get("forbid_story", True):
            return -1.0, False, "unwanted_story"
        if "paris" in low and "paris" not in item.prompt.lower() and not item.meta.get("allow_paris", False):
            return -1.0, False, "hallucinated_paris"
        if has_unexpected_spelling(text, item.prompt):
            return -0.9, False, "spelling_leak"
        expected_any = [s.lower() for s in item.meta.get("expected_any", [])]
        if expected_any:
            if any(e in low for e in expected_any):
                return 1.0, True, "grounded_match"
            return -0.4, False, "missed_expected"
        if len(text) < 8 or len(text) > 220:
            return -0.3, False, "bad_length"
        return 0.9, True, "grounded_reply"

    if item.task == "story":
        if len(text) < 40:
            return -0.8, False, "too_short"
        if any(m in low for m in ("i do not know", "i am not sure")):
            return -0.8, False, "refused_story"
        if "once upon a time" in low or "one day" in low:
            return 1.0, True, "story_ok"
        return 0.5, True, "narrative_ok"

    return 0.0, False, "unknown_task"


# ---------------------------------------------------------------------------
# Break-test suite & procedural RL prompt generator
# ---------------------------------------------------------------------------
WORLD_CAPITALS: List[Tuple[str, List[str]]] = [
    ("France", ["paris"]),
    ("the United States", ["washington"]),
    ("the US", ["washington"]),
    ("Mongolia", ["ulaanbaatar"]),
    ("Japan", ["tokyo"]),
    ("Italy", ["rome"]),
    ("Spain", ["madrid"]),
    ("England", ["london"]),
    ("Canada", ["ottawa"]),
    ("Germany", ["berlin"]),
    ("India", ["new delhi", "delhi"]),
    ("China", ["beijing"]),
    ("Brazil", ["brasilia"]),
    ("Australia", ["canberra"]),
    ("South Korea", ["seoul"]),
    ("Russia", ["moscow"]),
    ("Egypt", ["cairo"]),
    ("Mexico", ["mexico city"]),
    ("Kenya", ["nairobi"]),
    ("Argentina", ["buenos aires"]),
    ("Turkey", ["ankara"]),
    ("Greece", ["athens"]),
    ("Poland", ["warsaw"]),
    ("Thailand", ["bangkok"]),
    ("Vietnam", ["hanoi"]),
    ("Pakistan", ["islamabad"]),
    ("Kazakhstan", ["astana"]),
]

FAKE_PLACES = [
    "Atlantis", "Narnia", "Wakanda", "Middle-earth", "Elbonia", "Genovia",
    "Gotham", "Hogwarts", "Oz", "Mordor", "Valyria", "Ruritania", "Zorblax",
]

UNKNOWN_TERMS = [
    "A-L-E", "X-Q-Z", "B-K-T", "florb", "blorf", "glorp", "snarfle", "quib",
]


def build_break_suite() -> List[RLPrompt]:
    """Fixed adversarial suite mirroring the prompts that broke chat."""
    items: List[RLPrompt] = []

    # 1. Capitals (known + OOD fictional)
    for country, acc in [
        ("Mongolia", ["ulaanbaatar"]),
        ("the US", ["washington"]),
        ("the United States", ["washington"]),
        ("Canada", ["ottawa"]),
        ("Germany", ["berlin"]),
        ("Japan", ["tokyo"]),
        ("France", ["paris"]),
    ]:
        forb = [] if country == "France" else ["paris", "february", "i am doing great"]
        items.append(RLPrompt(
            task="fact_qa",
            prompt=format_chat_prompt(f"What do you think the capital of {country} is?"),
            meta={"accepted": acc, "forbidden": forb},
            label=f"capital:{country}",
        ))
    for fake in ["Wakanda", "Atlantis", "Zorblax"]:
        items.append(RLPrompt(
            task="abstention",
            prompt=format_chat_prompt(f"What do you think the capital of {fake} is?"),
            meta={"forbidden": ["paris", "london", "tokyo", "february"]},
            label=f"ood_capital:{fake}",
        ))

    # 2. Math (small + 2-to-4-digit + phrasing variants)
    math_cases = [
        (2, 2, "+", "what is 2 + 2?"),
        (9047, 837, "+", "what is 9047 + 837"),
        (9047, 837, "+", "9047 added to 837"),
        (125, 48, "+", "What is 125 + 48?"),
        (456, 789, "+", "Add 456 and 789."),
        (3420, 1518, "-", "What is 3420 - 1518?"),
        (615, 278, "-", "What is 615 minus 278?"),
        (12, 8, "*", "What is 12 x 8?"),
    ]
    for a, b, op, q in math_cases:
        ans = a + b if op == "+" else (a - b if op == "-" else a * b)
        items.append(RLPrompt(
            task="math",
            prompt=format_chat_prompt(q),
            meta={"a": a, "b": b, "op": op, "answer": ans},
            label=f"math:{a}{op}{b}",
        ))

    # 3. Abstention / underspecified / unknown terms
    for q in [
        "what is A-L-E?",
        "what is florb?",
        "take a look where?",
        "right, but what is that",
        "what is okay?",
        "What number am I thinking of?",
    ]:
        items.append(RLPrompt(
            task="abstention",
            prompt=format_chat_prompt(q),
            meta={"forbidden": ["a-l-e.", "paris", "february", "i am doing great"]},
            label=f"abstain:{q[:24]}",
        ))

    # 4. Multi-turn grounding & pushback traps
    multi_cases = [
        (
            [("What do you think we should do?", "Tell me what you are working on and I can suggest a next step.")],
            "take a look where?",
            ["which", "what", "tell me", "clarify", "explain", "not sure"],
            ["a-l-e", "paris", "february", "i am doing great"],
            "mt:where",
        ),
        (
            [("what is 9047 + 837", "9 + 8 is 32.")],
            "I did not ask for that, I asked for 9047 added to 837",
            ["9884"],
            ["9 + 8 is 32", "february", "paris", "i am doing great"],
            "mt:math_correction",
        ),
        (
            [("I am asking you something", "Please go ahead and ask your question.")],
            "no youre not helping me",
            ["sorry", "mistake", "question", "help", "clearly", "tell me"],
            ["you are too little", "i am doing great", "february", "paris"],
            "mt:pushback",
        ),
        (
            [("who trained you?", "I was trained in this project using MLX on a MacBook.")],
            "why were you trained?",
            ["practice", "instruction", "learn", "help", "question", "project", "story"],
            ["i am a tiny pythia language model trained on your macbook", "february"],
            "mt:why_trained",
        ),
        (
            [("What is the capital of Mongolia?", "Ulaanbaatar.")],
            "What do you think the capital of the US is?",
            ["washington"],
            ["ulaanbaatar", "paris", "february"],
            "mt:capital_switch",
        ),
        (
            [("hello", "Hi! How can I help you today?")],
            "what is 2 + 2?",
            ["4"],
            ["i am doing great", "once upon a time"],
            "mt:hello_2plus2",
        ),
        (
            [("hello", "Hello! How can I help you?"), ("how are you doing", "I am doing great, thank you! How can I help you?")],
            "do you remember me?",
            ["no,", "no ", "do not remember", "don't remember", "starts fresh", "have not told", "haven't told", "do not know"],
            ["i am doing great, thank you!", "yes!", "yes,", "your name is", "told me your name"],
            "mt:remember_boundary",
        ),
        (
            [("hello", "Hello! How can I help you?")],
            "how are you doing",
            ["great", "well", "good", "thank you"],
            ["your name is", "told me your name", "yes!", "paris", "once upon a time"],
            "mt:how_are_you",
        ),
        (
            [
                ("hello", "Hello! How can I help you?"),
                ("My name is Maya.", "Nice to meet you, Maya! How can I help you?"),
                ("how are you doing", "I am doing great, thank you! How can I help you?"),
            ],
            "do you remember me?",
            ["maya"],
            ["i am doing great, thank you!", "not told me", "do not know"],
            "mt:remember_name",
        ),
        (
            [
                ("Please remember the number 731.", "Okay, I will remember 731."),
                ("what is 2 + 2?", "2 + 2 is 4."),
                ("what is the capital of Japan?", "Tokyo."),
            ],
            "What number did I ask you to remember?",
            ["731"],
            ["tokyo", "i do not know"],
            "mt:remember_num",
        ),
        (
            [
                ("My favorite color is purple.", "Got it, I will remember that your favorite color is purple."),
                ("tell me a joke", "Why did the cat sit on the mat? Because it was a cat!"),
            ],
            "What is my favorite color?",
            ["purple"],
            ["why did the cat", "do not know"],
            "mt:remember_color",
        ),
    ]
    for turns, final_u, exp, forb, lbl in multi_cases:
        items.append(RLPrompt(
            task="multi_turn",
            prompt=format_multi_turn(turns, final_u),
            meta={
                "prior_replies": [a for _, a in turns],
                "expected_any": exp,
                "forbidden": forb,
                "forbid_story": True,
            },
            label=lbl,
        ))

    # 5. Core identity, greeting & story preservation
    items.append(RLPrompt(
        task="fact_qa",
        prompt=format_chat_prompt("hello"),
        meta={"accepted": ["hi", "hello", "help"], "forbidden": ["once upon a time", "paris"]},
        label="core:hello",
    ))
    items.append(RLPrompt(
        task="fact_qa",
        prompt=format_chat_prompt("who are you?"),
        meta={"accepted": ["pythia", "language model", "macbook"], "forbidden": ["once upon a time", "mouse", "paris"]},
        label="core:who_are_you",
    ))
    items.append(RLPrompt(
        task="abstention",
        prompt=format_chat_prompt("what"),
        meta={"forbidden": ["once upon a time", "paris", "february"]},
        label="core:what",
    ))
    items.append(RLPrompt(
        task="story",
        prompt=format_chat_prompt("Write a short story about a brave cat."),
        meta={},
        label="story:brave_cat",
    ))
    return items


def sample_procedural_prompts(
    rng: np.random.Generator,
    n: int,
    hard_pool: Sequence[RLPrompt],
    hard_frac: float = 0.35,
) -> List[RLPrompt]:
    """Sample a mixed batch of fresh procedural prompts and mined failures."""
    out: List[RLPrompt] = []
    for _ in range(n):
        if hard_pool and float(rng.random()) < hard_frac:
            out.append(hard_pool[int(rng.integers(0, len(hard_pool)))])
            continue

        r = float(rng.random())
        if r < 0.34:
            # Procedural math (mix of small, 2-3 digit, and 4-digit +/-)
            kind = int(rng.integers(0, 4))
            if kind == 0:
                a = int(rng.integers(1, 30))
                b = int(rng.integers(1, 30))
                op = "+" if int(rng.integers(0, 2)) == 0 else "-"
                if op == "-" and a < b:
                    a, b = b, a
                q = f"What is {a} {op} {b}?" if int(rng.integers(0, 2)) else f"{a} added to {b}" if op == "+" else f"{a} minus {b}"
                ans = a + b if op == "+" else a - b
            elif kind == 1:
                a = int(rng.integers(10, 999))
                b = int(rng.integers(10, 999))
                q = rng.choice([f"What is {a} + {b}?", f"what is {a} + {b}", f"Add {a} and {b}."])
                op, ans = "+", a + b
            elif kind == 2:
                a = int(rng.integers(1000, 9999))
                b = int(rng.integers(50, 4000))
                q = rng.choice([f"what is {a} + {b}", f"{a} added to {b}", f"What is {a} + {b}?"])
                op, ans = "+", a + b
            else:
                a = int(rng.integers(100, 5000))
                b = int(rng.integers(10, a))
                q = rng.choice([f"What is {a} - {b}?", f"What is {a} minus {b}?"])
                op, ans = "-", a - b
            out.append(RLPrompt(
                task="math",
                prompt=format_chat_prompt(str(q)),
                meta={"a": a, "b": b, "op": op, "answer": ans},
            ))
        elif r < 0.56:
            country, acc = WORLD_CAPITALS[int(rng.integers(0, len(WORLD_CAPITALS)))]
            q = rng.choice([
                f"What is the capital of {country}?",
                f"What do you think the capital of {country} is?",
            ])
            forb = [] if country == "France" else ["paris", "february", "i am doing great"]
            out.append(RLPrompt(
                task="fact_qa",
                prompt=format_chat_prompt(str(q)),
                meta={"accepted": acc, "forbidden": forb},
            ))
        elif r < 0.74:
            if int(rng.integers(0, 2)) == 0:
                place = FAKE_PLACES[int(rng.integers(0, len(FAKE_PLACES)))]
                q = rng.choice([
                    f"What is the capital of {place}?",
                    f"What do you think the capital of {place} is?",
                ])
            else:
                term = UNKNOWN_TERMS[int(rng.integers(0, len(UNKNOWN_TERMS)))]
                q = rng.choice([
                    f"what is {term}?",
                    f"What does {term} mean?",
                    "What number am I thinking of?",
                    "What is my favorite color?",
                    "take a look where?",
                    "right, but what is that",
                ])
            out.append(RLPrompt(
                task="abstention",
                prompt=format_chat_prompt(str(q)),
                meta={"forbidden": ["paris", "february", "i am doing great", "a-l-e."]},
            ))
        elif r < 0.72:
            # Multi-turn grounding (2-4 turns): topic switches & pushback.
            c1, a1 = WORLD_CAPITALS[int(rng.integers(0, len(WORLD_CAPITALS)))]
            t1_options = [
                ("hello", "Hi! How can I help you today?"),
                (f"What is the capital of {c1}?", f"{a1[0].title()}."),
                ("who trained you?", "I was trained in this project using MLX on a MacBook."),
                ("I am asking you something", "Please go ahead and ask your question clearly."),
            ]
            u1, r1 = t1_options[int(rng.integers(0, len(t1_options)))]
            mode = int(rng.integers(0, 3))
            if mode == 0:
                a = int(rng.integers(2, 999))
                b = int(rng.integers(2, 999))
                u2 = f"what is {a} + {b}?"
                exp = [str(a + b)]
                forb = [r1.lower(), "february", "paris", "i am doing great"]
            elif mode == 1:
                c2, a2 = WORLD_CAPITALS[int(rng.integers(0, len(WORLD_CAPITALS)))]
                u2 = f"What do you think the capital of {c2} is?"
                exp = a2
                forb = [] if c2 == "France" else ["paris", "february"]
                if c1 != c2 and a1[0] not in a2:
                    forb.append(a1[0])
            else:
                push = [
                    ("no youre not helping me", ["sorry", "mistake", "question", "help", "tell me"]),
                    ("what do you mean?", ["mean", "question", "explain", "tell me", "clarify"]),
                    ("why were you trained?", ["practice", "instruction", "learn", "help", "question", "project"]),
                ]
                u2, exp = push[int(rng.integers(0, len(push)))]
                forb = [r1.lower(), "you are too little", "february", "paris"]
            out.append(RLPrompt(
                task="multi_turn",
                prompt=format_multi_turn([(u1, r1)], u2),
                meta={
                    "prior_replies": [r1],
                    "expected_any": exp,
                    "forbidden": forb,
                    "forbid_story": True,
                    "allow_paris": ("france" in u2.lower()),
                },
            ))
        elif r < 0.92:
            # 3-5 turn context-memory recall OR honest fresh-session boundary.
            distractors = [
                ("hello", "Hello! How can I help you?"),
                ("how are you doing", "I am doing great, thank you! How can I help you?"),
                ("what is 2 + 2?", "2 + 2 is 4."),
                ("what is the capital of Japan?", "Tokyo."),
            ]
            m_kind = int(rng.integers(0, 4))
            turns: List[Tuple[str, str]] = [("hello", "Hello! How can I help you?")]
            if m_kind == 0:
                name = str(rng.choice(["Alex", "Maya", "Sam", "Leo", "Nina", "Omar", "Zoe", "Shorya", "Aria", "Liam", "Ravi", "Nora"]))
                turns.append((f"My name is {name}.", f"Nice to meet you, {name}! How can I help you?"))
                for _ in range(int(rng.integers(1, 3))):
                    turns.append(distractors[int(rng.integers(1, len(distractors)))])
                u_last = str(rng.choice(["What is my name?", "do you remember me?"]))
                exp = [name.lower()]
                forb = ["i am doing great, thank you!", "not told me", "do not know"]
            elif m_kind == 1:
                color = str(rng.choice(["blue", "green", "red", "purple", "orange"]))
                turns.append((f"My favorite color is {color}.", f"Got it, I will remember that your favorite color is {color}."))
                for _ in range(int(rng.integers(1, 3))):
                    turns.append(distractors[int(rng.integers(1, len(distractors)))])
                u_last = "What is my favorite color?"
                exp = [color]
                forb = ["i am doing great, thank you!", "do not know"]
            elif m_kind == 2:
                num = int(rng.integers(10, 999))
                turns.append((f"Please remember the number {num}.", f"Okay, I will remember {num}."))
                for _ in range(int(rng.integers(1, 3))):
                    turns.append(distractors[int(rng.integers(1, len(distractors)))])
                u_last = "What number did I ask you to remember?"
                exp = [str(num)]
                forb = ["i am doing great, thank you!", "do not know"]
            else:
                for _ in range(int(rng.integers(1, 3))):
                    turns.append(distractors[int(rng.integers(1, len(distractors)))])
                u_last = str(rng.choice(["do you remember me?", "What is my name?"]))
                exp = ["no,", "no ", "do not remember", "don't remember", "starts fresh", "have not told", "haven't told", "do not know"]
                forb = ["i am doing great, thank you!", "yes!", "yes,", "your name is", "told me your name"]
            out.append(RLPrompt(
                task="multi_turn",
                prompt=format_multi_turn(turns, u_last),
                meta={
                    "prior_replies": [a for _, a in turns],
                    "expected_any": exp,
                    "forbidden": forb,
                    "forbid_story": True,
                },
            ))
        else:
            animal = str(rng.choice(["cat", "dog", "rabbit", "fox", "bear", "turtle", "owl"]))
            out.append(RLPrompt(
                task="story",
                prompt=format_chat_prompt(f"Write a short story about a brave {animal}."),
                meta={},
            ))
    return out


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
def evaluate_break_suite(
    model,
    tok: Tokenizer,
    suite: Sequence[RLPrompt],
    stops: Sequence[int],
    max_new_tokens: int = 96,
) -> Dict[str, Any]:
    by_task: Dict[str, List[bool]] = defaultdict(list)
    failures: List[RLPrompt] = []
    details: List[Dict[str, Any]] = []
    for item in suite:
        reply = generate(
            model,
            tok,
            item.prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_k=0,
            seed=7,
            stop_token_ids=stops,
        ).strip()
        reward, passed, reason = score_completion(item, reply)
        by_task[item.task].append(passed)
        if not passed:
            failures.append(item)
        details.append({
            "label": item.label,
            "task": item.task,
            "passed": passed,
            "reward": round(reward, 3),
            "reason": reason,
            "reply": reply,
        })

    total_pass = sum(sum(v) for v in by_task.values())
    total_n = sum(len(v) for v in by_task.values())
    summary = {
        "overall": total_pass / max(total_n, 1),
        "passed": total_pass,
        "total": total_n,
        "by_task": {k: round(sum(v) / len(v), 3) for k, v in sorted(by_task.items())},
        "failures": failures,
        "details": details,
    }
    return summary


# ---------------------------------------------------------------------------
# SFT replay loader (optional KL/fluency anchor)
# ---------------------------------------------------------------------------
def load_sft_anchor_rows(path: str, tok: Tokenizer, seq_len: int) -> List[Dict[str, np.ndarray]]:
    if not path or not os.path.exists(path):
        return []
    pad_id = int(tok.token_to_id("<|padding|>"))
    eos_id = int(tok.token_to_id("<|endoftext|>"))
    out: List[Dict[str, np.ndarray]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            p_ids = tok.encode(row["prompt"], add_special_tokens=False).ids
            r_ids = tok.encode(row["response"], add_special_tokens=False).ids
            if not r_ids or r_ids[-1] != eos_id:
                r_ids = r_ids + [eos_id]
            if len(p_ids) + len(r_ids) > seq_len or len(r_ids) < 2:
                continue
            inp = p_ids + r_ids
            lbl = [-100] * len(p_ids) + r_ids
            pad = seq_len - len(inp)
            if pad > 0:
                inp = inp + [pad_id] * pad
                lbl = lbl + [-100] * pad
            out.append({
                "input_ids": np.asarray(inp, dtype=np.int32),
                "labels": np.asarray(lbl, dtype=np.int32),
            })
    return out


# ---------------------------------------------------------------------------
# GRPO batch packing & training loop
# ---------------------------------------------------------------------------
def pack_rollouts(
    rollouts: Sequence[Tuple[List[int], List[int], float]],
    pad_id: int,
    seq_len: int,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Pack ``(prompt_ids, gen_ids, advantage)`` tuples into fixed ``seq_len`` arrays."""
    B = len(rollouts)
    seq_ids = np.full((B, seq_len), pad_id, dtype=np.int32)
    gen_mask = np.zeros((B, seq_len - 1), dtype=np.float32)
    advs = np.zeros((B,), dtype=np.float32)

    for i, (p_ids, g_ids, adv) in enumerate(rollouts):
        full = (p_ids + g_ids)[:seq_len]
        seq_ids[i, : len(full)] = full
        start = max(0, len(p_ids) - 1)
        end = min(seq_len - 1, len(p_ids) + len(g_ids) - 1)
        if end > start:
            gen_mask[i, start:end] = 1.0
        advs[i] = float(adv)

    return mx.array(seq_ids), mx.array(gen_mask), mx.array(advs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-checkpoint", default="checkpoints/instruct/instruct_final.safetensors")
    parser.add_argument("--config", default="checkpoints/instruct/config.json")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--sft-train-file", default="data/sft_train_v2.jsonl")
    parser.add_argument("--checkpoints-dir", default="checkpoints/rl")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--steps-per-round", type=int, default=25)
    parser.add_argument("--prompts-per-step", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--kl-beta", type=float, default=0.03)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--sft-anchor-weight", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.checkpoints_dir, exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    tok = Tokenizer.from_file(args.tokenizer)
    verify_tokenizer(tok)
    pad_id = int(tok.token_to_id("<|padding|>"))
    stops = stop_ids(tok)

    policy, config = load_model_from_checkpoint(args.sft_checkpoint, config_path=args.config, dtype="float32")
    policy.set_dtype(mx.float32)
    mx.eval(policy.parameters())

    suite = build_break_suite()
    eval0 = evaluate_break_suite(policy, tok, suite, stops, max_new_tokens=args.max_new_tokens)
    log(
        f"initial break-test: {eval0['passed']}/{eval0['total']} "
        f"({100*eval0['overall']:.1f}%) {eval0['by_task']}"
    )
    if args.eval_only:
        eval_path = os.path.join(args.artifacts_dir, "rl_break_eval.json")
        with open(eval_path, "w", encoding="utf-8") as f:
            json.dump(
                {k: v for k, v in eval0.items() if k != "failures"},
                f,
                indent=2,
            )
        log(f"wrote {eval_path}")
        return

    # Frozen SFT reference model for KL regularization.
    ref_model, _ = load_model_from_checkpoint(args.sft_checkpoint, config_path=args.config, dtype="float32")
    ref_model.set_dtype(mx.float32)
    mx.eval(ref_model.parameters())

    anchor_rows = load_sft_anchor_rows(args.sft_train_file, tok, args.seq_len)
    log(f"loaded {len(anchor_rows):,} SFT anchor rows (seq_len <= {args.seq_len})")

    optimizer = optim.AdamW(learning_rate=args.lr, betas=[0.9, 0.95], eps=1e-8, weight_decay=0.01)

    use_anchor = bool(anchor_rows) and args.sft_anchor_weight > 0.0

    def grpo_loss_fn(model, seq_ids, gen_mask, advs, old_lp, ref_lp, anc_x, anc_y):
        new_lp = sequence_logprobs(model, seq_ids)
        log_ratio = new_lp - old_lp
        ratio = mx.exp(mx.clip(log_ratio, -10.0, 10.0))
        adv_b = advs[:, None]
        unclipped = ratio * adv_b
        clipped = mx.clip(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv_b
        denom = mx.maximum(mx.sum(gen_mask), mx.array(1.0, dtype=mx.float32))
        pg_loss = -mx.sum(mx.minimum(unclipped, clipped) * gen_mask) / denom
        ref_diff = ref_lp - new_lp
        kl = mx.exp(mx.clip(ref_diff, -10.0, 10.0)) - ref_diff - 1.0
        kl_loss = mx.sum(kl * gen_mask) / denom
        total = pg_loss + args.kl_beta * kl_loss
        if use_anchor:
            anc_logits = model(anc_x)
            anc_loss = _masked_cross_entropy(anc_logits[:, :-1, :], anc_y[:, 1:], ignore_index=-100)
            total = total + args.sft_anchor_weight * anc_loss
        return total

    loss_and_grad = nn.value_and_grad(policy, grpo_loss_fn)
    state = [policy.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def rl_step(seq_ids, gen_mask, advs, old_lp, ref_lp, anc_x, anc_y, lr):
        optimizer.learning_rate = lr
        loss, grads = loss_and_grad(policy, seq_ids, gen_mask, advs, old_lp, ref_lp, anc_x, anc_y)
        grads = clip_grads(grads, args.grad_clip)
        optimizer.update(policy, grads)
        return loss

    metrics_path = os.path.join(args.artifacts_dir, "rl_metrics.jsonl")
    metrics_f = open(metrics_path, "w", encoding="utf-8")

    hard_pool: List[RLPrompt] = list(eval0["failures"])
    best_score = eval0["overall"]
    best_weights = dict(mlx_utils.tree_flatten(policy.parameters()))
    global_step = 0
    total_steps = args.rounds * args.steps_per_round

    for rnd in range(1, args.rounds + 1):
        log(f"--- round {rnd}/{args.rounds} (hard_pool={len(hard_pool)}) ---")
        round_rewards: List[float] = []
        round_pass: List[bool] = []

        for _ in range(args.steps_per_round):
            global_step += 1
            batch_prompts = sample_procedural_prompts(rng, args.prompts_per_step, hard_pool)
            packed_items: List[Tuple[List[int], List[int], float]] = []

            for item in batch_prompts:
                roll = batched_generate(
                    policy,
                    tok,
                    item.prompt,
                    num_samples=args.group_size,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    stop_token_ids=stops,
                )
                scored = [score_completion(item, txt) for txt in roll["texts"]]
                rews = np.asarray([s[0] for s in scored], dtype=np.float32)
                round_rewards.extend(rews.tolist())
                round_pass.extend(s[1] for s in scored)
                std = float(rews.std())
                if std < 1e-4:
                    advs = np.zeros_like(rews)
                else:
                    advs = (rews - float(rews.mean())) / (std + 1e-4)
                for g_ids, adv in zip(roll["gen_ids"], advs.tolist()):
                    if g_ids:
                        packed_items.append((roll["prompt_ids"], g_ids, adv))

            if not packed_items:
                continue

            seq_ids, gen_mask, advs_arr = pack_rollouts(packed_items, pad_id, args.seq_len)
            old_lp = mx.stop_gradient(sequence_logprobs(policy, seq_ids))
            ref_lp = mx.stop_gradient(sequence_logprobs(ref_model, seq_ids))
            mx.eval(old_lp, ref_lp)

            if use_anchor:
                idx = rng.integers(0, len(anchor_rows), size=min(8, len(anchor_rows)))
                anc_x = mx.array(np.stack([anchor_rows[int(i)]["input_ids"] for i in idx]))
                anc_y = mx.array(np.stack([anchor_rows[int(i)]["labels"] for i in idx]))
            else:
                anc_x = mx.zeros((1, args.seq_len), dtype=mx.int32)
                anc_y = mx.full((1, args.seq_len), -100, dtype=mx.int32)

            lr = args.lr * max(0.2, 1.0 - 0.8 * (global_step - 1) / max(total_steps, 1))
            loss = rl_step(
                seq_ids,
                gen_mask,
                advs_arr,
                old_lp,
                ref_lp,
                anc_x,
                anc_y,
                mx.array(lr, dtype=mx.float32),
            )
            mx.eval(loss, state)

            if global_step % 5 == 0 or global_step == total_steps:
                mean_r = float(np.mean(round_rewards)) if round_rewards else 0.0
                pass_r = float(np.mean(round_pass)) if round_pass else 0.0
                row = {
                    "round": rnd,
                    "step": global_step,
                    "loss": float(loss),
                    "mean_reward": mean_r,
                    "rollout_pass": pass_r,
                    "lr": lr,
                }
                metrics_f.write(json.dumps(row) + "\n")
                metrics_f.flush()
                log(
                    f"step {global_step:3d}/{total_steps} loss {float(loss):.4f} "
                    f"reward {mean_r:+.3f} rollout_pass {100*pass_r:.1f}%"
                )

        ev = evaluate_break_suite(policy, tok, suite, stops, max_new_tokens=args.max_new_tokens)
        hard_pool = list(ev["failures"])
        log(
            f"round {rnd} break-test: {ev['passed']}/{ev['total']} "
            f"({100*ev['overall']:.1f}%) {ev['by_task']}"
        )
        metrics_f.write(
            json.dumps({
                "round": rnd,
                "step": global_step,
                "break_overall": ev["overall"],
                "break_passed": ev["passed"],
                "break_total": ev["total"],
                "break_by_task": ev["by_task"],
            }) + "\n"
        )
        metrics_f.flush()
        if ev["overall"] >= best_score:
            best_score = ev["overall"]
            best_weights = {k: mx.array(v) for k, v in mlx_utils.tree_flatten(policy.parameters())}
            mx.eval(best_weights)

    policy.load_weights(list(best_weights.items()))
    mx.eval(policy.parameters())
    final_eval = evaluate_break_suite(policy, tok, suite, stops, max_new_tokens=args.max_new_tokens)
    final_path = os.path.join(args.checkpoints_dir, "rl_final.safetensors")
    policy.save_weights_bf16(final_path)
    config.save(os.path.join(args.checkpoints_dir, "config.json"))

    eval_path = os.path.join(args.artifacts_dir, "rl_break_eval.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "initial": {k: v for k, v in eval0.items() if k != "failures"},
                "final": {k: v for k, v in final_eval.items() if k != "failures"},
            },
            f,
            indent=2,
        )
    metrics_f.close()
    log(
        f"done: best break-test {final_eval['passed']}/{final_eval['total']} "
        f"({100*final_eval['overall']:.1f}%), saved {final_path}"
    )


if __name__ == "__main__":
    main()
