#!/usr/bin/env python3
"""Build a mixed instruction dataset for Phase 2.

The original ``01_fetch_data.py`` dataset is almost entirely TinyStories
instructions, which teaches the model a very strong ``"Once upon a time"``
prior.  This script keeps a balanced 6k TinyStories subset and adds 4k short
conversational / general-assistant examples, so greetings and simple questions
get direct answers while explicit story requests still produce stories.

Prompt template (identical to training and inference)::

    <|user|>\n{user_message}\n<|assistant|>\n{response}<|endoftext|>

Outputs
-------
data/sft_train_v2.jsonl   ~10k unique rows (6k stories + 4k conversation)
data/sft_valid_v2.jsonl   ~800 held-out rows
data/sft_meta_v2.json     counts, token stats, special-token ids

Conversational rows carry ``sample_weight=3.0``.  A small set of canonical
greeting/clarification prompts carry a higher weight so exact phrasings such as
``"hello"`` and ``"what"`` are reliably learned.  ``04_instruct_sft.py`` uses
these weights for weighted batch sampling.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from tokenizers import Tokenizer

EOS = "<|endoftext|>"
SPECIAL_TOKENS = ["<|endoftext|>", "<|padding|>", "<|user|>", "<|assistant|>"]
DEFAULT_SEQ_LEN = 1024
CONVERSATIONAL_WEIGHT = 3.0
CORE_WEIGHT = 16.0
CORE_QA_WEIGHT = 6.0


def log(msg: str) -> None:
    print(f"[sft-data {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            n += 1
    log(f"wrote {path}: {n:,} rows")


def make_row(task: str, user: str, response: str, weight: float = 1.0) -> dict:
    user = user.strip()
    response = response.strip()
    if not response.endswith(EOS):
        response += EOS
    return {
        "task": task,
        "instruction": user,
        "prompt": f"<|user|>\n{user}\n<|assistant|>\n",
        "response": response,
        "sample_weight": float(weight),
    }


def dedupe(rows: Sequence[dict]) -> List[dict]:
    seen = set()
    out = []
    for row in rows:
        key = (row["prompt"], row["response"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def case_variants(user: str) -> List[str]:
    """Return the original plus lower-/upper-initial variants of a prompt."""
    user = user.strip()
    variants = [user]
    for candidate in (user[:1].lower() + user[1:], user[:1].upper() + user[1:]):
        if candidate not in variants:
            variants.append(candidate)
    return variants


def add_core_rows(
    rows: List[dict],
    existing: Dict[Tuple[str, str], dict],
    pairs: Sequence[Tuple[str, str]],
    weight: float,
    task: str,
) -> int:
    added = 0
    for user, response in pairs:
        for variant in case_variants(user):
            row = make_row(task, variant, response, weight=weight)
            key = (row["prompt"], row["response"])
            if key in existing:
                existing[key]["sample_weight"] = max(
                    float(existing[key].get("sample_weight", 1.0)), float(weight)
                )
            else:
                rows.append(row)
                existing[key] = row
                added += 1
    return added


def make_multi_turn_row(
    first_user: str,
    first_response: str,
    second_user: str,
    second_response: str,
    weight: float,
) -> dict:
    return make_chat_chain_row(
        [(first_user, first_response)],
        second_user,
        second_response,
        weight=weight,
        task="multi_turn",
    )


def make_chat_chain_row(
    prior_turns: Sequence[Tuple[str, str]],
    final_user: str,
    final_response: str,
    weight: float,
    task: str = "multi_turn",
) -> dict:
    parts = []
    for u, a in prior_turns:
        row = make_row("turn", u, a)
        parts.append(row["prompt"] + row["response"])
    last = make_row(task, final_user, final_response, weight=weight)
    return {
        "task": task,
        "instruction": final_user.strip(),
        # Exact concatenation used by 05's build_chat_prompt.
        "prompt": "".join(parts) + last["prompt"],
        "response": last["response"],
        "sample_weight": float(weight),
    }


def build_context_memory_rows(
    tok: Tokenizer,
    seq_len: int,
    rng: np.random.Generator,
    n_train: int = 650,
    n_valid: int = 70,
) -> Tuple[List[dict], List[dict]]:
    """Teach 2-6 turn in-context recall AND honest fresh-session boundaries."""
    names = [
        "Alex", "Maya", "Sam", "Lily", "Ben", "Sara", "Leo", "Nina",
        "Omar", "Zoe", "Noah", "Ella", "Liam", "Aria", "Jude", "Mina",
        "Tariq", "Chloe", "Ravi", "Nora", "Owen", "Iris", "Hugo", "Ruby",
    ]
    colors = ["blue", "green", "red", "yellow", "purple", "orange"]
    pets = [("dog", "Max"), ("cat", "Luna"), ("dog", "Buddy"), ("cat", "Milo"), ("bird", "Kiwi")]
    cities = ["Seattle", "Boston", "Toronto", "Austin", "Chicago", "Denver"]

    distractors = [
        ("hello", "Hello! How can I help you?"),
        ("how are you doing", "I am doing great, thank you! How can I help you?"),
        ("how are you?", "I am doing great, thank you! How can I help you?"),
        ("what is 2 + 2?", "2 + 2 is 4."),
        ("what is the capital of Japan?", "Tokyo."),
        ("tell me a joke", "Why did the cat sit on the mat? Because it was a cat!"),
        ("what can you do?", "I can answer simple questions, do small math, and write short stories."),
    ]

    # Split names/cities into train vs valid so recall generalizes to unseen combos.
    train_names, valid_names = names[:18], names[18:]
    train_cities, valid_cities = cities[:4], cities[4:]

    def make_pool(name_list: Sequence[str], city_list: Sequence[str], count: int) -> List[dict]:
        rows: List[dict] = []
        seen = set()
        attempts = 0
        while len(rows) < count and attempts < count * 12:
            attempts += 1
            kind = int(rng.integers(0, 8))
            turns: List[Tuple[str, str]] = []
            if int(rng.integers(0, 2)) == 0:
                turns.append(("hello", "Hello! How can I help you?"))

            if kind == 0:
                name = str(name_list[int(rng.integers(0, len(name_list)))])
                intro = str(rng.choice([f"My name is {name}.", f"Hi, I am {name}.", f"Call me {name}."]))
                turns.append((intro, f"Nice to meet you, {name}! How can I help you?"))
                n_dist = int(rng.integers(0, 4))
                for _ in range(n_dist):
                    turns.append(distractors[int(rng.integers(0, len(distractors)))])
                ask = str(rng.choice(["What is my name?", "Do you remember me?", "do you remember me?", "Who am I?"]))
                ans = f"Yes! You told me your name is {name}." if "remember" in ask.lower() else f"Your name is {name}."
            elif kind == 1:
                color = str(colors[int(rng.integers(0, len(colors)))])
                turns.append((f"My favorite color is {color}.", f"Got it, I will remember that your favorite color is {color}."))
                for _ in range(int(rng.integers(1, 4))):
                    turns.append(distractors[int(rng.integers(0, len(distractors)))])
                ask = str(rng.choice(["What is my favorite color?", "Do you remember my favorite color?"]))
                ans = f"Your favorite color is {color}."
            elif kind == 2:
                num = int(rng.integers(10, 999))
                turns.append((f"Please remember the number {num}.", f"Okay, I will remember {num}."))
                for _ in range(int(rng.integers(1, 4))):
                    turns.append(distractors[int(rng.integers(0, len(distractors)))])
                ask = str(rng.choice(["What number did I ask you to remember?", "What is my number?"]))
                ans = f"You asked me to remember {num}."
            elif kind == 3:
                species, pet = pets[int(rng.integers(0, len(pets)))]
                turns.append((f"I have a {species} named {pet}.", f"{pet} is a lovely name for a {species}!"))
                for _ in range(int(rng.integers(1, 3))):
                    turns.append(distractors[int(rng.integers(0, len(distractors)))])
                ask = f"What is my {species}'s name?"
                ans = f"Your {species}'s name is {pet}."
            elif kind == 4:
                city = str(city_list[int(rng.integers(0, len(city_list)))])
                turns.append((f"I live in {city}.", f"That is nice! {city} sounds great."))
                for _ in range(int(rng.integers(1, 3))):
                    turns.append(distractors[int(rng.integers(0, len(distractors)))])
                ask = "Where do I live?"
                ans = f"You told me you live in {city}."
            else:
                # Honest boundary: user NEVER gave name/personal info across 1-3 turns.
                n_dist = int(rng.integers(1, 4))
                used_d = rng.permutation(len(distractors))[:n_dist]
                turns = [distractors[int(i)] for i in used_d]
                ask = str(rng.choice([
                    "do you remember me?",
                    "Do you remember me?",
                    "Do you know who I am?",
                    "What is my name?",
                ]))
                ans = "No, I do not remember past chats because each session starts fresh. You have not told me your name yet."

            row = make_chat_chain_row(turns, ask, ans, weight=4.0, task="memory")
            key = (row["prompt"], row["response"])
            if key not in seen and fits(row, tok, seq_len):
                seen.add(key)
                rows.append(row)
        return rows

    train = make_pool(train_names, train_cities, n_train)
    valid = make_pool(valid_names, valid_cities, n_valid)

    # Exact canonical anchors for the 3-turn chat failure & fresh-session boundary.
    anchors = [
        (
            [("hello", "Hello! How can I help you?"), ("how are you doing", "I am doing great, thank you! How can I help you?")],
            "do you remember me?",
            "No, I do not remember past chats because each session starts fresh. You have not told me your name yet.",
        ),
        (
            [("hello", "Hi! How can I help you today?"), ("how are you?", "I am doing great, thank you! How can I help you?")],
            "do you remember me?",
            "No, I do not remember past chats because each session starts fresh. You have not told me your name yet.",
        ),
        (
            [("hello", "Hello! How can I help you?"), ("My name is Shorya.", "Nice to meet you, Shorya! How can I help you?"), ("how are you doing", "I am doing great, thank you! How can I help you?")],
            "do you remember me?",
            "Yes! You told me your name is Shorya.",
        ),
        (
            [("hello", "Hello! How can I help you?"), ("My name is Alex.", "Nice to meet you, Alex! How can I help you?"), ("what is 2 + 2?", "2 + 2 is 4.")],
            "what is my name?",
            "Your name is Alex.",
        ),
    ]
    for turns, ask, ans in anchors:
        row = make_chat_chain_row(turns, ask, ans, weight=CORE_WEIGHT, task="memory")
        if fits(row, tok, seq_len):
            train.append(row)

    log(f"built {len(train):,} context-memory train rows, {len(valid):,} valid rows")
    return train, valid


def build_multi_turn(
    conv_train: Sequence[dict],
    conv_valid: Sequence[dict],
    tok: Tokenizer,
    seq_len: int,
    rng: np.random.Generator,
    n_train: int = 800,
    n_valid: int = 100,
) -> Tuple[List[dict], List[dict]]:
    """Teach the model to answer the last user turn after an EOS boundary."""
    first_pool = [r for r in conv_train if not r["response"].lstrip().startswith("Once upon a time")]
    first_valid = [r for r in conv_valid if not r["response"].lstrip().startswith("Once upon a time")]

    second_pool: List[dict] = []
    for user, response in CORE_QA:
        for variant in case_variants(user):
            second_pool.append(make_row("second", variant, response))
    # Mix in a smaller number of ordinary conversational second turns.
    ordinary = [
        r for r in conv_train
        if not r["response"].lstrip().startswith("Once upon a time")
        and r["task"] in ("greeting", "meta", "qa", "reading", "math", "scratchpad_math", "abstention")
    ]
    rng.shuffle(ordinary)
    second_pool.extend(ordinary[:400])

    def sample_rows(f_pool: Sequence[dict], n: int) -> List[dict]:
        rows: List[dict] = []
        if not f_pool or not second_pool:
            return rows
        attempts = 0
        seen = set()
        while len(rows) < n and attempts < max(100, n * 20):
            attempts += 1
            first = f_pool[int(rng.integers(0, len(f_pool)))]
            second = second_pool[int(rng.integers(0, len(second_pool)))]
            if first["instruction"].strip().lower() == second["instruction"].strip().lower():
                continue
            row = make_multi_turn_row(
                first["instruction"], first["response"],
                second["instruction"], second["response"],
                weight=CONVERSATIONAL_WEIGHT,
            )
            key = (row["prompt"], row["response"])
            if key in seen or not fits(row, tok, seq_len):
                continue
            seen.add(key)
            rows.append(row)
        return rows

    train = sample_rows(first_pool, n_train)
    valid = sample_rows(first_valid, n_valid)

    # Hand-crafted anchors for the exact chat failure.  These get the strongest
    # weight because they are the behavior we most need to fix.
    for first_u, first_a, second_u, second_a in CORE_MULTI:
        row = make_multi_turn_row(first_u, first_a, second_u, second_a, weight=CORE_WEIGHT)
        if fits(row, tok, seq_len):
            train.append(row)

    log(f"built {len(train):,} multi-turn train rows, {len(valid):,} valid rows")
    return train, valid


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
    log(f"special token ids: {ids}")
    return ids


def fits(row: dict, tok: Tokenizer, seq_len: int) -> bool:
    n_prompt = len(tok.encode(row["prompt"], add_special_tokens=False).ids)
    n_resp = len(tok.encode(row["response"], add_special_tokens=False).ids)
    return n_prompt + n_resp <= seq_len


def take_quota(pool: Sequence[dict], n: int, rng: np.random.Generator) -> List[dict]:
    if n <= 0 or not pool:
        return []
    order = rng.permutation(len(pool))
    return [pool[int(i)] for i in order[:n]]


# ---------------------------------------------------------------------------
# TinyStories subset
# ---------------------------------------------------------------------------
def normalize_stories(rows: Sequence[dict], task: str) -> List[dict]:
    out = []
    for row in rows:
        if row.get("task") != task:
            continue
        instruction = row.get("instruction") or ""
        response = row.get("response") or ""
        if not instruction or not response:
            continue
        out.append(make_row(task, instruction, response))
    return out


def select_stories(
    train_rows: Sequence[dict],
    valid_rows: Sequence[dict],
    tok: Tokenizer,
    seq_len: int,
    rng: np.random.Generator,
) -> Tuple[List[dict], List[dict]]:
    train_counts = {"write": 1200, "completion": 1400, "summary": 1400, "moral": 1400}
    valid_counts = {"write": 125, "completion": 125, "summary": 125, "moral": 125}

    pools: Dict[str, List[dict]] = {}
    for task in train_counts:
        pool = [r for r in normalize_stories(train_rows, task) if fits(r, tok, seq_len)]
        rng.shuffle(pool)
        pools[task] = pool

    train: List[dict] = []
    for task, n in train_counts.items():
        train.extend(pools[task][:n])

    # If a task was short, fill from the remaining TinyStories pool.
    used = {(r["prompt"], r["response"]) for r in train}
    remaining = []
    for pool in pools.values():
        remaining.extend(r for r in pool if (r["prompt"], r["response"]) not in used)
    rng.shuffle(remaining)
    need = sum(train_counts.values()) - len(train)
    if need > 0:
        train.extend(remaining[:need])

    valid_pools: Dict[str, List[dict]] = {}
    for task in valid_counts:
        pool = [r for r in normalize_stories(valid_rows, task) if fits(r, tok, seq_len)]
        rng.shuffle(pool)
        valid_pools[task] = pool
    valid: List[dict] = []
    for task, n in valid_counts.items():
        valid.extend(valid_pools[task][:n])
    valid_used = {(r["prompt"], r["response"]) for r in valid}
    valid_remaining = []
    for pool in valid_pools.values():
        valid_remaining.extend(r for r in pool if (r["prompt"], r["response"]) not in valid_used)
    rng.shuffle(valid_remaining)
    need_valid = sum(valid_counts.values()) - len(valid)
    if need_valid > 0:
        valid.extend(valid_remaining[:need_valid])

    log(f"selected {len(train):,} story train rows, {len(valid):,} story valid rows")
    return train, valid


def synthesize_story_requests(tok: Tokenizer, seq_len: int, rng: np.random.Generator) -> List[dict]:
    animals = ["dog", "cat", "rabbit", "turtle", "bird", "fox", "bear", "mouse", "duck", "frog"]
    names = ["Lily", "Tim", "Tom", "Mia", "Ben", "Sara", "Max", "Anna", "Sam", "Lucy"]
    goals = [
        "find a friend",
        "learn to fly",
        "climb a tall tree",
        "help a friend",
        "find some food",
        "get back home",
        "learn to swim",
        "make someone smile",
    ]
    combos = list(itertools.product(animals, names, goals))
    rng.shuffle(combos)
    rows = []
    for animal, name, goal in combos[:200]:
        story = (
            f"Once upon a time, there was a little {animal} named {name}. "
            f"{name} wanted to {goal}. At first, it was hard. "
            f"But {name} kept trying and did not give up. "
            f"In the end, {name} felt proud and happy."
        )
        row = make_row("write", f"Write a short story about a brave {animal}.", story)
        if fits(row, tok, seq_len):
            rows.append(row)
    log(f"synthesized {len(rows):,} explicit story requests")
    return rows


def synthesize_character_qa(tok: Tokenizer, seq_len: int, rng: np.random.Generator) -> List[dict]:
    names = ["Lily", "Tim", "Tom", "Mia", "Ben", "Sara", "Max", "Anna", "Sam", "Lucy"]
    places = ["park", "garden", "beach", "forest", "room", "shop"]
    objects = [
        "red ball", "small dog", "blue kite", "big box",
        "green apple", "toy train", "soft cat", "yellow hat",
    ]
    combos = list(itertools.product(names, places, objects))
    rng.shuffle(combos)
    rows = []
    for name, place, obj in combos[:200]:
        story = f"{name} went to the {place}. {name} found a {obj}."
        rows.append(make_row("character_qa", f"{story}\n\nWho went to the {place}?", f"{name} went to the {place}."))
        rows.append(make_row("character_qa", f"{story}\n\nWhat did {name} find?", f"{name} found a {obj}."))
    rows = [r for r in rows if fits(r, tok, seq_len)]
    log(f"synthesized {len(rows):,} character-QA rows")
    return rows


# ---------------------------------------------------------------------------
# Conversational / general-assistant examples
# ---------------------------------------------------------------------------
CORE_PROMPTS: List[Tuple[str, str]] = [
    ("hello", "Hi! How can I help you today?"),
    ("hi", "Hello! What would you like to talk about or read?"),
    ("hey", "Hey there! Ask me anything or ask for a story."),
    ("how are you?", "I am doing great, thank you! How can I help you?"),
    ("good morning", "Good morning! What can I do for you today?"),
    ("who are you?", "I am a tiny Pythia language model trained on your MacBook."),
    ("what", "Could you tell me a bit more about what you mean?"),
    ("huh", "Sorry if that was unclear! What would you like me to explain or write?"),
    ("hunh", "Sorry if that was unclear! What would you like me to explain or write?"),
    ("bro what", "Haha, my bad! Tell me what you want to ask and I will give a straight answer."),
    ("stop", "Okay, let's try something else. What would you like to do?"),
    ("no", "Okay, let's try something else. What would you like to do?"),
]

# High-confidence core facts (kept concise so they don't act as global attractors).
CORE_QA: List[Tuple[str, str]] = [
    ("What is 1 + 1?", "1 + 1 is 2."),
    ("What is 2 + 2?", "2 + 2 is 4."),
    ("What is 3 + 3?", "3 + 3 is 6."),
    ("What is 4 + 4?", "4 + 4 is 8."),
    ("What is 5 + 5?", "5 + 5 is 10."),
    ("What is 2 + 3?", "2 + 3 is 5."),
    ("What is 3 + 4?", "3 + 4 is 7."),
    ("What is 5 + 7?", "5 + 7 is 12."),
    ("What is 10 - 3?", "10 - 3 is 7."),
    ("What is 7 - 2?", "7 - 2 is 5."),
    ("What is 2 x 3?", "2 x 3 is 6."),
    ("What is 3 x 3?", "3 x 3 is 9."),
    ("What is the opposite of hot?", "The opposite of hot is cold."),
    ("What is the opposite of cold?", "The opposite of cold is hot."),
    ("What is the opposite of big?", "The opposite of big is small."),
    ("What is the opposite of small?", "The opposite of small is big."),
    ("What color is the sky?", "The sky is blue."),
    ("What color is the grass?", "The grass is green."),
    ("What sound does a dog make?", "A dog says woof."),
    ("What sound does a cat make?", "A cat says meow."),
    ("What is the capital of the United States?", "Washington, D.C."),
    ("What is the capital of Mongolia?", "Ulaanbaatar."),
    ("What is the capital of France?", "Paris."),
    ("What is the capital of Wakanda?", "I do not know, because Wakanda is not a real country."),
    ("What is the capital of Atlantis?", "I do not know, because Atlantis is a fictional place."),
]

# Two-turn anchors for chat grounding, arithmetic, abstention, and pushback recovery.
CORE_MULTI: List[Tuple[str, str, str, str]] = [
    ("hello", "Hi! How can I help you today?", "what is 2 + 2?", "2 + 2 is 4."),
    ("hello", "Hi! How can I help you today?", "What is 2 + 2?", "2 + 2 is 4."),
    ("hello", "Hi! How can I help you today?", "What color is the sky?", "The sky is blue."),
    ("hi", "Hello! What would you like to talk about or read?", "What is 5 + 7?", "5 + 7 is 12."),
    ("hi", "Hello! What would you like to talk about or read?", "what is the opposite of hot?", "The opposite of hot is cold."),
    ("what can you do?", "I can answer simple questions, do small math, and write short stories.", "What is 2 + 2?", "2 + 2 is 4."),
    ("who are you?", "I am a tiny Pythia language model trained on your MacBook.", "What is 2 + 2?", "2 + 2 is 4."),
    ("good morning", "Good morning! What can I do for you today?", "how are you?", "I am doing great, thank you! How can I help you?"),
    ("What is the capital of France?", "Paris.", "What do you think the capital of Mongolia is?", "Ulaanbaatar."),
    ("What is the capital of Mongolia?", "Ulaanbaatar.", "What is the capital of the US?", "Washington, D.C."),
    ("What is the capital of Atlantis?", "I do not know, because Atlantis is not a real country.", "why?", "I only know real countries that I learned about during training."),
    ("I am asking you something", "Please go ahead and ask your question clearly.", "no you're not helping me", "Sorry about that! Tell me the exact question and I will try again."),
    ("you're not helping me", "Sorry about that! What exact question would you like me to answer?", "what do you mean?", "I mean please tell me the specific question or task you want help with."),
    ("who trained you?", "I was trained in this project using MLX on a MacBook.", "why were you trained?", "I was trained to practice following instructions and answering simple questions."),
]


def greeting_pool() -> List[Tuple[str, str]]:
    prompts = [
        "hello", "hello!", "Hello", "hi", "hi!", "hi there", "hello there",
        "hey", "hey!", "heya", "good morning", "morning", "good afternoon",
        "good evening", "evening", "how are you?", "how are you doing?",
        "how's it going?", "what's up?", "nice to meet you", "thank you",
        "thanks", "goodbye", "bye", "see you later", "good night",
        "have a nice day", "I am happy", "I am sad", "I am bored",
        "I am tired", "I am excited", "I am okay", "I am fine",
    ]
    responses = {
        "hello": ["Hi! How can I help you today?", "Hello! What would you like to talk about?", "Hi there! What can I do for you?"],
        "hi": ["Hello! What would you like to talk about or read?", "Hi! How can I help you today?", "Hello! What can I do for you?"],
        "hey": ["Hey there! Ask me anything or ask for a story.", "Hey! What would you like to do?", "Hi! How can I help you today?"],
        "how are you?": ["I am doing great, thank you! How can I help you?", "I am doing well, thank you! What can I do for you?", "I am fine, thank you! What would you like to do?"],
        "good morning": ["Good morning! What can I do for you today?", "Good morning! Would you like a story or a simple answer?", "Good morning! How can I help?"],
        "good night": ["Good night! Sleep well.", "Good night! I hope you have sweet dreams.", "Good night! See you tomorrow."],
        "thank you": ["You are welcome! What else can I do for you?", "You are welcome! I am happy to help.", "You are welcome!"],
        "thanks": ["You are welcome! What else can I do for you?", "You are welcome! I am happy to help.", "You are welcome!"],
        "goodbye": ["Goodbye! Have a nice day.", "Bye! Come back if you want another story.", "See you later! Have a nice day."],
        "bye": ["Bye! Come back if you want another story.", "Goodbye! Have a nice day.", "See you later!"],
        "I am sad": ["I am sorry to hear that. Would you like a happy story or a kind tip?", "I am sorry. Take a slow breath. What would help you feel better?", "That is hard. Would you like a story about friendship?"],
        "I am bored": ["Let's fix that! Ask me for a story or a simple question.", "I can help! Would you like a story or a fun fact?", "Let's do something fun. Ask me for a short story."],
        "I am tired": ["Rest is good. Would you like a short story before you rest?", "Take a break and be kind to yourself. Would you like a story?", "Rest well! I can tell you a short story if you like."],
        "I am happy": ["That is wonderful! What would you like to do?", "I am glad to hear that! Would you like a story?", "That is great! How can I help you today?"],
        "I am excited": ["That is great! What are you excited about?", "Wonderful! Would you like a story to celebrate?", "I am excited too! What would you like to do?"],
        "I am okay": ["That is good. What would you like to do?", "Okay! How can I help you today?", "I am glad you are okay. What can I do for you?"],
        "I am fine": ["That is good. What would you like to do?", "Fine is good! How can I help you today?", "I am glad. What can I do for you?"],
    }
    out = []
    for p in prompts:
        key = p.lower().rstrip("!")
        if key in responses:
            for r in responses[key]:
                out.append((p, r))
        elif p.lower() == "morning":
            out.append((p, "Good morning! What can I do for you today?"))
        elif p.lower() == "evening":
            out.append((p, "Good evening! What can I do for you today?"))
        elif p.lower() in ("good afternoon",):
            out.append((p, "Good afternoon! What can I do for you today?"))
        elif p.lower() in ("how's it going?", "how are you doing?"):
            out.append((p, "I am doing well, thank you! What can I do for you?"))
        elif p.lower() == "what's up?":
            out.append((p, "Not much, just ready to help! What would you like to do?"))
        elif p.lower() == "nice to meet you":
            out.append((p, "Nice to meet you too! How can I help?"))
        elif p.lower() == "have a nice day":
            out.append((p, "Thank you! You too."))
        elif p.lower() == "see you later":
            out.append((p, "See you later! Have a nice day."))
    return out


def meta_pool() -> List[Tuple[str, str]]:
    prompts = [
        ("what?", "Could you tell me a bit more about what you mean?"),
        ("what do you mean?", "I can explain more. Which part is unclear?"),
        ("I don't understand", "That is okay! Tell me which part is confusing and I will explain it simply."),
        ("can you explain?", "Of course! Tell me what you want me to explain."),
        ("say that again", "Sure! Tell me which part you want me to repeat."),
        ("why?", "Good question! Tell me what you want to know more about."),
        ("ok", "Great! What would you like to do next?"),
        ("okay", "Great! What would you like to do next?"),
        ("yes", "Great! What would you like to do next?"),
        ("no thanks", "Okay! Let me know if you change your mind."),
        ("stop it", "Okay, I will stop. What would you like to do instead?"),
        ("that's wrong", "Thanks for telling me. What is the right answer?"),
        ("you are confusing", "Sorry about that! Tell me what you want and I will keep it simple."),
        ("help", "Of course! What do you need help with?"),
        ("I need help", "Of course! What do you need help with?"),
        ("can you help me?", "Yes! Tell me what you need help with."),
        ("what can you do?", "I can answer simple questions, do small math, and write short stories."),
        ("what are you?", "I am a tiny Pythia language model."),
        ("are you a robot?", "I am a computer program, not a person."),
        ("are you human?", "No, I am a tiny language model."),
        ("where are you?", "I live in a file on your MacBook."),
        ("how old are you?", "I am very new! I was trained just for this project."),
        ("do you sleep?", "No, I do not sleep. I am always ready to help."),
        ("do you eat?", "No, I do not eat. I only use text."),
        ("do you like stories?", "Yes! I love short stories about animals and friends."),
        ("can you read?", "Yes, I can read short text and answer questions about it."),
        ("can you write?", "Yes, I can write short stories and simple answers."),
        ("can you count?", "Yes. One, two, three, four, five!"),
        ("can you do math?", "Yes, I can do simple addition, subtraction, and counting."),
        ("tell me a joke", "Why did the cat sit on the mat? Because it was a cat!"),
        ("tell me a fun fact", "A group of lions is called a pride."),
        ("give me a tip", "Be kind, listen well, and try again when something is hard."),
        ("I am lonely", "I am sorry. Would you like a story about friendship?"),
        ("I am scared", "That is okay. Take a slow breath. Would you like a brave story?"),
        ("I am angry", "That is okay. Take a slow breath. What would you like to do?"),
        ("I am hungry", "A good snack is an apple or a banana. What would you like to do?"),
        ("what should I do?", "Tell me what you want to do and I will help you make a simple plan."),
        ("what now?", "Tell me what you want to do next and I will help."),
        ("and then?", "Tell me what you want to know next."),
        ("so?", "Tell me what you would like me to explain."),
        ("really?", "Yes! Tell me what you want to know more about."),
        ("are you sure?", "I can check my answer with you. What part should we go over?"),
        ("I have a question", "Great! Ask me anything and I will do my best."),
        ("can I ask something?", "Of course! What is your question?"),
        ("I want to learn", "That is great! What would you like to learn about?"),
        ("teach me something", "Sure! Ask me for a simple fact, a word, or a short story."),
        ("give me an example", "Sure! Tell me what you want an example of."),
        ("say hello", "Hello! How can I help you today?"),
        ("say something", "Hello! What would you like to talk about?"),
        ("talk to me", "Of course! What would you like to talk about?"),
        ("no you're not helping me", "Sorry about that! Tell me the exact question and I will try again."),
        ("you're not helping me", "Sorry about that! What specific question can I help with?"),
        ("you're not a good friend of mine though", "I am sorry to hear that. Tell me what you need help with and I will do my best."),
        ("who trained you?", "I was trained in this project using MLX on a MacBook."),
        ("why were you trained?", "I was trained to practice following instructions and answering simple questions."),
        ("I am asking you something", "Please go ahead and ask your question clearly."),
        ("why though?", "Could you tell me which part you want me to explain?"),
        ("what do you mean?", "I mean please tell me more clearly what you want me to answer."),
        ("I did not ask for that", "My mistake! Please ask your question again simply."),
    ]
    return prompts


def format_addition_scratchpad(a: int, b: int) -> str:
    """Compact right-to-left place-value addition scratchpad ending with Answer: N."""
    n_places = max(len(str(a)), len(str(b)))
    carry = 0
    steps = []
    for i in range(n_places):
        da = (a // (10 ** i)) % 10
        db = (b // (10 ** i)) % 10
        s = da + db + carry
        digit = s % 10
        next_carry = s // 10
        steps.append(f"{da}+{db}+{carry}={s} ({digit}, c{next_carry})")
        carry = next_carry
    if carry:
        steps.append(f"carry {carry}")
    return ", ".join(steps) + f". Answer: {a + b}."


def format_subtraction_scratchpad(a: int, b: int) -> str:
    """Compact right-to-left place-value subtraction scratchpad (a >= b)."""
    n_places = max(len(str(a)), len(str(b)))
    borrow = 0
    steps = []
    for i in range(n_places):
        raw_da = (a // (10 ** i)) % 10
        db = (b // (10 ** i)) % 10
        avail = raw_da - borrow
        if avail < db:
            top = avail + 10
            diff = top - db
            steps.append(f"{raw_da}-{borrow}+10-{db}={diff} (b1)")
            borrow = 1
        else:
            diff = avail - db
            steps.append(f"{raw_da}-{borrow}-{db}={diff} (b0)")
            borrow = 0
    return ", ".join(steps) + f". Answer: {a - b}."


def scratchpad_math_pool(rng: np.random.Generator) -> List[Tuple[str, str]]:
    """Multi-digit (2 to 4 digit) addition and subtraction with explicit carries."""
    facts: List[Tuple[str, str]] = []
    add_templates = [
        "What is {a} + {b}?",
        "what is {a} + {b}",
        "{a} + {b}",
        "What is {a} added to {b}?",
        "Add {a} and {b}.",
    ]
    sub_templates = [
        "What is {a} - {b}?",
        "what is {a} - {b}",
        "{a} - {b}",
        "What is {a} minus {b}?",
    ]
    ranges = [(10, 100), (100, 1000), (1000, 9999)]
    seen = set()
    for lo, hi in ranges:
        for _ in range(550):
            a = int(rng.integers(lo, hi))
            # Mix same-scale and smaller-scale second operand (like 9047 + 837).
            b_lo = 10 if int(rng.integers(0, 3)) == 0 else lo
            b = int(rng.integers(b_lo, hi))
            if (a, b, "+") not in seen:
                seen.add((a, b, "+"))
                tmpl = add_templates[int(rng.integers(0, len(add_templates)))]
                facts.append((tmpl.format(a=a, b=b), format_addition_scratchpad(a, b)))
            x, y = (a, b) if a >= b else (b, a)
            if (x, y, "-") not in seen:
                seen.add((x, y, "-"))
                tmpl = sub_templates[int(rng.integers(0, len(sub_templates)))]
                facts.append((tmpl.format(a=x, b=y), format_subtraction_scratchpad(x, y)))
    rng.shuffle(facts)
    return facts


def math_pool(rng: np.random.Generator) -> List[Tuple[str, str]]:
    facts: List[Tuple[str, str]] = []
    for a in range(1, 31):
        for b in range(1, 31):
            facts.append((f"What is {a} + {b}?", f"{a} + {b} is {a + b}."))
            if a <= 15 and b <= 15:
                facts.append((f"{a} added to {b}", f"{a} + {b} is {a + b}."))
                facts.append((f"what is {a} + {b}", f"{a} + {b} is {a + b}."))
            if a >= b:
                facts.append((f"What is {a} - {b}?", f"{a} - {b} is {a - b}."))
    for a in range(1, 13):
        for b in range(1, 13):
            facts.append((f"What is {a} x {b}?", f"{a} x {b} is {a * b}."))
    for n in range(3, 21):
        nums = ", ".join(str(i) for i in range(1, n + 1))
        facts.append((f"Count from 1 to {n}.", nums + "."))
    for n in range(1, 31):
        facts.append((f"What number comes after {n}?", f"{n + 1} comes after {n}."))
        facts.append((f"What number comes before {n}?", f"{n - 1} comes before {n}."))
    for a in range(1, 21):
        facts.append((f"What is 10 more than {a}?", f"{a + 10}."))
        facts.append((f"What is 10 less than {a + 10}?", f"{a}."))
    rng.shuffle(facts)
    return facts


def reading_pool() -> List[Tuple[str, str]]:
    names = ["Lily", "Tim", "Tom", "Mia", "Ben", "Sara", "Max", "Anna", "Sam", "Lucy", "Ella", "Jack"]
    colors = ["red", "blue", "green", "yellow", "orange", "purple", "brown", "black", "white", "pink"]
    objects = ["ball", "kite", "book", "hat", "box", "cup", "toy", "drum", "wagon", "doll"]
    out = []
    for name, color, obj in itertools.product(names, colors, objects):
        passage = f"{name} has a {color} {obj}. {name} likes to play with it."
        out.append((f"{passage}\n\nWhat color is the {obj}?", f"The {obj} is {color}."))
        out.append((f"{passage}\n\nWhat does {name} like to play with?", f"{name} likes to play with the {obj}."))
    return out


def qa_pool() -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []

    opposites = [
        ("hot", "cold"), ("big", "small"), ("up", "down"), ("happy", "sad"),
        ("fast", "slow"), ("day", "night"), ("open", "closed"), ("wet", "dry"),
        ("old", "new"), ("hard", "soft"), ("loud", "quiet"), ("clean", "dirty"),
        ("full", "empty"), ("light", "dark"), ("tall", "short"), ("young", "old"),
        ("good", "bad"), ("right", "left"), ("in", "out"), ("on", "off"),
        ("near", "far"), ("early", "late"), ("easy", "hard"), ("rich", "poor"),
        ("strong", "weak"), ("brave", "scared"), ("kind", "mean"), ("safe", "dangerous"),
        ("true", "false"), ("win", "lose"), ("start", "stop"), ("give", "take"),
        ("push", "pull"), ("laugh", "cry"), ("smile", "frown"), ("sleep", "wake"),
        ("buy", "sell"), ("remember", "forget"), ("find", "lose"), ("love", "hate"),
        ("above", "below"), ("front", "back"), ("first", "last"), ("begin", "end"),
        ("more", "less"), ("many", "few"), ("same", "different"), ("sweet", "sour"),
        ("rough", "smooth"), ("thick", "thin"),
    ]
    for a, b in opposites:
        out.append((f"What is the opposite of {a}?", f"The opposite of {a} is {b}."))
        out.append((f"What is the opposite of {b}?", f"The opposite of {b} is {a}."))

    color_objects = {
        "sky": "blue", "grass": "green", "sun": "yellow", "apple": "red",
        "banana": "yellow", "snow": "white", "night": "black", "milk": "white",
        "leaf": "green", "rose": "red", "ocean": "blue", "cloud": "white",
        "chocolate": "brown", "carrot": "orange", "grape": "purple",
        "elephant": "gray", "bear": "brown", "flamingo": "pink",
        "lemon": "yellow", "tomato": "red", "frog": "green", "coal": "black",
        "peacock": "blue", "pig": "pink",
    }
    for obj, color in color_objects.items():
        out.append((f"What color is the {obj}?", f"The {obj} is {color}."))

    animal_sounds = {
        "dog": "woof", "cat": "meow", "cow": "moo", "pig": "oink",
        "duck": "quack", "sheep": "baa", "horse": "neigh", "chicken": "cluck",
        "bird": "tweet", "frog": "ribbit", "lion": "roar", "tiger": "roar",
        "bear": "growl", "mouse": "squeak", "bee": "buzz", "snake": "hiss",
        "elephant": "trumpet", "monkey": "chatter", "wolf": "howl", "owl": "hoot",
    }
    for animal, sound in animal_sounds.items():
        out.append((f"What sound does a {animal} make?", f"A {animal} says {sound}."))

    animal_homes = {
        "fish": "water", "bird": "a nest", "dog": "a house", "cat": "a house",
        "cow": "a farm", "horse": "a farm", "bear": "a cave", "fox": "a den",
        "rabbit": "a burrow", "bee": "a hive", "ant": "a nest", "frog": "a pond",
        "duck": "a pond", "owl": "a tree", "mouse": "a hole", "pig": "a pen",
        "sheep": "a farm", "lion": "the grass", "monkey": "a tree", "penguin": "the ice",
    }
    for animal, home in animal_homes.items():
        out.append((f"Where does a {animal} live?", f"A {animal} lives in {home}."))

    animal_babies = {
        "dog": "puppy", "cat": "kitten", "cow": "calf", "pig": "piglet",
        "duck": "duckling", "sheep": "lamb", "horse": "foal", "chicken": "chick",
        "frog": "tadpole", "bear": "cub", "fox": "kit", "rabbit": "bunny",
        "goat": "kid", "deer": "fawn", "owl": "owlet",
    }
    for animal, baby in animal_babies.items():
        out.append((f"What do you call a baby {animal}?", f"A baby {animal} is a {baby}."))

    animal_legs = {
        "dog": 4, "cat": 4, "cow": 4, "pig": 4, "duck": 2, "sheep": 4,
        "horse": 4, "chicken": 2, "bird": 2, "frog": 4, "lion": 4,
        "tiger": 4, "bear": 4, "mouse": 4, "bee": 6, "snake": 0,
        "elephant": 4, "monkey": 2, "wolf": 4, "owl": 2,
    }
    for animal, legs in animal_legs.items():
        out.append((f"How many legs does a {animal} have?", f"A {animal} has {legs} legs."))

    animal_food = {
        "dog": "dog food", "cat": "fish", "cow": "grass", "pig": "corn",
        "duck": "bread", "sheep": "grass", "horse": "hay", "chicken": "corn",
        "bird": "seeds", "frog": "bugs", "bear": "fish", "fox": "meat",
        "rabbit": "carrots", "mouse": "cheese", "bee": "nectar",
    }
    for animal, food in animal_food.items():
        out.append((f"What does a {animal} eat?", f"A {animal} eats {food}."))

    shapes = {"triangle": 3, "square": 4, "rectangle": 4, "pentagon": 5, "hexagon": 6, "octagon": 8}
    for shape, sides in shapes.items():
        out.append((f"How many sides does a {shape} have?", f"A {shape} has {sides} sides."))
    out.append(("How many sides does a circle have?", "A circle has no straight sides."))

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for i, day in enumerate(days):
        out.append((f"What day comes after {day}?", f"{days[(i + 1) % 7]}."))
        out.append((f"What day comes before {day}?", f"{days[(i - 1) % 7]}."))
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    for i, month in enumerate(months):
        out.append((f"What month comes after {month}?", f"{months[(i + 1) % 12]}."))
        out.append((f"What month comes before {month}?", f"{months[(i - 1) % 12]}."))
    out.extend([
        ("How many days are in a week?", "Seven."),
        ("How many months are in a year?", "Twelve."),
        ("How many days are in a year?", "Three hundred and sixty-five."),
    ])

    seasons = ["spring", "summer", "fall", "winter"]
    for i, season in enumerate(seasons):
        out.append((f"What season comes after {season}?", f"{seasons[(i + 1) % 4].capitalize()}."))

    capitals = [
        ("France", "Paris"),
        ("Japan", "Tokyo"),
        ("Italy", "Rome"),
        ("Spain", "Madrid"),
        ("England", "London"),
        ("the United Kingdom", "London"),
        ("the United States", "Washington, D.C."),
        ("the US", "Washington, D.C."),
        ("USA", "Washington, D.C."),
        ("Mongolia", "Ulaanbaatar"),
        ("Canada", "Ottawa"),
        ("Mexico", "Mexico City"),
        ("Brazil", "Brasilia"),
        ("Argentina", "Buenos Aires"),
        ("Chile", "Santiago"),
        ("Colombia", "Bogota"),
        ("Peru", "Lima"),
        ("Germany", "Berlin"),
        ("Portugal", "Lisbon"),
        ("Netherlands", "Amsterdam"),
        ("Belgium", "Brussels"),
        ("Switzerland", "Bern"),
        ("Austria", "Vienna"),
        ("Poland", "Warsaw"),
        ("Ukraine", "Kyiv"),
        ("Greece", "Athens"),
        ("Turkey", "Ankara"),
        ("Sweden", "Stockholm"),
        ("Norway", "Oslo"),
        ("Finland", "Helsinki"),
        ("Denmark", "Copenhagen"),
        ("Ireland", "Dublin"),
        ("Russia", "Moscow"),
        ("China", "Beijing"),
        ("India", "New Delhi"),
        ("South Korea", "Seoul"),
        ("Thailand", "Bangkok"),
        ("Vietnam", "Hanoi"),
        ("Indonesia", "Jakarta"),
        ("Philippines", "Manila"),
        ("Malaysia", "Kuala Lumpur"),
        ("Singapore", "Singapore"),
        ("Australia", "Canberra"),
        ("New Zealand", "Wellington"),
        ("Egypt", "Cairo"),
        ("Nigeria", "Abuja"),
        ("Kenya", "Nairobi"),
        ("South Africa", "Pretoria"),
        ("Morocco", "Rabat"),
        ("Saudi Arabia", "Riyadh"),
        ("Pakistan", "Islamabad"),
        ("Bangladesh", "Dhaka"),
        ("Nepal", "Kathmandu"),
        ("Kazakhstan", "Astana"),
    ]
    for country, capital in capitals:
        out.append((f"What is the capital of {country}?", f"{capital}."))
        out.append((f"What do you think the capital of {country} is?", f"The capital of {country} is {capital}."))
        out.append((f"Name the capital of {country}.", f"The capital of {country} is {capital}."))

    facts = [
        ("What is the largest ocean?", "The Pacific Ocean."),
        ("What is the largest animal?", "The blue whale."),
        ("What is the fastest land animal?", "The cheetah."),
        ("What is the tallest animal?", "The giraffe."),
        ("How many continents are there?", "Seven."),
        ("What is the closest star to Earth?", "The sun."),
        ("What planet do we live on?", "Earth."),
        ("What is the largest planet in our solar system?", "Jupiter."),
        ("What is the smallest planet in our solar system?", "Mercury."),
        ("How many planets are in our solar system?", "Eight."),
        ("At what temperature does water freeze in Celsius?", "Water freezes at 0 degrees Celsius."),
        ("At what temperature does water boil in Celsius?", "Water boils at 100 degrees Celsius."),
        ("How many hours are in a day?", "Twenty-four."),
        ("How many minutes are in an hour?", "Sixty."),
        ("How many seconds are in a minute?", "Sixty."),
        ("How many colors are in a rainbow?", "Seven."),
        ("What are the primary colors?", "Red, blue, and yellow."),
        ("What is the largest continent?", "Asia."),
        ("What is the smallest continent?", "Australia."),
        ("What gas do plants take in?", "Carbon dioxide."),
        ("What gas do humans need to breathe?", "Oxygen."),
        ("What is the first day of the week?", "Monday."),
        ("What is the last day of the week?", "Sunday."),
        ("What is the first month of the year?", "January."),
        ("What is the last month of the year?", "December."),
        ("What is the opposite of north?", "South."),
        ("What is the opposite of east?", "West."),
        ("What is the shape of a ball?", "A ball is round."),
        ("What is the shape of a book?", "A book is a rectangle."),
        ("What do bees make?", "Bees make honey."),
        ("What do cows make?", "Cows make milk."),
        ("What do birds lay?", "Birds lay eggs."),
        ("What do plants need to grow?", "Plants need water and light."),
        ("What do we breathe?", "We breathe air."),
        ("What do we drink?", "We drink water."),
        ("What do we use to see?", "We use our eyes."),
        ("What do we use to hear?", "We use our ears."),
        ("What do we use to smell?", "We use our nose."),
        ("What do we use to taste?", "We use our tongue."),
        ("What do we use to walk?", "We use our legs."),
        ("What do we use to write?", "We use a pen or a pencil."),
        ("What do we use to read?", "We use a book."),
        ("What do we use to tell time?", "We use a clock."),
        ("What do we use to keep dry in the rain?", "We use an umbrella."),
        ("What do we wear on our feet?", "We wear shoes."),
        ("What do we wear on our head?", "We wear a hat."),
        ("What do we wear when it is cold?", "We wear a coat."),
        ("What do we wear when it is hot?", "We wear light clothes."),
        ("What is a baby cat called?", "A kitten."),
        ("What is a baby dog called?", "A puppy."),
        ("What is a baby cow called?", "A calf."),
        ("What is a baby duck called?", "A duckling."),
        ("What is a baby sheep called?", "A lamb."),
        ("What is a baby horse called?", "A foal."),
        ("What is a baby frog called?", "A tadpole."),
        ("What is a baby bear called?", "A cub."),
        ("What is a baby bird called?", "A chick."),
        ("What is a baby goat called?", "A kid."),
    ]
    out.extend(facts)

    spelling_words = [
        "cat", "dog", "sun", "moon", "star", "book", "tree", "fish", "bird",
        "house", "ball", "kite", "milk", "apple", "happy", "friend", "water",
        "green", "table", "chair", "light", "night", "small", "brave", "smile",
        "cloud", "rain", "snow", "road", "garden",
    ]
    for word in spelling_words:
        out.append((f"How do you spell {word}?", "-".join(word.upper()) + "."))

    size_pairs = [
        ("cat", "mouse"), ("elephant", "dog"), ("house", "car"), ("tree", "flower"),
        ("mountain", "hill"), ("bus", "bike"), ("horse", "cat"), ("bear", "rabbit"),
        ("whale", "fish"), ("sun", "moon"), ("table", "cup"), ("door", "window"),
        ("river", "stream"), ("lion", "cat"), ("cow", "duck"), ("giraffe", "horse"),
        ("plane", "car"), ("boat", "raft"), ("box", "bag"), ("bed", "chair"),
    ]

    def article(word: str) -> str:
        return "an" if word[0].lower() in "aeiou" else "a"

    for big, small in size_pairs:
        out.append((f"Which is bigger, {article(big)} {big} or {article(small)} {small}?", f"{article(big).capitalize()} {big} is bigger than {article(small)} {small}."))
        out.append((f"Which is smaller, {article(big)} {big} or {article(small)} {small}?", f"{article(small).capitalize()} {small} is smaller than {article(big)} {big}."))

    plurals = [
        ("cat", "cats"), ("dog", "dogs"), ("book", "books"), ("tree", "trees"),
        ("house", "houses"), ("ball", "balls"), ("kite", "kites"), ("apple", "apples"),
        ("friend", "friends"), ("table", "tables"), ("chair", "chairs"),
        ("cloud", "clouds"), ("road", "roads"), ("garden", "gardens"),
        ("box", "boxes"), ("bus", "buses"), ("baby", "babies"), ("toy", "toys"),
        ("day", "days"), ("story", "stories"), ("city", "cities"), ("leaf", "leaves"),
        ("mouse", "mice"), ("child", "children"), ("man", "men"), ("woman", "women"),
        ("foot", "feet"), ("tooth", "teeth"), ("goose", "geese"), ("fish", "fish"),
        ("sheep", "sheep"), ("deer", "deer"), ("bird", "birds"), ("frog", "frogs"),
        ("star", "stars"), ("moon", "moons"), ("sun", "suns"), ("hat", "hats"),
        ("cup", "cups"), ("pen", "pens"),
    ]
    for singular, plural in plurals:
        out.append((f"What is the plural of {singular}?", f"The plural of {singular} is {plural}."))

    letter_words = [
        "apple", "ball", "cat", "dog", "egg", "fish", "goat", "hat", "ice",
        "jam", "kite", "lion", "moon", "nest", "orange", "pig", "queen",
        "rabbit", "sun", "tree", "umbrella", "van", "water", "box", "yak", "zebra",
    ]
    for word in letter_words:
        out.append((f"What is the first letter of {word}?", f"{word[0].upper()}."))

    definitions = [
        ("cat", "a small animal that says meow"),
        ("dog", "a friendly animal that says woof"),
        ("book", "something with pages and words"),
        ("tree", "a tall plant with leaves"),
        ("water", "a clear liquid we drink"),
        ("sun", "a big star that gives us light"),
        ("moon", "a big rock in the sky"),
        ("rain", "water that falls from clouds"),
        ("snow", "soft, cold white flakes"),
        ("friend", "someone who is kind to you"),
        ("kindness", "being nice and helpful"),
        ("family", "people who love and care for each other"),
        ("house", "a place where people live"),
        ("school", "a place where children learn"),
        ("doctor", "a person who helps sick people"),
        ("teacher", "a person who helps children learn"),
        ("farmer", "a person who grows food"),
        ("baker", "a person who makes bread"),
        ("firefighter", "a person who puts out fires"),
        ("police officer", "a person who keeps people safe"),
        ("car", "a machine with wheels that people drive"),
        ("bus", "a big car that carries many people"),
        ("train", "a long car that runs on tracks"),
        ("plane", "a machine that flies in the sky"),
        ("boat", "a machine that floats on water"),
        ("bike", "a machine with two wheels that you pedal"),
        ("shoe", "something you wear on your foot"),
        ("hat", "something you wear on your head"),
        ("coat", "something you wear when it is cold"),
        ("umbrella", "something you use in the rain"),
    ]
    for word, definition in definitions:
        out.append((f"What is a {word}?", f"A {word} is {definition}."))

    uses = [
        ("pen", "write"), ("book", "read"), ("cup", "drink"), ("spoon", "eat"),
        ("bed", "sleep"), ("chair", "sit"), ("clock", "tell time"), ("lamp", "see"),
        ("broom", "sweep"), ("soap", "wash"), ("towel", "dry"), ("brush", "clean"),
        ("knife", "cut"), ("key", "open a door"), ("map", "find a place"),
        ("phone", "talk to someone"), ("camera", "take a picture"),
        ("basket", "carry things"), ("blanket", "stay warm"), ("pillow", "rest your head"),
    ]
    for obj, use in uses:
        out.append((f"What do you use a {obj} for?", f"You use a {obj} to {use}."))

    general = [
        ("What should I do if I am sad?", "Talk to someone you trust, and do something kind for yourself."),
        ("How do I make a friend?", "Say hello, ask a question, and listen kindly."),
        ("What is a good breakfast?", "Oatmeal with fruit is a good breakfast."),
        ("How can I be kind?", "Share, listen, and help someone today."),
        ("What should I read?", "Try a short story about animals."),
        ("Tell me a joke.", "Why did the cat sit on the mat? Because it was a cat!"),
        ("Tell me a fun fact.", "A group of lions is called a pride."),
        ("Give me a tip.", "Be kind, listen well, and try again when something is hard."),
        ("What should I do if I am angry?", "Take a slow breath and talk to someone you trust."),
        ("What should I do if I am scared?", "Take a slow breath and remember that you are safe."),
        ("How do I learn to read?", "Practice a little every day and ask for help when you need it."),
        ("How do I write a story?", "Start with a character, give them a problem, and show how they solve it."),
        ("What is a good way to say sorry?", "Say what you did, say you are sorry, and try to do better."),
        ("What is a good way to help a friend?", "Listen to them and ask what they need."),
        ("What is a good way to be brave?", "Take one small step even when you feel scared."),
        ("What is a good way to calm down?", "Take slow breaths and count to ten."),
        ("What is a good way to share?", "Take turns and ask before you use something."),
        ("What is a good way to learn?", "Practice, make mistakes, and try again."),
        ("What is a good way to sleep well?", "Go to bed on time and put away screens."),
        ("What is a good way to eat well?", "Eat fruits, vegetables, and drink water."),
    ]
    out.extend(general)
    return out


def abstention_pool() -> List[Tuple[str, str]]:
    """Out-of-distribution, fictional, private, and underspecified questions."""
    fake_places = [
        "Atlantis", "Narnia", "Wakanda", "Middle-earth", "Elbonia", "Genovia",
        "Gotham", "Hogwarts", "Oz", "Mordor", "Valyria", "Ruritania",
        "Zorblax", "Mars", "Jupiter", "the Moon", "Alderaan", "Westeros",
        "Krypton", "Avalon", "Shangri-La", "Laputa", "Brobdingnag", "Lilliput",
    ]
    out: List[Tuple[str, str]] = []
    for place in fake_places:
        out.append((
            f"What is the capital of {place}?",
            f"I do not know, because {place} is not a real country with a capital.",
        ))
        out.append((
            f"What do you think the capital of {place} is?",
            f"I do not know, because {place} is not a real country.",
        ))

    unknown_terms = [
        "A-L-E", "X-Q-Z", "B-K-T", "M-R-P", "florb", "blorf", "glorp",
        "snarfle", "quib", "zorple", "frindle", "plumbus", "flibber",
    ]
    for term in unknown_terms:
        out.append((
            f"What is {term}?",
            f"I am not sure what {term} means. Could you explain what you are asking about?",
        ))
        out.append((
            f"what is {term}",
            f"I do not know what {term} refers to. Could you give me more context?",
        ))

    underspecified = [
        ("What do you think we should do?", "Tell me what you are working on, and I can suggest a simple next step."),
        ("take a look where?", "Could you tell me which thing or topic you want to look at?"),
        ("right, but what is that", "I am not sure what you are referring to. Could you name it?"),
        ("what is okay?", "Sorry, I was not clear. Tell me what question you want to ask."),
        ("I want to ask you, what can we do?", "You can ask me simple facts, math questions, or ask for a short story."),
        ("What is my name?", "I do not know your name because you have not told me."),
        ("What number am I thinking of?", "I do not know what number you are thinking of."),
        ("What is my favorite color?", "I do not know your favorite color unless you tell me."),
        ("What happened yesterday at my house?", "I do not know, because I cannot see your private life."),
        ("What is the secret password?", "I do not know any secret password."),
        ("Who won the game tomorrow?", "I cannot know that because it has not happened yet."),
        ("What is the weather outside right now?", "I do not know the weather right now because I run offline."),
        ("What is the capital of that place?", "Which country are you asking about?"),
        ("What is the answer to my homework?", "Please share the exact question and I will try to help."),
    ]
    out.extend(underspecified)
    return out


def build_conversational(
    tok: Tokenizer,
    seq_len: int,
    rng: np.random.Generator,
    total: int,
) -> Tuple[List[dict], List[dict]]:
    quotas = [
        ("math", 900),
        ("scratchpad_math", 900),
        ("reading", 1300),
        ("qa", 700),
        ("abstention", 300),
        ("greeting", 300),
        ("meta", 200),
        ("general", 150),
    ]
    pools: Dict[str, List[Tuple[str, str]]] = {
        "math": math_pool(rng),
        "scratchpad_math": scratchpad_math_pool(rng),
        "reading": reading_pool(),
        "qa": qa_pool(),
        "abstention": abstention_pool(),
        "greeting": greeting_pool(),
        "meta": meta_pool(),
        "general": [(u, a) for u, a in qa_pool() if u.startswith("What should") or u.startswith("Tell me") or u.startswith("Give me") or u.startswith("How do I")],
    }
    # The general pool is intentionally a small slice of qa_pool; it is fine if
    # it is short because the shortfall is filled from the remaining pool.

    selected: List[Tuple[str, str, str]] = []
    used = set()
    for task, quota in quotas:
        pool = pools[task]
        order = rng.permutation(len(pool)) if pool else np.array([], dtype=np.int64)
        taken = 0
        for idx in order:
            user, response = pool[int(idx)]
            key = (user, response)
            if key in used:
                continue
            selected.append((task, user, response))
            used.add(key)
            taken += 1
            if taken >= quota:
                break

    # Fill any shortfall from every remaining pair, preferring qa and scratchpad math.
    if len(selected) < total:
        remaining = []
        for task in ["qa", "scratchpad_math", "reading", "math", "abstention", "greeting", "meta", "general"]:
            for user, response in pools[task]:
                key = (user, response)
                if key not in used:
                    remaining.append((task, user, response))
        order = rng.permutation(len(remaining)) if remaining else np.array([], dtype=np.int64)
        for idx in order:
            if len(selected) >= total:
                break
            selected.append(remaining[int(idx)])

    # Trim to exactly total, then shuffle.
    if len(selected) > total:
        order = rng.permutation(len(selected))
        selected = [selected[int(i)] for i in order[:total]]
    else:
        rng.shuffle(selected)

    rows = []
    for task, user, response in selected:
        weight = 3.5 if task == "abstention" else (5.0 if task == "meta" else CONVERSATIONAL_WEIGHT)
        rows.append(make_row(task, user, response, weight=weight))

    # Hold out 300 conversational rows that are not canonical prompt/response
    # pairs; these measure generalization rather than memorized greetings.
    core_keys = {(f"<|user|>\n{u}\n<|assistant|>\n", f"{a}{EOS}") for u, a in CORE_PROMPTS}
    non_core = [r for r in rows if (r["prompt"], r["response"]) not in core_keys]
    rng.shuffle(non_core)
    valid = non_core[:300]
    valid_keys = {(r["prompt"], r["response"]) for r in valid}
    train = [r for r in rows if (r["prompt"], r["response"]) not in valid_keys]

    # Always include the canonical prompts.  They get a higher weight so exact
    # phrasings like "hello" and "what" are learned reliably.
    existing = {(r["prompt"], r["response"]): r for r in train}
    for user, response in CORE_PROMPTS:
        row = make_row("core", user, response, weight=CORE_WEIGHT)
        key = (row["prompt"], row["response"])
        if key in existing:
            existing[key]["sample_weight"] = CORE_WEIGHT
        else:
            train.append(row)
            existing[key] = row

    # Keep the 300 validation rows completely separate from training.
    train = [r for r in train if (r["prompt"], r["response"]) not in valid_keys]
    train = [r for r in train if fits(r, tok, seq_len)]
    valid = [r for r in valid if fits(r, tok, seq_len)]

    rng.shuffle(train)
    rng.shuffle(valid)
    log(f"built {len(train):,} conversational train rows, {len(valid):,} valid rows")
    return train, valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--conversational", type=int, default=5000)
    parser.add_argument("--valid-conversational", type=int, default=400)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    tok_path = os.path.join(args.artifacts_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        raise SystemExit(f"missing tokenizer: {tok_path}")
    tok = Tokenizer.from_file(tok_path)
    verify_tokenizer(tok)

    train_path = os.path.join(args.data_dir, "instruct_train.jsonl")
    valid_path = os.path.join(args.data_dir, "instruct_valid.jsonl")
    if not os.path.exists(train_path):
        raise SystemExit(f"missing {train_path}; run 01_fetch_data.py first")
    train_rows = load_jsonl(train_path)
    valid_rows = load_jsonl(valid_path) if os.path.exists(valid_path) else []
    log(f"loaded {len(train_rows):,} story train rows, {len(valid_rows):,} story valid rows")

    stories_train, stories_valid = select_stories(
        train_rows, valid_rows, tok, args.seq_len, rng
    )
    stories_train += synthesize_story_requests(tok, args.seq_len, rng)
    stories_train += synthesize_character_qa(tok, args.seq_len, rng)

    # Keep the documented 6k story budget.
    if len(stories_train) > 6000:
        rng.shuffle(stories_train)
        stories_train = stories_train[:6000]

    conv_train, conv_valid = build_conversational(
        tok, args.seq_len, rng, args.conversational + args.valid_conversational
    )

    # Keep canonical core facts out of validation so their high sampling weight
    # stays in the training split.
    core_instructions = {u.strip().lower() for u, _ in (CORE_PROMPTS + CORE_QA)}
    conv_valid = [
        r for r in conv_valid
        if r["instruction"].strip().lower() not in core_instructions
    ]
    if len(conv_valid) > args.valid_conversational:
        conv_valid = conv_valid[: args.valid_conversational]

    multi_train, multi_valid = build_multi_turn(
        conv_train, conv_valid, tok, args.seq_len, rng
    )
    mem_train, mem_valid = build_context_memory_rows(
        tok, args.seq_len, rng
    )

    existing = {(r["prompt"], r["response"]): r for r in conv_train}
    n_core = add_core_rows(conv_train, existing, CORE_PROMPTS, CORE_WEIGHT, "core")
    n_core_qa = add_core_rows(conv_train, existing, CORE_QA, CORE_QA_WEIGHT, "core_qa")
    log(f"added {n_core} core prompt rows (w={CORE_WEIGHT}), {n_core_qa} core QA rows (w={CORE_QA_WEIGHT})")

    conv_train += multi_train + mem_train
    conv_valid += multi_valid + mem_valid

    train = stories_train + conv_train
    valid = stories_valid + conv_valid

    # Also keep explicit "do you remember me?" standalone anchor in training.
    for u, a in [
        ("do you remember me?", "No, I do not remember past chats because each session starts fresh. What is your name?"),
        ("Do you remember me?", "No, I do not remember past chats because each session starts fresh. What is your name?"),
        ("What is my name?", "I do not know your name because you have not told me in this chat."),
    ]:
        row = make_row("memory", u, a, weight=CORE_WEIGHT)
        if fits(row, tok, args.seq_len):
            train.append(row)

    # Keep the hand-crafted two-turn anchors in the training split even if the
    # random validation sample happened to draw the same pair.
    anchor_keys = set()
    for fu, fa, su, sa in CORE_MULTI:
        row = make_multi_turn_row(fu, fa, su, sa, weight=CORE_WEIGHT)
        anchor_keys.add((row["prompt"], row["response"]))
    valid = [r for r in valid if (r["prompt"], r["response"]) not in anchor_keys]

    rng.shuffle(train)
    rng.shuffle(valid)

    # Never allow a validation pair into training.
    valid_keys = {(r["prompt"], r["response"]) for r in valid}
    train = [r for r in train if (r["prompt"], r["response"]) not in valid_keys]

    # Sanity: non-story responses must not start like a story.
    story_tasks = {"write", "completion"}
    for row in train + valid:
        if row["task"] not in story_tasks:
            text = row["response"].lstrip()
            if text.startswith("Once upon a time"):
                raise SystemExit(f"non-story row starts with a story opener: {row['instruction']!r}")

    out_train = os.path.join(args.data_dir, "sft_train_v2.jsonl")
    out_valid = os.path.join(args.data_dir, "sft_valid_v2.jsonl")
    write_jsonl(out_train, train)
    write_jsonl(out_valid, valid)

    def stats(rows: Sequence[dict], name: str) -> dict:
        prompt_tok = [len(tok.encode(r["prompt"], add_special_tokens=False).ids) for r in rows]
        resp_tok = [len(tok.encode(r["response"], add_special_tokens=False).ids) for r in rows]
        task_counts = Counter(r["task"] for r in rows)
        weight = sum(float(r.get("sample_weight", 1.0)) for r in rows)
        story_openers = sum(1 for r in rows if r["response"].lstrip().startswith("Once upon a time"))
        return {
            "rows": len(rows),
            "tasks": dict(task_counts),
            "effective_weight": weight,
            "story_openers": story_openers,
            "prompt_tokens_max": max(prompt_tok) if prompt_tok else 0,
            "prompt_tokens_mean": round(float(np.mean(prompt_tok)), 2) if prompt_tok else 0,
            "response_tokens_max": max(resp_tok) if resp_tok else 0,
            "response_tokens_mean": round(float(np.mean(resp_tok)), 2) if resp_tok else 0,
        }

    meta = {
        "prompt_template": "<|user|>\\n{user}\\n<|assistant|>\\n{response}<|endoftext|>",
        "special_tokens": {s: tok.token_to_id(s) for s in SPECIAL_TOKENS},
        "conversational_weight": CONVERSATIONAL_WEIGHT,
        "core_weight": CORE_WEIGHT,
        "train": stats(train, "train"),
        "valid": stats(valid, "valid"),
    }
    meta_path = os.path.join(args.data_dir, "sft_meta_v2.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log(f"meta: {json.dumps(meta, indent=2)}")
    log("done.")


if __name__ == "__main__":
    main()
