#!/usr/bin/env python3
"""01 — Download TinyStories and synthesize instruction data.

No Hugging Face libraries are used.  The two plain-text files are fetched with
``urllib`` (following the CDN redirect) and the pretraining corpus is truncated
at a story boundary.

Outputs
-------
data/pretrain_train.txt      clean ~150 MB story corpus
data/pretrain_valid.txt      full TinyStoriesV2 validation corpus
data/instruct_train.jsonl    ~15k synthesized instruction examples
data/instruct_valid.jsonl    held-out instruction examples
data/stories_meta.json       corpus statistics
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.request
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

BASE_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
TRAIN_URL = f"{BASE_URL}/TinyStoriesV2-GPT4-train.txt"
VALID_URL = f"{BASE_URL}/TinyStoriesV2-GPT4-valid.txt"

SPECIAL_TOKENS = ["<|endoftext|>", "<|padding|>", "<|user|>", "<|assistant|>"]

# ---------------------------------------------------------------------------
# Small heuristics for turning narratives into instruction/response pairs
# ---------------------------------------------------------------------------

THEMES: Dict[str, Sequence[str]] = {
    "friendship": ("friend", "friends", "friendship", "playmate"),
    "kindness": ("kind", "kindness", "help", "helpful", "caring"),
    "honesty": ("honest", "honesty", "truth", "truthful", "lie", "lied"),
    "bravery": ("brave", "bravery", "courage", "courageous", "scared", "fear"),
    "sharing": ("share", "shared", "sharing"),
    "patience": ("patient", "patience", "wait", "waited"),
    "hard work": ("worked hard", "hard work", "practice", "practiced", "tried"),
    "family": ("mom", "mother", "dad", "father", "family", "sister", "brother", "grandma", "grandpa"),
    "animals": ("cat", "dog", "bird", "rabbit", "bear", "fox", "mouse", "frog", "duck", "horse"),
    "magic": ("magic", "magical", "wand", "fairy", "wizard"),
    "dragons": ("dragon", "dragons"),
    "princesses": ("princess", "prince", "king", "queen", "castle"),
    "robots": ("robot", "robots"),
    "school": ("school", "teacher", "classroom", "class"),
    "nature": ("forest", "tree", "river", "mountain", "garden", "flower"),
    "food": ("cake", "cookie", "bread", "apple", "soup", "pancake"),
}

# Short, sensible one-line morals keyed by detected theme.
MORAL_TEMPLATES: Dict[str, str] = {
    "friendship": "It is important to be a good friend.",
    "kindness": "Always be kind and help others.",
    "honesty": "Always tell the truth, even when it is hard.",
    "bravery": "It is okay to be scared, but we should still try to be brave.",
    "sharing": "Sharing with others makes everyone happier.",
    "patience": "Good things come to those who are patient.",
    "hard work": "Hard work and practice help us improve.",
    "family": "Family and loved ones are the most important things.",
    "animals": "We should treat animals with care and kindness.",
    "magic": "Magic is best when it is used to help others.",
    "dragons": "Even a scary dragon can become a good friend.",
    "princesses": "Being kind is more important than being a princess.",
    "robots": "A robot can learn to be a good friend.",
    "school": "We can learn something new every day.",
    "nature": "We should take care of the world around us.",
    "food": "Sharing food with others is a kind thing to do.",
}

NAME_STOPWORDS = {
    "The", "Once", "One", "When", "Then", "And", "But", "She", "He", "They",
    "It", "This", "That", "There", "After", "Before", "Finally", "Suddenly",
    "Every", "Little", "His", "Her", "Their", "Its", "As", "At", "In", "On",
    "For", "With", "So", "Now", "Soon", "Later", "First", "Next", "Last",
    "However", "Although", "While", "Because", "If", "Not", "No", "Yes",
    "But", "Or", "A", "An", "My", "We", "You", "I", "Do", "Did", "Was",
    "Were", "Is", "Are", "Have", "Has", "Had", "Can", "Could", "Would",
    "Should", "Will", "Just", "Very", "Too", "All", "Some", "Many", "One",
    "Two", "Three", "Today", "Yesterday", "Tomorrow", "Dear", "Hello",
}

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def log(msg: str) -> None:
    print(f"[01 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def download_limited(url: str, path: str, max_bytes: int, timeout: int = 180) -> int:
    """Download at most ``max_bytes`` of ``url`` to ``path`` (with retries)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".part"
    last_error: Optional[Exception] = None
    for attempt in range(1, 4):
        total = 0
        t0 = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (tiny-pythia)"})
            with urllib.request.urlopen(req, timeout=timeout) as response, open(tmp, "wb") as f:
                while total < max_bytes:
                    chunk = response.read(min(1 << 20, max_bytes - total))
                    if not chunk:
                        break
                    f.write(chunk)
                    total += len(chunk)
                    if total % (16 << 20) < (1 << 20):
                        log(f"  downloaded {total / (1 << 20):.1f} MiB")
            if total == 0:
                raise IOError("empty response")
            os.replace(tmp, path)
            log(f"  saved {path} ({total / (1 << 20):.1f} MiB in {time.time() - t0:.1f}s)")
            return total
        except Exception as exc:  # network hiccups / expiring CDN URLs
            last_error = exc
            log(f"  download attempt {attempt}/3 failed: {exc}")
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            time.sleep(2 * attempt)
    raise RuntimeError(f"failed to download {url}: {last_error}")


def trim_to_story_boundary(path: str) -> int:
    with open(path, "rb") as f:
        data = f.read()
    marker = b"<|endoftext|>"
    idx = data.rfind(marker)
    if idx >= 0:
        data = data[: idx + len(marker)]
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def split_stories(text: str) -> List[str]:
    return [s.strip() for s in text.split("<|endoftext|>") if s.strip()]


def clean_story(story: str) -> Optional[str]:
    story = story.replace("\r\n", "\n").replace("\r", "\n")
    story = re.sub(r"<\|[^|>]*\|>", "", story)
    story = re.sub(r"\n{3,}", "\n\n", story)
    story = story.strip()
    if len(story) < 220 or len(story) > 1600:
        return None
    if not re.search(r"[A-Za-z]", story):
        return None
    sentences = SENTENCE_RE.split(story)
    if len(sentences) < 3:
        return None
    return story


def detect_theme(story: str) -> Optional[str]:
    low = story.lower()
    for theme, keywords in THEMES.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", low):
                return theme
    return None


def detect_keyword(story: str) -> Optional[str]:
    """Return a concrete noun-like keyword that appears in the story."""
    low = story.lower()
    for theme, keywords in THEMES.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", low):
                return kw
    return None


def main_character(story: str) -> str:
    counts = Counter(
        w for w in re.findall(r"\b([A-Z][a-z]{2,})\b", story) if w not in NAME_STOPWORDS
    )
    if counts:
        return counts.most_common(1)[0][0]
    return "the main character"


def last_sentence(story: str) -> str:
    parts = [p.strip() for p in SENTENCE_RE.split(story.strip()) if p.strip()]
    if not parts:
        return ""
    s = parts[-1]
    if s and s[-1] not in ".!?":
        s += "."
    return s


def split_story_for_completion(story: str) -> Optional[Tuple[str, str]]:
    sentences = [s.strip() for s in SENTENCE_RE.split(story) if s.strip()]
    if len(sentences) < 3:
        return None
    total = sum(len(s) for s in sentences)
    acc = 0
    split_at = None
    for k in range(1, len(sentences)):
        acc += len(sentences[k - 1])
        ratio = acc / max(total, 1)
        if 0.35 <= ratio <= 0.68:
            split_at = k
            break
    if split_at is None:
        split_at = max(1, len(sentences) // 2)
    opening = " ".join(sentences[:split_at]).strip()
    rest = " ".join(sentences[split_at:]).strip()
    if len(opening) < 80 or len(rest) < 80:
        return None
    return opening, rest


def make_example(story: str, task: str, rng: random.Random) -> Optional[Tuple[str, str]]:
    char = main_character(story)
    theme = detect_theme(story) or "kindness"

    if task == "completion":
        split = split_story_for_completion(story)
        if split is None:
            return None
        opening, rest = split
        sad_ending = re.search(
            r"\b(sad|cry|cried|tears|never|died|gone|alone|lost|missed)\b",
            story[-260:],
            flags=re.IGNORECASE,
        )
        if sad_ending:
            template = rng.choice(
                [
                    "Continue this story:\n\n{opening}",
                    "Finish the story:\n\n{opening}",
                    "Complete the story:\n\n{opening}",
                ]
            )
        else:
            template = rng.choice(
                [
                    "Continue this story with a happy ending:\n\n{opening}",
                    "Finish this story:\n\n{opening}",
                    "Complete the story in a satisfying way:\n\n{opening}",
                ]
            )
        return template.format(opening=opening), rest

    if task == "write":
        keyword = detect_keyword(story)
        if keyword is None:
            return None
        template = rng.choice(
            [
                "Write a short story about {kw}.",
                "Tell me a story about {kw}.",
                "Can you write a story about {kw}?",
            ]
        )
        return template.format(kw=keyword), story

    if task == "moral":
        lesson = last_sentence(story)
        looks_like_moral = bool(
            re.search(
                r"\b(always|never|important|learn|learned|remember|should|kind|brave|honest|share)\b",
                lesson,
                flags=re.IGNORECASE,
            )
        )
        if not looks_like_moral or not (15 <= len(lesson) <= 220):
            lesson = MORAL_TEMPLATES.get(theme, f"Always remember the importance of {theme}.")
        template = rng.choice(
            [
                "What moral or lesson does this story teach?\n\n{story}",
                "What is the lesson of this story?\n\n{story}",
                "What can we learn from this story?\n\n{story}",
            ]
        )
        return template.format(story=story), lesson

    if task == "summary":
        template = rng.choice(
            [
                "Summarize the following story in one sentence:\n\n{story}",
                "Write a one-sentence summary of this story:\n\n{story}",
                "Give a one-sentence summary:\n\n{story}",
            ]
        )
        summary_choices = [
            f"{char} learns an important lesson about {theme}.",
            f"This is a story about {char} and the importance of {theme}.",
            f"{char} discovers that {theme} can make a big difference.",
        ]
        return template.format(story=story), rng.choice(summary_choices)

    raise ValueError(task)


def build_instruction_examples(
    stories: Sequence[str],
    target: int,
    rng: random.Random,
    start_task: int = 0,
    desc: str = "instruct",
) -> List[dict]:
    tasks = ["completion", "write", "moral", "summary"]
    examples: List[dict] = []
    seen = set()
    task_idx = start_task
    for story in stories:
        if len(examples) >= target:
            break
        # Try the next task, then fall back to the others if it is not applicable.
        for offset in range(len(tasks)):
            task = tasks[(task_idx + offset) % len(tasks)]
            made = make_example(story, task, rng)
            if made is None:
                continue
            instruction, response = made
            key = hashlib.blake2b((instruction + "\x00" + response).encode("utf-8"), digest_size=16).digest()
            if key in seen:
                continue
            seen.add(key)
            prompt = f"<|user|>\n{instruction.strip()}\n<|assistant|>\n"
            response = response.strip()
            text = prompt + response + "<|endoftext|>"
            examples.append(
                {
                    "task": task,
                    "instruction": instruction.strip(),
                    "prompt": prompt,
                    "response": response + "<|endoftext|>",
                    "text": text,
                }
            )
            task_idx = (task_idx + offset + 1) % len(tasks)
            break
        if len(examples) % 1000 == 0 and len(examples):
            log(f"  {desc}: {len(examples)}/{target}")
    rng.shuffle(examples)
    return examples


def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--train-mb", type=int, default=150, help="pretraining corpus size in MiB")
    parser.add_argument("--num-instruct", type=int, default=15000)
    parser.add_argument("--num-valid-instruct", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    train_path = os.path.join(args.data_dir, "pretrain_train.txt")
    valid_path = os.path.join(args.data_dir, "pretrain_valid.txt")

    target_bytes = args.train_mb << 20
    need_train = args.force or not os.path.exists(train_path) or os.path.getsize(train_path) < max(1 << 20, int(target_bytes * 0.85))
    if need_train:
        log(f"downloading first {args.train_mb} MiB of TinyStoriesV2-GPT4-train.txt ...")
        download_limited(TRAIN_URL, train_path, target_bytes)
        n = trim_to_story_boundary(train_path)
        log(f"trimmed corpus to {n / (1 << 20):.1f} MiB at a story boundary")
    else:
        log(f"using existing {train_path} ({os.path.getsize(train_path) / (1 << 20):.1f} MiB)")

    if args.force or not os.path.exists(valid_path) or os.path.getsize(valid_path) < 1 << 20:
        log("downloading TinyStoriesV2-GPT4-valid.txt ...")
        download_limited(VALID_URL, valid_path, 1 << 40)
    else:
        log(f"using existing {valid_path}")

    log("parsing stories ...")
    train_stories_raw = split_stories(read_text(train_path))
    valid_stories_raw = split_stories(read_text(valid_path))
    train_stories = [s for s in (clean_story(x) for x in train_stories_raw) if s]
    valid_stories = [s for s in (clean_story(x) for x in valid_stories_raw) if s]
    log(f"  clean stories: train={len(train_stories):,} valid={len(valid_stories):,}")

    rng = random.Random(args.seed)

    # Use a manageable, high-quality slice for instruction synthesis.
    max_train_stories = min(len(train_stories), max(args.num_instruct, 8000))
    max_valid_stories = min(len(valid_stories), max(1000, args.num_valid_instruct * 4))
    train_pool = train_stories[:max_train_stories]
    valid_pool = valid_stories[:max_valid_stories]

    log(f"synthesizing {args.num_instruct:,} training instructions ...")
    train_examples = build_instruction_examples(train_pool, args.num_instruct, rng, 0, "train")
    log(f"synthesizing {args.num_valid_instruct:,} validation instructions ...")
    valid_examples = build_instruction_examples(valid_pool, args.num_valid_instruct, rng, 2, "valid")

    train_jsonl = os.path.join(args.data_dir, "instruct_train.jsonl")
    valid_jsonl = os.path.join(args.data_dir, "instruct_valid.jsonl")
    n_train = write_jsonl(train_jsonl, train_examples)
    n_valid = write_jsonl(valid_jsonl, valid_examples)

    train_chars = os.path.getsize(train_path)
    valid_chars = os.path.getsize(valid_path)
    meta = {
        "train_stories": len(train_stories),
        "valid_stories": len(valid_stories),
        "train_chars": train_chars,
        "valid_chars": valid_chars,
        "instruct_train": n_train,
        "instruct_valid": n_valid,
        "task_counts": dict(Counter(e["task"] for e in train_examples)),
        "special_tokens": SPECIAL_TOKENS,
    }
    with open(os.path.join(args.data_dir, "stories_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log(f"wrote {train_jsonl} ({n_train:,} examples)")
    log(f"wrote {valid_jsonl} ({n_valid:,} examples)")
    log(f"task mix: {meta['task_counts']}")
    log("done.")


if __name__ == "__main__":
    main()
