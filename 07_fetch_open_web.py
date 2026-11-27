#!/usr/bin/env python3
"""07 — Bounded, model-guided open-knowledge fetcher (Wikipedia / Simple Wikipedia).

Lets the trained model propose curiosity topics (plus topics extracted from its
break-test failures) and explores linked encyclopedic articles up to strict
page and byte caps (hard-capped at 100 MiB; default 10 MiB / 120 pages).

Outputs
-------
data/open_web_train.txt           clean plain-text articles separated by <|endoftext|>
data/open_web_train_tokens.npy    uint16 tokens for new open-knowledge articles
data/open_web_valid_tokens.npy    uint16 held-out validation tokens
data/mixed_train_tokens.npy       balanced mix of old TinyStories + open knowledge
data/mixed_valid_tokens.npy       mixed validation tokens
data/open_web_meta.json           sources, titles, byte & token stats
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from tokenizers import Tokenizer

from model import format_chat_prompt, generate, load_model_from_checkpoint

HARD_CAP_MB = 100.0
ALLOWED_HOSTS = ("en.wikipedia.org", "simple.wikipedia.org")
USER_AGENT = "tiny-pythia-research/1.0 (local-mlx-educational-corpus)"

DEFAULT_SEEDS = [
    "Mongolia",
    "Ulaanbaatar",
    "United States",
    "Washington, D.C.",
    "Canada",
    "Ottawa",
    "Germany",
    "Berlin",
    "Japan",
    "Tokyo",
    "France",
    "Paris",
    "Arithmetic",
    "Addition",
    "Subtraction",
    "Solar System",
    "Earth",
    "Photosynthesis",
    "Water cycle",
    "Continent",
]


def log(msg: str) -> None:
    print(f"[07 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def http_get_json(url: str, timeout: int = 15, delay: float = 0.35) -> Optional[dict]:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.netloc not in ALLOWED_HOSTS:
        raise ValueError(f"blocked disallowed URL host: {parsed.netloc!r}")
    for attempt in range(3):
        if delay > 0:
            time.sleep(delay)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read(2 << 20)  # cap single API payload to 2 MiB
            return json.loads(raw.decode("utf-8", errors="ignore"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 2:
                retry_after = float(exc.headers.get("Retry-After", 2.0 * (attempt + 1)))
                time.sleep(min(retry_after, 8.0))
                continue
            log(f"  request skipped ({parsed.netloc}): {exc}")
            return None
        except Exception as exc:
            log(f"  request skipped ({parsed.netloc}): {exc}")
            return None
    return None


def fetch_wikipedia_batch(
    titles: Sequence[str],
    host: str = "simple.wikipedia.org",
    delay: float = 0.35,
) -> List[Tuple[str, str, List[str]]]:
    """Fetch up to 10 Wikipedia articles in a single polite API call."""
    if not titles:
        return []
    params = urllib.parse.urlencode({
        "action": "query",
        "titles": "|".join(titles[:10]),
        "prop": "extracts|links",
        "explaintext": 1,
        "exsectionformat": "plain",
        "plnamespace": 0,
        "pllimit": 20,
        "redirects": 1,
        "format": "json",
    })
    data = http_get_json(f"https://{host}/w/api.php?{params}", delay=delay)
    if not data:
        return []
    out: List[Tuple[str, str, List[str]]] = []
    pages = data.get("query", {}).get("pages", {})
    for pid, page in pages.items():
        if str(pid) == "-1" or "missing" in page:
            continue
        canon = str(page.get("title", "")).strip()
        extract = str(page.get("extract", "")).strip()
        extract = re.sub(r"\n{3,}", "\n\n", extract)
        for stop_hdr in ("\nReferences\n", "\nSee also\n", "\nOther websites\n", "\nExternal links\n"):
            idx = extract.find(stop_hdr)
            if idx > 200:
                extract = extract[:idx].strip()
        links = [
            str(lk.get("title", "")).strip()
            for lk in page.get("links", [])
            if lk.get("title")
            and not any(bad in str(lk.get("title")) for bad in (":", "List of", "disambiguation"))
        ]
        if canon and len(extract) >= 250:
            out.append((canon, extract, links))
    return out


def search_wikipedia_title(query: str, host: str = "simple.wikipedia.org") -> Optional[str]:
    q = re.sub(r"[^A-Za-z0-9 ,.\-']", " ", query).strip()
    if len(q) < 2:
        return None
    params = urllib.parse.urlencode({
        "action": "query",
        "list": "search",
        "srsearch": q[:80],
        "srlimit": 1,
        "format": "json",
    })
    data = http_get_json(f"https://{host}/w/api.php?{params}")
    if not data:
        return None
    hits = data.get("query", {}).get("search", [])
    if not hits:
        return None
    return str(hits[0].get("title", "")).strip() or None


def fetch_wikipedia_article(
    title: str,
    host: str = "simple.wikipedia.org",
) -> Tuple[Optional[str], Optional[str], List[str]]:
    """Return ``(canonical_title, clean_text, linked_titles)``."""
    params = urllib.parse.urlencode({
        "action": "query",
        "titles": title,
        "prop": "extracts|links",
        "explaintext": 1,
        "exsectionformat": "plain",
        "plnamespace": 0,
        "pllimit": 25,
        "redirects": 1,
        "format": "json",
    })
    data = http_get_json(f"https://{host}/w/api.php?{params}")
    if not data:
        return None, None, []
    pages = data.get("query", {}).get("pages", {})
    for pid, page in pages.items():
        if str(pid) == "-1" or "missing" in page:
            continue
        canon = str(page.get("title", title)).strip()
        extract = str(page.get("extract", "")).strip()
        extract = re.sub(r"\n{3,}", "\n\n", extract)
        # Drop trailing boilerplate sections.
        for stop_hdr in ("\nReferences\n", "\nSee also\n", "\nOther websites\n", "\nExternal links\n"):
            idx = extract.find(stop_hdr)
            if idx > 200:
                extract = extract[:idx].strip()
        links = [
            str(lk.get("title", "")).strip()
            for lk in page.get("links", [])
            if lk.get("title")
            and not any(bad in str(lk.get("title")) for bad in (":", "List of", "disambiguation"))
        ]
        if len(extract) < 250:
            return canon, None, links
        return canon, extract, links
    return None, None, []


def propose_model_topics(
    checkpoint_path: str,
    tokenizer_path: str,
    break_eval_path: str,
    n_samples: int = 10,
) -> List[str]:
    """Let the model + its break-test failures propose topics to explore."""
    topics: List[str] = []

    if os.path.exists(break_eval_path):
        try:
            with open(break_eval_path, "r", encoding="utf-8") as f:
                ev = json.load(f)
            details = ev.get("final", ev).get("details", [])
            for row in details:
                if row.get("passed"):
                    continue
                lbl = str(row.get("label", ""))
                if lbl.startswith("capital:"):
                    topics.append(lbl.split(":", 1)[1].replace("the ", ""))
                elif lbl.startswith("math:"):
                    topics.extend(["Arithmetic", "Addition", "Subtraction"])
        except Exception:
            pass

    if os.path.exists(checkpoint_path) and os.path.exists(tokenizer_path):
        try:
            tok = Tokenizer.from_file(tokenizer_path)
            model, _ = load_model_from_checkpoint(checkpoint_path, dtype="float32")
            stops = [
                tok.token_to_id(s)
                for s in ("<|endoftext|>", "<|user|>", "<|assistant|>", "<|padding|>")
                if tok.token_to_id(s) is not None
            ]
            curiosity_prompts = [
                "Name one country or city in the world.",
                "Name one science topic about nature, space, or animals.",
                "What is a topic in math or geography?",
            ]
            for i in range(n_samples):
                p = format_chat_prompt(curiosity_prompts[i % len(curiosity_prompts)])
                out = generate(
                    model, tok, p, max_new_tokens=20, temperature=0.9, top_k=40,
                    seed=100 + i, stop_token_ids=stops,
                ).strip()
                # Extract clean noun-like candidate words/phrases.
                cleaned = re.sub(r"[^A-Za-z0-9 ,\-]", " ", out).strip()
                words = [w for w in cleaned.split() if len(w) >= 3]
                if words:
                    topics.append(" ".join(words[:3]))
            del model
        except Exception as exc:
            log(f"curiosity generation skipped: {exc}")

    # Deduplicate preserving order.
    seen: Set[str] = set()
    uniq: List[str] = []
    for t in topics + DEFAULT_SEEDS:
        key = t.strip().lower()
        if key and key not in seen:
            seen.add(key)
            uniq.append(t.strip())
    return uniq


def encode_articles_to_npy(tok: Tokenizer, articles: Sequence[str], out_path: str) -> int:
    ids: List[int] = []
    eos = tok.token_to_id("<|endoftext|>")
    for art in articles:
        enc = tok.encode(art.strip(), add_special_tokens=False).ids
        if enc:
            ids.extend(enc)
            ids.append(int(eos))
    arr = np.asarray(ids, dtype=np.uint16)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.save(out_path, arr)
    return int(arr.size)


def build_mixed_npy(
    old_path: str,
    new_path: str,
    out_path: str,
    new_repeat: int = 4,
    max_old_tokens: int = 8_000_000,
    seed: int = 1337,
) -> Tuple[int, int, int]:
    """Mix old TinyStories tokens with new open-web tokens cleanly."""
    rng = np.random.default_rng(seed)
    new_arr = np.load(new_path)
    if os.path.exists(old_path):
        old_mmap = np.load(old_path, mmap_mode="r")
        take_old = min(len(old_mmap), max_old_tokens)
        # Sample contiguous blocks of old text so documents stay intact.
        block = 4096
        n_blocks = max(1, take_old // block)
        starts = rng.integers(0, max(1, len(old_mmap) - block), size=n_blocks)
        old_slice = np.concatenate([np.asarray(old_mmap[s : s + block], dtype=np.uint16) for s in starts])
    else:
        old_slice = np.empty((0,), dtype=np.uint16)

    new_rep = np.tile(new_arr, max(1, new_repeat)).astype(np.uint16, copy=False)
    combined = np.concatenate([old_slice, new_rep])
    # Shuffle in 2048-token chunks so old and new texts interleave evenly.
    chunk = 2048
    n_chunks = len(combined) // chunk
    if n_chunks > 1:
        trimmed = combined[: n_chunks * chunk].reshape(n_chunks, chunk)
        perm = rng.permutation(n_chunks)
        mixed = trimmed[perm].reshape(-1)
    else:
        mixed = combined
    np.save(out_path, mixed.astype(np.uint16, copy=False))
    return int(mixed.size), int(old_slice.size), int(new_rep.size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--checkpoint", default="checkpoints/rl/rl_final.safetensors")
    parser.add_argument("--max-mb", type=float, default=5.0, help="soft cap in MiB (hard-capped at 100 MiB)")
    parser.add_argument("--max-pages", type=int, default=120)
    parser.add_argument("--delay", type=float, default=0.35, help="seconds between API calls")
    parser.add_argument("--host", choices=list(ALLOWED_HOSTS), default="simple.wikipedia.org")
    parser.add_argument("--fallback-host", choices=list(ALLOWED_HOSTS), default="en.wikipedia.org")
    parser.add_argument("--new-repeat", type=int, default=6, help="upsample factor for open-web tokens in mixed_train_tokens.npy")
    args = parser.parse_args()

    max_mb = min(float(args.max_mb), HARD_CAP_MB)
    max_bytes = int(max_mb * (1 << 20))
    tok_path = os.path.join(args.artifacts_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        raise SystemExit(f"missing tokenizer: {tok_path}")
    tok = Tokenizer.from_file(tok_path)

    ckpt = args.checkpoint
    if not os.path.exists(ckpt):
        ckpt = "checkpoints/instruct/instruct_final.safetensors"

    seeds = propose_model_topics(
        ckpt,
        tok_path,
        os.path.join(args.artifacts_dir, "rl_break_eval.json"),
    )
    log(f"curiosity + failure seeds ({len(seeds)}): {seeds[:10]}")

    queue: deque[str] = deque(seeds)
    visited: Set[str] = set()
    seen_hashes: Set[str] = set()
    articles: List[str] = []
    titles: List[str] = []
    total_bytes = 0

    while queue and len(articles) < args.max_pages and total_bytes < max_bytes:
        batch: List[str] = []
        while queue and len(batch) < 8:
            raw_topic = queue.popleft()
            key = raw_topic.lower().strip()
            if key and key not in visited:
                visited.add(key)
                batch.append(raw_topic.strip())
        if not batch:
            break

        fetched = fetch_wikipedia_batch(batch, host=args.host, delay=args.delay)
        if not fetched and args.fallback_host != args.host:
            fetched = fetch_wikipedia_batch(batch, host=args.fallback_host, delay=args.delay)

        for canon, text, links in fetched:
            if len(articles) >= args.max_pages or total_bytes >= max_bytes:
                break
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest not in seen_hashes:
                seen_hashes.add(digest)
                formatted = f"{canon}\n\n{text}"
                b_len = len(formatted.encode("utf-8"))
                if total_bytes + b_len <= max_bytes or not articles:
                    articles.append(formatted)
                    titles.append(canon)
                    total_bytes += b_len
                    for lk in links[:6]:
                        if lk.lower() not in visited and len(queue) < args.max_pages * 4:
                            queue.append(lk)
                    if len(articles) % 15 == 0:
                        log(f"  fetched {len(articles)}/{args.max_pages} articles ({total_bytes / 1024:.1f} KiB)")

    if not articles:
        raise SystemExit("no open-web articles could be fetched (check network connection)")

    os.makedirs(args.data_dir, exist_ok=True)
    txt_path = os.path.join(args.data_dir, "open_web_train.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n<|endoftext|>\n".join(articles) + "\n<|endoftext|>\n")

    split = max(1, int(0.9 * len(articles)))
    if split == len(articles) and len(articles) > 1:
        split -= 1
    train_arts = articles[:split]
    valid_arts = articles[split:] if split < len(articles) else articles[-1:]

    ow_train_path = os.path.join(args.data_dir, "open_web_train_tokens.npy")
    ow_valid_path = os.path.join(args.data_dir, "open_web_valid_tokens.npy")
    n_ow_train = encode_articles_to_npy(tok, train_arts, ow_train_path)
    n_ow_valid = encode_articles_to_npy(tok, valid_arts, ow_valid_path)

    mixed_train_path = os.path.join(args.data_dir, "mixed_train_tokens.npy")
    mixed_valid_path = os.path.join(args.data_dir, "mixed_valid_tokens.npy")
    n_mix_train, n_old_t, n_new_t = build_mixed_npy(
        os.path.join(args.data_dir, "train_tokens.npy"),
        ow_train_path,
        mixed_train_path,
        new_repeat=args.new_repeat,
    )
    n_mix_valid, n_old_v, n_new_v = build_mixed_npy(
        os.path.join(args.data_dir, "valid_tokens.npy"),
        ow_valid_path,
        mixed_valid_path,
        new_repeat=1,
        max_old_tokens=500_000,
    )

    meta = {
        "hosts": [args.host, args.fallback_host],
        "hard_cap_mb": HARD_CAP_MB,
        "articles": len(articles),
        "bytes": total_bytes,
        "open_web_train_tokens": n_ow_train,
        "open_web_valid_tokens": n_ow_valid,
        "mixed_train_tokens": n_mix_train,
        "mixed_train_old": n_old_t,
        "mixed_train_new_upsampled": n_new_t,
        "mixed_valid_tokens": n_mix_valid,
        "sample_titles": titles[:25],
    }
    meta_path = os.path.join(args.data_dir, "open_web_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    log(
        f"done: {len(articles)} articles ({total_bytes / 1024:.1f} KiB, "
        f"{n_ow_train:,} train tok, {n_ow_valid:,} valid tok); "
        f"mixed train={n_mix_train:,} tok -> {meta_path}"
    )


if __name__ == "__main__":
    main()
