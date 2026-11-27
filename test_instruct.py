#!/usr/bin/env python3
"""Non-interactive smoke test for the instruction-tuned checkpoint.

Checks the four prompts that exposed the original "always tells a story" bug:

    hello
    who are you?
    what
    write a short story about a brave cat

Greetings/clarifications must be short and direct; the explicit story request
must still produce a story.  Exit code is nonzero if any check fails.
"""

from __future__ import annotations

import argparse
import sys

from tokenizers import Tokenizer

from model import format_chat_prompt, generate, load_model_from_checkpoint

SPECIAL_TOKENS = ["<|endoftext|>", "<|padding|>", "<|user|>", "<|assistant|>"]


def verify_tokenizer(tok: Tokenizer) -> None:
    for special in SPECIAL_TOKENS:
        idx = tok.token_to_id(special)
        enc = tok.encode(special, add_special_tokens=False).ids
        if idx is None or enc != [idx]:
            raise SystemExit(f"bad special token {special!r}: id={idx}, encode={enc}")


def is_story(text: str) -> bool:
    return text.lstrip().lower().startswith("once upon a time")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/instruct/instruct_final.safetensors")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)
    verify_tokenizer(tok)
    model, config = load_model_from_checkpoint(args.checkpoint, dtype="float32")
    stops = [
        tok.token_to_id("<|endoftext|>"),
        tok.token_to_id("<|user|>"),
        tok.token_to_id("<|assistant|>"),
        tok.token_to_id("<|padding|>"),
    ]

    cases = [
        ("hello", 40, 0.0),
        ("who are you?", 50, 0.0),
        ("what", 40, 0.0),
        ("write a short story about a brave cat", 120, 0.7),
    ]

    results = []
    for user, max_new, temp in cases:
        prompt = format_chat_prompt(user)
        reply = generate(
            model,
            tok,
            prompt,
            max_new_tokens=min(max_new, args.max_new_tokens),
            temperature=temp,
            top_k=40,
            seed=args.seed,
            stop_token_ids=stops,
        ).strip()
        low = reply.lower()
        story = is_story(reply)
        if user == "hello":
            greeted = low.startswith(("hi", "hello", "hey", "good")) or "help you" in low
            ok = bool(reply) and len(reply) <= 120 and not story and greeted
            why = "short greeting" if ok else "expected a short greeting"
        elif user == "who are you?":
            ok = bool(reply) and len(reply) <= 180 and not story and any(
                w in low for w in ("model", "pythia", "language", "computer", "program")
            )
            why = "self-description" if ok else "expected a short self-description"
        elif user == "what":
            ok = bool(reply) and len(reply) <= 180 and not story and any(
                w in low for w in ("mean", "more", "explain", "unclear", "tell me", "question")
            )
            why = "clarification" if ok else "expected a clarification question"
        else:
            ok = bool(reply) and len(reply) >= 40 and story
            why = "story with Once upon a time" if ok else "expected a story starting with Once upon a time"
        results.append((user, reply, ok, why))

    print()
    print("=" * 78)
    print("INSTRUCTION CHECKPOINT SMOKE TEST")
    print("=" * 78)
    print(f"model: {config.num_hidden_layers} layers, hidden={config.hidden_size}, "
          f"vocab={config.vocab_size}")
    print()
    for user, reply, ok, why in results:
        print(f"user> {user}")
        print(f"model> {reply or '<empty>'}")
        print(f"  [{'PASS' if ok else 'FAIL'}] {why}  ({len(reply)} chars)")
        print()

    failures = [r for r in results if not r[2]]
    if failures:
        print(f"FAILED: {len(failures)} of {len(results)} checks")
        sys.exit(1)
    print(f"ALL {len(results)} CHECKS PASSED")


if __name__ == "__main__":
    main()
