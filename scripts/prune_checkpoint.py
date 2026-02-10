#!/usr/bin/env python
import argparse
import gzip
import os
import pickle
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(description="Prune a large n-gram checkpoint without retraining.")
    parser.add_argument("--input", required=True, help="path to input checkpoint")
    parser.add_argument("--output", required=True, help="path to output checkpoint")
    parser.add_argument("--min_context_count", type=int, default=5, help="drop contexts below this total count")
    parser.add_argument("--max_contexts", type=int, default=300000, help="keep top-N contexts by total count")
    parser.add_argument(
        "--max_chars_per_context",
        type=int,
        default=48,
        help="keep top-K next-char counts per context",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with gzip.open(args.input, "rb") as handle:
        payload = pickle.load(handle)

    context_counts = payload.get("context_counts", {})

    ranked = []
    for context, counts in context_counts.items():
        if not counts:
            continue
        total = sum(counts.values())
        if total < args.min_context_count:
            continue
        ranked.append((context, total, counts))

    ranked.sort(key=lambda item: item[1], reverse=True)
    ranked = ranked[: args.max_contexts]

    trimmed = {}
    for context, _, counts in ranked:
        top_items = Counter(counts).most_common(args.max_chars_per_context)
        if top_items:
            trimmed[context] = dict(top_items)

    payload["context_counts"] = trimmed
    payload["max_chars_per_context"] = args.max_chars_per_context
    payload["min_context_count"] = args.min_context_count
    payload["max_contexts"] = args.max_contexts

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with gzip.open(args.output, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    in_size = os.path.getsize(args.input)
    out_size = os.path.getsize(args.output)
    print(f"Pruned contexts: {len(trimmed)}")
    print(f"Input size:  {in_size / (1024 * 1024):.2f} MB")
    print(f"Output size: {out_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
