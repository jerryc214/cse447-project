#!/usr/bin/env python
import argparse
import glob
import os
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Create a balanced multilingual dev set (equal per file).")
    parser.add_argument("--data_glob", default="data/mc4_*.txt", help="glob for corpus files")
    parser.add_argument("--out_dir", default="eval2", help="output directory")
    parser.add_argument("--per_file", type=int, default=200, help="examples sampled per file")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--min_len", type=int, default=2, help="minimum line length")
    return parser.parse_args()


def read_valid_lines(path, min_len):
    rows = []
    with open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            text = line.rstrip("\n")
            if len(text) >= min_len:
                rows.append(text)
    return rows


def main():
    args = parse_args()
    files = sorted(glob.glob(args.data_glob))
    if not files:
        raise SystemExit(f"No files matched: {args.data_glob}")

    rng = random.Random(args.seed)
    examples = []
    stats = []

    for path in files:
        try:
            lines = read_valid_lines(path, args.min_len)
        except OSError:
            continue
        if not lines:
            continue
        n = min(args.per_file, len(lines))
        picks = rng.sample(lines, n)
        stats.append((os.path.basename(path), n, len(lines)))
        for line in picks:
            examples.append((line[:-1], line[-1]))

    rng.shuffle(examples)
    os.makedirs(args.out_dir, exist_ok=True)
    in_path = os.path.join(args.out_dir, "dev_input.txt")
    ans_path = os.path.join(args.out_dir, "dev_answer.txt")

    with open(in_path, "wt", encoding="utf-8") as fi, open(ans_path, "wt", encoding="utf-8") as fa:
        for prefix, target in examples:
            fi.write(prefix + "\n")
            fa.write(target + "\n")

    print(f"Files used: {len(stats)}")
    print(f"Total examples: {len(examples)}")
    print(f"Input:  {in_path}")
    print(f"Answer: {ans_path}")

    # Compact summary for quick sanity-check.
    for name, taken, total in stats[:8]:
        print(f"{name}: took {taken}/{total}")
    if len(stats) > 8:
        print(f"... ({len(stats)-8} more files)")


if __name__ == "__main__":
    main()
