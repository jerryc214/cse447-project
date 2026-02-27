#!/usr/bin/env python
import argparse
import glob
import os
import random
from typing import List


def parse_args():
    parser = argparse.ArgumentParser(description="Create held-out dev files for next-char prediction.")
    parser.add_argument("--data_glob", default="data/mc4_*.txt", help="glob for corpus text files")
    parser.add_argument("--out_dir", default="eval", help="output directory for dev files")
    parser.add_argument("--size", type=int, default=5000, help="number of dev examples")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--min_len", type=int, default=2, help="minimum line length to keep")
    return parser.parse_args()


def load_lines(paths: List[str], min_len: int) -> List[str]:
    lines = []
    for path in paths:
        try:
            with open(path, "rt", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    text = line.rstrip("\n")
                    if len(text) >= min_len:
                        lines.append(text)
        except OSError:
            continue
    return lines


def main():
    args = parse_args()
    paths = sorted(glob.glob(args.data_glob))
    if not paths:
        raise SystemExit(f"No files matched: {args.data_glob}")

    lines = load_lines(paths, args.min_len)
    if not lines:
        raise SystemExit("No usable lines found in corpus.")

    random.seed(args.seed)
    n = min(args.size, len(lines))
    sample = random.sample(lines, n)

    os.makedirs(args.out_dir, exist_ok=True)
    in_path = os.path.join(args.out_dir, "dev_input.txt")
    ans_path = os.path.join(args.out_dir, "dev_answer.txt")

    with open(in_path, "wt", encoding="utf-8") as in_f, open(ans_path, "wt", encoding="utf-8") as ans_f:
        for line in sample:
            in_f.write(line[:-1] + "\n")
            ans_f.write(line[-1] + "\n")

    print(f"Wrote {n} examples")
    print(f"Input:  {in_path}")
    print(f"Answer: {ans_path}")


if __name__ == "__main__":
    main()
