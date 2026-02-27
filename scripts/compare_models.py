#!/usr/bin/env python
import argparse
import os
import sys
import time


ROOT = os.path.dirname(os.path.dirname(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from lm.data_io import load_test_data  # noqa: E402
from lm.ngram_model import CharNGramLanguageModel  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two trained checkpoints on the same dev set.")
    parser.add_argument("--work_a", required=True, help="work dir for model A")
    parser.add_argument("--work_b", required=True, help="work dir for model B")
    parser.add_argument("--input", default="eval/dev_input.txt", help="dev input file")
    parser.add_argument("--answer", default="eval/dev_answer.txt", help="dev answer file")
    parser.add_argument("--name_a", default="baseline", help="name for model A")
    parser.add_argument("--name_b", default="candidate", help="name for model B")
    return parser.parse_args()


def read_answers(path):
    with open(path, "rt", encoding="utf-8", errors="ignore") as handle:
        return [line.rstrip("\n") for line in handle]


def evaluate(work_dir, inputs, answers):
    model = CharNGramLanguageModel.load(work_dir)
    t0 = time.perf_counter()
    preds = model.predict_batch(inputs, k=3)
    dt = time.perf_counter() - t0
    correct = sum(ans in pred for pred, ans in zip(preds, answers))
    total = len(answers)
    acc = correct / total if total else 0.0
    ms_per_sample = (dt / total * 1000.0) if total else 0.0
    ckpt = os.path.join(work_dir, "model.checkpoint")
    ckpt_mb = os.path.getsize(ckpt) / (1024 * 1024) if os.path.exists(ckpt) else 0.0
    return {
        "correct": correct,
        "total": total,
        "acc": acc,
        "seconds": dt,
        "ms_per_sample": ms_per_sample,
        "checkpoint_mb": ckpt_mb,
    }


def print_result(name, result):
    print(f"{name}:")
    print(f"  acc: {result['acc']:.4f} ({result['correct']}/{result['total']})")
    print(f"  infer_sec: {result['seconds']:.4f}")
    print(f"  ms_per_sample: {result['ms_per_sample']:.4f}")
    print(f"  checkpoint_mb: {result['checkpoint_mb']:.2f}")


def main():
    args = parse_args()
    inputs = load_test_data(args.input)
    answers = read_answers(args.answer)
    if len(inputs) != len(answers):
        raise SystemExit(f"Mismatched input/answer size: {len(inputs)} vs {len(answers)}")

    res_a = evaluate(args.work_a, inputs, answers)
    res_b = evaluate(args.work_b, inputs, answers)

    print_result(args.name_a, res_a)
    print_result(args.name_b, res_b)

    print("winner_by_acc:", args.name_a if res_a["acc"] >= res_b["acc"] else args.name_b)
    if abs(res_a["acc"] - res_b["acc"]) < 1e-12:
        print("tie_breaker_by_speed:", args.name_a if res_a["seconds"] <= res_b["seconds"] else args.name_b)


if __name__ == "__main__":
    main()
