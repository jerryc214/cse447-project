#!/usr/bin/env python
"""
Efficient hyperparameter tuning for CharNGramLanguageModel (KN version).

ngram_order별로 한 번만 훈련하고,
나머지 파라미터(kn_discount, laplace_alpha, min_context_count, max_contexts)는
같은 raw counts에서 재계산만 함 → 훈련 횟수 최소화.

실행 (프로젝트 루트에서):
    python src/tune.py
"""

import copy
import itertools
import os
import random
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from lm.ngram_model import CharNGramLanguageModel
from lm.data_io import load_text_lines, resolve_training_files
from lm.text_utils import normalize_text

PARAM_GRID = {
    "ngram_order":       [5, 6, 7, 8],       
    "kn_discount":       [0.5, 0.75, 0.9],  
    "laplace_alpha":     [0.01, 0.1, 1.0], 
    "min_context_count": [2, 3, 5],   
    "max_contexts":      [300000, 500000, 1000000],  
}
FIXED = {
    "max_chars_per_context": 48,
}

DEV_SIZE       = 1000
DEV_SEED       = 42
RESULT_FILE    = "tune_results.txt"
EXAMPLE_INPUT  = "example/input.txt"
EXAMPLE_ANSWER = "example/answer.txt"
# ────────────────────────────────────────────────────────────────


def load_example():
    with open(EXAMPLE_INPUT, encoding="utf-8") as f:
        inputs = [normalize_text(l.rstrip("\n")) for l in f]
    with open(EXAMPLE_ANSWER, encoding="utf-8") as f:
        answers = [normalize_text(l.rstrip("\n")) for l in f]
    return inputs, answers


def make_dev_set(dev_size, seed):
    random.seed(seed)
    files = resolve_training_files()
    all_lines = []
    for f in files:
        try:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line = normalize_text(line.rstrip("\n"))
                    if len(line) >= 2:
                        all_lines.append(line)
        except OSError:
            continue
    sample = random.sample(all_lines, min(dev_size, len(all_lines)))
    return [l[:-1] for l in sample], [l[-1] for l in sample]


def evaluate(model, inputs, answers):
    preds = model.predict_batch(inputs, k=3)
    correct = sum(ans in pred for pred, ans in zip(preds, answers))
    return correct, len(answers)


def apply_trim_params(base_model, min_context_count, max_contexts):
    """
    base_model의 raw context_counts를 복사해서
    min_context_count, max_contexts만 바꿔 trim 재적용.
    재훈련 없이 새 model 반환.
    """
    m = copy.copy(base_model)
    m.context_counts = copy.deepcopy(base_model.raw_context_counts)
    m.min_context_count = min_context_count
    m.max_contexts = max_contexts
    m._trim_context_tables()
    m._refresh_default_chars()
    return m


def main():
    total_combos = 1
    for v in PARAM_GRID.values():
        total_combos *= len(v)

    print("[1/3] Loading training data...")
    t0 = time.time()
    train_data = load_text_lines(resolve_training_files())
    print(f"      {len(train_data)} lines  ({time.time()-t0:.1f}s)")

    print("[2/3] Loading example + dev set...")
    ex_inputs, ex_answers = load_example()
    dev_inputs, dev_answers = make_dev_set(DEV_SIZE, DEV_SEED)
    print(f"      example={len(ex_inputs)}  dev={len(dev_inputs)}")

    print(f"\n[3/3] Grid search start\n")

    results = []
    done = 0
    t_start = time.time()

    for ngram_order in PARAM_GRID["ngram_order"]:

        print(f"  [Training] ngram_order={ngram_order} ...", end=" ", flush=True)
        t0 = time.time()
        base_model = CharNGramLanguageModel(
            ngram_order=ngram_order,
            **FIXED,
        )
        base_model.fit(train_data)
        base_model.raw_context_counts = copy.deepcopy(base_model.context_counts)
        print(f"완료 ({time.time()-t0:.1f}s)")

        trim_combos = list(itertools.product(
            PARAM_GRID["min_context_count"],
            PARAM_GRID["max_contexts"],
        ))

        for min_cc, max_ctx in trim_combos:
            m = apply_trim_params(base_model, min_cc, max_ctx)

            for kn_d, lap_a in itertools.product(
                PARAM_GRID["kn_discount"],
                PARAM_GRID["laplace_alpha"],
            ):
                m.kn_discount   = kn_d
                m.laplace_alpha = lap_a

                ex_correct,  ex_total  = evaluate(m, ex_inputs,  ex_answers)
                dev_correct, dev_total = evaluate(m, dev_inputs, dev_answers)
                done += 1

                params = {
                    "ngram_order":       ngram_order,
                    "kn_discount":       kn_d,
                    "laplace_alpha":     lap_a,
                    "min_context_count": min_cc,
                    "max_contexts":      max_ctx,
                }
                r = {
                    "params":  params,
                    "ex":      (ex_correct, ex_total),
                    "dev":     (dev_correct, dev_total),
                    "ex_acc":  ex_correct  / ex_total,
                    "dev_acc": dev_correct / dev_total,
                }
                results.append(r)

                elapsed = time.time() - t_start
                avg     = elapsed / done
                eta     = avg * (total_combos - done)
                print(f"    [{done:>4}/{total_combos}]  ETA {eta:>5.0f}s  "
                      f"ex={ex_correct}/{ex_total}  dev={dev_correct/dev_total:.4f}  "
                      f"discount={kn_d}  alpha={lap_a}  mcc={min_cc}  maxctx={max_ctx}")

    results.sort(key=lambda x: x["dev_acc"], reverse=True)
    keys = list(PARAM_GRID.keys())
    header  = f"{'Rank':>4}  {'Exmpl':>6}  {'DevAcc':>7}  " + "  ".join(f"{k:>20}" for k in keys)
    divider = "-" * len(header)
    print(f"\n{header}")
    print(divider)
    for rank, r in enumerate(results[:20], 1): 
        ex_str = f"{r['ex'][0]}/{r['ex'][1]}"
        row = f"{rank:>4}  {ex_str:>6}  {r['dev_acc']:>7.4f}  " + \
              "  ".join(f"{str(r['params'][k]):>20}" for k in keys)
        print(row)

    best = results[0]
    print(f"\n{'='*60}")
    print(f"Best dev accuracy : {best['dev_acc']:.4f}  ({best['dev'][0]}/{best['dev'][1]})")
    print(f"Example set score : {best['ex'][0]}/{best['ex'][1]}")
    print(f"Best params:")
    for k in keys:
        print(f"  --{k} {best['params'][k]}")
    print(f"Total elapsed: {time.time()-t_start:.1f}s")

    with open(RESULT_FILE, "w") as f:
        f.write(header + "\n" + divider + "\n")
        for rank, r in enumerate(results, 1):
            ex_str = f"{r['ex'][0]}/{r['ex'][1]}"
            row = f"{rank:>4}  {ex_str:>6}  {r['dev_acc']:>7.4f}  " + \
                  "  ".join(f"{str(r['params'][k]):>20}" for k in keys)
            f.write(row + "\n")
        f.write(f"\nBest: dev={best['dev_acc']:.4f}  example={best['ex'][0]}/{best['ex'][1]}\n")
        f.write(f"Params: {best['params']}\n")
    print(f"Results saved → {RESULT_FILE}")


if __name__ == "__main__":
    main()
