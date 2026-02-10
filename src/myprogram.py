#!/usr/bin/env python
import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from lm.data_io import load_test_data, load_text_lines, resolve_training_files, write_predictions
from lm.ngram_model import CharNGramLanguageModel


class MyProgram:
    @staticmethod
    def train(
        work_dir,
        ngram_order,
        laplace_alpha,
        max_chars_per_context,
        min_context_count,
        max_contexts,
    ):
        if not os.path.isdir(work_dir):
            print(f"Making working directory {work_dir}")
            os.makedirs(work_dir)

        print("Instantiating model")
        model = CharNGramLanguageModel(
            ngram_order=ngram_order,
            laplace_alpha=laplace_alpha,
            max_chars_per_context=max_chars_per_context,
            min_context_count=min_context_count,
            max_contexts=max_contexts,
        )

        print("Loading training data")
        training_files = resolve_training_files()
        train_data = load_text_lines(training_files)
        print(f"Loaded {len(train_data)} lines from {len(training_files)} file(s)")

        print("Training")
        model.fit(train_data)

        print("Saving model")
        model.save(work_dir)

    @staticmethod
    def test(work_dir, test_data_path, test_output_path):
        print("Loading model")
        model = CharNGramLanguageModel.load(work_dir)

        print(f"Loading test data from {test_data_path}")
        test_data = load_test_data(test_data_path)

        print("Making predictions")
        predictions = model.predict_batch(test_data, k=3)

        print(f"Writing predictions to {test_output_path}")
        assert len(predictions) == len(test_data), (
            f"Expected {len(test_data)} predictions but got {len(predictions)}"
        )
        write_predictions(predictions, test_output_path)


def build_parser():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("mode", choices=("train", "test"), help="what to run")
    parser.add_argument("--work_dir", help="where to save", default="work")
    parser.add_argument("--test_data", help="path to test data", default="example/input.txt")
    parser.add_argument("--test_output", help="path to write test predictions", default="pred.txt")
    parser.add_argument("--ngram_order", type=int, default=6, help="n-gram order")
    parser.add_argument(
        "--laplace_alpha",
        type=float,
        default=1.0,
        help="Laplace smoothing alpha (add-alpha); use 1.0 for standard Laplace",
    )
    parser.add_argument(
        "--max_chars_per_context",
        type=int,
        default=64,
        help="max stored next-char candidates per context",
    )
    parser.add_argument(
        "--min_context_count",
        type=int,
        default=3,
        help="drop contexts observed fewer than this many times",
    )
    parser.add_argument(
        "--max_contexts",
        type=int,
        default=1000000,
        help="keep only this many most frequent contexts (<=0 means no cap)",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.mode == "train":
        MyProgram.train(
            work_dir=args.work_dir,
            ngram_order=args.ngram_order,
            laplace_alpha=args.laplace_alpha,
            max_chars_per_context=args.max_chars_per_context,
            min_context_count=args.min_context_count,
            max_contexts=args.max_contexts,
        )
    elif args.mode == "test":
        MyProgram.test(
            work_dir=args.work_dir,
            test_data_path=args.test_data,
            test_output_path=args.test_output,
        )
    else:
        raise NotImplementedError(f"Unknown mode {args.mode}")
