#!/usr/bin/env python
import argparse
import os
import re
import unicodedata

from datasets import load_dataset


def normalize_line(text):
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def keep_line(text, min_chars, max_chars):
    if len(text) < min_chars:
        return False
    if len(text) > max_chars:
        return False
    return True


def download_language(dataset_name, lang, lines_per_lang, output_dir, min_chars, max_chars):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"mc4_{lang}.txt")
    try:
        stream = load_dataset(dataset_name, lang, split="train", streaming=True)
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" in str(exc):
            raise RuntimeError(
                f"Dataset '{dataset_name}' requires a remote loading script that this "
                "datasets version blocks. Install a 2.x datasets release (for example "
                "'pip install \"datasets==2.19.2\"') and rerun this command."
            ) from exc
        raise

    kept = 0
    with open(out_path, "wt", encoding="utf-8") as out:
        for row in stream:
            text = normalize_line(row.get("text", ""))
            if not keep_line(text, min_chars=min_chars, max_chars=max_chars):
                continue
            out.write(text + "\n")
            kept += 1
            if kept >= lines_per_lang:
                break

    return out_path, kept


def parse_args():
    parser = argparse.ArgumentParser(description="Download multilingual mC4 text into local .txt files.")
    parser.add_argument(
        "--dataset",
        default="mc4",
        help="Hugging Face dataset id to stream from",
    )
    parser.add_argument(
        "--langs",
        default="en,es,fr,de,pt,it,nl,ru,uk,pl,tr,ar,fa,hi,bn,ur,zh,ja,ko,th,vi,id,sw",
        help="comma-separated language configs to download",
    )
    parser.add_argument(
        "--lines_per_lang",
        type=int,
        default=20000,
        help="number of kept lines per language",
    )
    parser.add_argument(
        "--output_dir",
        default="data",
        help="where to write output .txt files",
    )
    parser.add_argument(
        "--min_chars",
        type=int,
        default=10,
        help="minimum normalized characters per line",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=1000,
        help="maximum normalized characters per line",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]
    if not langs:
        raise ValueError("No languages provided in --langs")

    print(f"Dataset: {args.dataset}")
    print(f"Languages: {langs}")
    print(f"Lines per language: {args.lines_per_lang}")
    print(f"Output dir: {args.output_dir}")

    for lang in langs:
        out_path, kept = download_language(
            dataset_name=args.dataset,
            lang=lang,
            lines_per_lang=args.lines_per_lang,
            output_dir=args.output_dir,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
        )
        print(f"{lang}: wrote {kept} lines to {out_path}")


if __name__ == "__main__":
    main()
