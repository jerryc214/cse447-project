import glob
import os
from typing import List

from .text_utils import normalize_text

def iter_default_training_files():
    patterns = ["data/**/*.txt", "corpus/**/*.txt", "train/**/*.txt"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    return sorted(set([path for path in files if os.path.isfile(path)]))


def resolve_training_files():
    train_files_env = os.environ.get("CSE447_TRAIN_FILES", "").strip()
    if train_files_env:
        return [path for path in train_files_env.split(":") if path]

    discovered = iter_default_training_files()
    if discovered:
        return discovered

    # Fallback so train mode can still run with a tiny local dataset.
    return ["example/input.txt"]


def load_text_lines(file_paths):
    max_lines_env = os.environ.get("CSE447_MAX_TRAIN_LINES", "").strip()
    max_lines = int(max_lines_env) if max_lines_env else None
    balanced_env = os.environ.get("CSE447_BALANCED_BY_FILE", "").strip().lower()
    use_balanced = balanced_env in ("1", "true", "yes")

    if use_balanced and max_lines is not None and len(file_paths) > 1:
        return _load_text_lines_balanced(file_paths, max_lines)

    lines = []
    for file_path in file_paths:
        try:
            with open(file_path, "rt", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    text = normalize_text(line.rstrip("\n"))
                    if text:
                        lines.append(text)
                    if max_lines is not None and len(lines) >= max_lines:
                        return lines
        except OSError:
            continue
    return lines


def _load_text_lines_balanced(file_paths: List[str], max_lines: int):
    """
    Load lines in round-robin order across files to reduce language/file skew
    when training with a global line cap.
    """
    active = []
    for path in file_paths:
        try:
            handle = open(path, "rt", encoding="utf-8", errors="ignore")
            active.append(handle)
        except OSError:
            continue

    lines = []
    try:
        while active and len(lines) < max_lines:
            next_active = []
            for handle in active:
                raw = handle.readline()
                if not raw:
                    handle.close()
                    continue
                text = normalize_text(raw.rstrip("\n"))
                if text:
                    lines.append(text)
                    if len(lines) >= max_lines:
                        break
                next_active.append(handle)
            active = next_active
    finally:
        for handle in active:
            try:
                handle.close()
            except Exception:
                pass

    return lines


def load_test_data(file_path):
    lines = []
    with open(file_path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            lines.append(normalize_text(line.rstrip("\n")))
    return lines


def write_predictions(predictions, file_path):
    with open(file_path, "wt", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(f"{prediction}\n")
