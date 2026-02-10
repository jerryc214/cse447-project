import gzip
import os
import pickle
from collections import Counter, defaultdict

from .text_utils import normalize_text

class CharNGramLanguageModel:
    def __init__(
        self,
        ngram_order=6,
        laplace_alpha=1.0,
        max_chars_per_context=64,
        min_context_count=3,
        max_contexts=1000000,
    ):
        self.ngram_order = int(ngram_order)
        self.laplace_alpha = float(laplace_alpha)
        self.max_chars_per_context = int(max_chars_per_context)
        self.min_context_count = int(min_context_count)
        self.max_contexts = int(max_contexts)
        self.context_counts = defaultdict(Counter)
        self.unigram = Counter()
        # Keep fallback symbols script-agnostic. Real defaults are learned from unigram counts.
        self.default_chars = [" ", ".", ",", "。", "،", "।", "，", "・", "-"]

    def fit(self, lines):
        max_context = self.ngram_order - 1
        for raw_line in lines:
            sequence = normalize_text(raw_line)
            for idx, next_char in enumerate(sequence):
                if next_char in ("\n", "\r"):
                    continue

                self.unigram[next_char] += 1

                start = max(0, idx - max_context)
                for left in range(start, idx):
                    context = sequence[left:idx]
                    if context:
                        self.context_counts[context][next_char] += 1

        self._trim_context_tables()
        self._refresh_default_chars()

    def predict_top_k(self, text, k=3):
        sequence = normalize_text(text)
        scores = Counter()
        max_context = self.ngram_order - 1

        for ctx_len in range(min(max_context, len(sequence)), 0, -1):
            context = sequence[-ctx_len:]
            next_char_counts = self.context_counts.get(context)
            if not next_char_counts:
                continue

            total = sum(next_char_counts.values())
            if total <= 0:
                continue

            # Weighted backoff: longer context gets more influence.
            context_weight = 1.0 / float(max_context - ctx_len + 1)
            vocab_size = max(1, len(next_char_counts))

            for char, count in next_char_counts.items():
                if char in ("\n", "\r"):
                    continue
                # Laplace smoothing (add-alpha); default alpha=1.0.
                prob = (count + self.laplace_alpha) / (total + self.laplace_alpha * vocab_size)
                scores[char] += context_weight * prob

        if self.unigram:
            self._add_unigram_backoff(scores)

        guesses = []
        for char, _ in scores.most_common():
            if char in ("\n", "\r"):
                continue
            if char not in guesses:
                guesses.append(char)
            if len(guesses) == k:
                break

        if len(guesses) < k:
            for char in self.default_chars:
                if char not in guesses and char not in ("\n", "\r"):
                    guesses.append(char)
                if len(guesses) == k:
                    break

        while len(guesses) < k:
            guesses.append(" ")

        return guesses[:k]

    def predict_batch(self, inputs, k=3):
        predictions = []
        for line in inputs:
            try:
                top_k = self.predict_top_k(line, k=k)
            except Exception:
                top_k = self.default_chars[:k]
                while len(top_k) < k:
                    top_k.append(" ")
            predictions.append("".join(top_k))
        return predictions

    def save(self, work_dir):
        payload = {
            "ngram_order": self.ngram_order,
            "laplace_alpha": self.laplace_alpha,
            "max_chars_per_context": self.max_chars_per_context,
            "min_context_count": self.min_context_count,
            "max_contexts": self.max_contexts,
            "context_counts": {context: dict(counts) for context, counts in self.context_counts.items()},
            "unigram": dict(self.unigram),
            "default_chars": self.default_chars,
        }
        checkpoint_path = os.path.join(work_dir, "model.checkpoint")
        with gzip.open(checkpoint_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, work_dir):
        checkpoint_path = os.path.join(work_dir, "model.checkpoint")
        with gzip.open(checkpoint_path, "rb") as handle:
            payload = pickle.load(handle)

        model = cls(
            ngram_order=payload.get("ngram_order", 6),
            laplace_alpha=payload.get("laplace_alpha", 1.0),
            max_chars_per_context=payload.get("max_chars_per_context", 64),
            min_context_count=payload.get("min_context_count", 3),
            max_contexts=payload.get("max_contexts", 1000000),
        )
        model.context_counts = defaultdict(Counter)
        for context, counts in payload.get("context_counts", {}).items():
            model.context_counts[context] = Counter(counts)
        model.unigram = Counter(payload.get("unigram", {}))
        model.default_chars = payload.get("default_chars", model.default_chars)
        return model

    def _trim_context_tables(self):
        ranked_contexts = []
        for context, counts in self.context_counts.items():
            total = sum(counts.values())
            if total < self.min_context_count:
                continue
            ranked_contexts.append((context, total, counts))

        ranked_contexts.sort(key=lambda item: item[1], reverse=True)
        if self.max_contexts > 0:
            ranked_contexts = ranked_contexts[: self.max_contexts]

        trimmed = defaultdict(Counter)
        for context, _, counts in ranked_contexts:
            top_items = counts.most_common(self.max_chars_per_context)
            if top_items:
                trimmed[context] = Counter(dict(top_items))
        self.context_counts = trimmed

    def _refresh_default_chars(self):
        if not self.unigram:
            return

        frequent_defaults = [
            char for char, _ in self.unigram.most_common(self.max_chars_per_context) if char not in ("\n", "\r")
        ]
        merged = []
        seen = set()

        for char in frequent_defaults + self.default_chars:
            if char not in seen:
                seen.add(char)
                merged.append(char)

        self.default_chars = merged

    def _add_unigram_backoff(self, scores):
        total = sum(self.unigram.values())
        if total <= 0:
            return

        vocab_size = max(1, len(self.unigram))
        for char, count in self.unigram.most_common(128):
            if char in ("\n", "\r"):
                continue
            prob = (count + self.laplace_alpha) / (total + self.laplace_alpha * vocab_size)
            scores[char] += 0.05 * prob
