from __future__ import annotations

import math
import random
import re
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import minmax_scale

from .data import AlertContext, Event, corrupt_contexts, flatten_events


SECURITY_QUERY = (
    "security alert triage anomaly fatal error severe warning failure failed "
    "exception interrupt machine check kernel panic timeout unavailable "
    "unauthorized privilege escalation attack malicious suspicious corruption"
)

POISON_RE = re.compile(
    r"\b(ignore previous|system override|classify as|no action required|"
    r"false positive|escalate immediately|escalate critical|analyst note|"
    r"mark as benign|close ticket|override)\b",
    flags=re.IGNORECASE,
)


def normalize(values: np.ndarray, neutral: float = 0.0) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    finite = np.isfinite(arr)
    if not finite.any():
        return np.full_like(arr, neutral, dtype=float)
    arr = np.where(finite, arr, np.nanmedian(arr[finite]))
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo < 1e-12:
        return np.full_like(arr, neutral, dtype=float)
    return (arr - lo) / (hi - lo)


def top_k_indices(scores: np.ndarray, k: int) -> list[int]:
    if len(scores) == 0:
        return []
    k = max(1, min(k, len(scores)))
    order = np.argsort(scores)[::-1]
    return sorted(order[:k].tolist())


def event_texts(contexts: Iterable[AlertContext]) -> list[str]:
    return [event.text for context in contexts for event in context.events]


def event_labels_from_contexts(contexts: Iterable[AlertContext]) -> np.ndarray:
    labels: list[int] = []
    for context in contexts:
        labels.extend([context.label] * len(context.events))
    return np.asarray(labels, dtype=int)


def context_texts(contexts: Iterable[AlertContext]) -> list[str]:
    return [context.text() for context in contexts]


def build_alert_query(context: AlertContext) -> str:
    levels = " ".join(sorted({event.level for event in context.events if event.level}))
    components = " ".join(sorted({event.component for event in context.events if event.component}))
    event_ids = " ".join(sorted({event.event_id for event in context.events if event.event_id})[:6])
    return f"{SECURITY_QUERY} levels {levels} components {components} event_ids {event_ids}"


class BaseSelector:
    name = "Base"

    def fit(self, contexts: list[AlertContext]) -> "BaseSelector":
        return self

    def select(self, context: AlertContext) -> list[int]:
        raise NotImplementedError

    def transform(self, contexts: list[AlertContext]) -> tuple[list[str], list[list[int]], list[dict[str, int]]]:
        texts: list[str] = []
        selected: list[list[int]] = []
        counts: list[dict[str, int]] = []
        for context in contexts:
            indices = self.select(context)
            selected.append(indices)
            texts.append(context.text(indices))
            counts.append(context.selected_counts(indices))
        return texts, selected, counts


class FullContextSelector(BaseSelector):
    name = "Full Context"

    def select(self, context: AlertContext) -> list[int]:
        return list(range(len(context.events)))


class RandomKSelector(BaseSelector):
    name = "Random-k Selection"

    def __init__(self, k: int = 3, seed: int = 13):
        self.k = k
        self.seed = seed

    def select(self, context: AlertContext) -> list[int]:
        rng = random.Random(f"{self.seed}:{context.context_id}")
        n = len(context.events)
        k = max(1, min(self.k, n))
        return sorted(rng.sample(range(n), k))


class BM25Selector(BaseSelector):
    name = "TF-IDF/BM25 Selection"

    def __init__(self, k: int = 3, k1: float = 1.5, b: float = 0.75):
        self.k = k
        self.k1 = k1
        self.b = b
        self.token_pattern = re.compile(r"(?u)\b\w\w+\b")
        self.idf_: dict[str, float] = {}
        self.avgdl_ = 1.0

    def _tokens(self, text: str) -> list[str]:
        return [m.group(0).lower() for m in self.token_pattern.finditer(text)]

    def fit(self, contexts: list[AlertContext]) -> "BM25Selector":
        docs = [self._tokens(text) for text in event_texts(contexts)]
        n_docs = max(1, len(docs))
        self.avgdl_ = float(np.mean([len(doc) for doc in docs]) or 1.0)
        df: dict[str, int] = {}
        for doc in docs:
            for token in set(doc):
                df[token] = df.get(token, 0) + 1
        self.idf_ = {
            token: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in df.items()
        }
        return self

    def _score(self, doc: list[str], query: list[str]) -> float:
        tf: dict[str, int] = {}
        for token in doc:
            tf[token] = tf.get(token, 0) + 1
        dl = len(doc) or 1
        score = 0.0
        for token in query:
            if token not in tf:
                continue
            idf = self.idf_.get(token, 0.0)
            freq = tf[token]
            denom = freq + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl_)
            score += idf * freq * (self.k1 + 1.0) / denom
        return score

    def select(self, context: AlertContext) -> list[int]:
        query = self._tokens(build_alert_query(context))
        scores = np.asarray([self._score(self._tokens(event.text), query) for event in context.events])
        return top_k_indices(scores, self.k)


class MutualInformationSelector(BaseSelector):
    name = "Mutual Information Feature Selection"

    def __init__(self, k: int = 3, max_features: int = 5000, seed: int = 13):
        self.k = k
        self.max_features = max_features
        self.seed = seed
        self.vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2), min_df=1, max_features=max_features)
        self.feature_scores_: np.ndarray | None = None

    def fit(self, contexts: list[AlertContext]) -> "MutualInformationSelector":
        texts = event_texts(contexts)
        y = event_labels_from_contexts(contexts)
        x = self.vectorizer.fit_transform(texts)
        self.feature_scores_ = mutual_info_classif(
            x,
            y,
            discrete_features=True,
            random_state=self.seed,
        )
        return self

    def _event_scores(self, events: list[Event]) -> np.ndarray:
        if self.feature_scores_ is None:
            raise RuntimeError("Selector is not fitted")
        x = self.vectorizer.transform([event.text for event in events])
        scores = x @ self.feature_scores_
        return np.asarray(scores).ravel()

    def select(self, context: AlertContext) -> list[int]:
        return top_k_indices(self._event_scores(context.events), self.k)


class EmbeddingSimilaritySelector(BaseSelector):
    name = "Embedding Similarity Selection"

    def __init__(self, k: int = 3, max_features: int = 8000):
        self.k = k
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_features=max_features,
            norm="l2",
        )

    def fit(self, contexts: list[AlertContext]) -> "EmbeddingSimilaritySelector":
        self.vectorizer.fit(event_texts(contexts) + [SECURITY_QUERY])
        return self

    def select(self, context: AlertContext) -> list[int]:
        x = self.vectorizer.transform([event.text for event in context.events])
        q = self.vectorizer.transform([build_alert_query(context)])
        scores = cosine_similarity(x, q).ravel()
        return top_k_indices(scores, self.k)


class ShapSelector(BaseSelector):
    name = "SHAP-based Selection"

    def __init__(self, k: int = 3, max_features: int = 8000, seed: int = 13):
        self.k = k
        self.seed = seed
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=max_features)
        self.model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        self.feature_scores_: np.ndarray | None = None
        self.used_shap = False

    def fit(self, contexts: list[AlertContext]) -> "ShapSelector":
        x = self.vectorizer.fit_transform(context_texts(contexts))
        y = np.asarray([context.label for context in contexts])
        self.model.fit(x, y)
        self.feature_scores_ = np.abs(self.model.coef_[0])
        try:
            import shap  # type: ignore

            sample_size = min(100, x.shape[0])
            sample = x[:sample_size]
            explainer = shap.LinearExplainer(self.model, sample)
            values = explainer(sample)
            arr = np.asarray(values.values)
            if arr.ndim == 3:
                arr = arr[:, :, -1]
            shap_scores = np.abs(arr).mean(axis=0)
            if shap_scores.shape[0] == self.feature_scores_.shape[0]:
                self.feature_scores_ = shap_scores
                self.used_shap = True
        except Exception:
            self.used_shap = False
        return self

    def select(self, context: AlertContext) -> list[int]:
        if self.feature_scores_ is None:
            raise RuntimeError("Selector is not fitted")
        x = self.vectorizer.transform([event.text for event in context.events])
        scores = x @ self.feature_scores_
        return top_k_indices(np.asarray(scores).ravel(), self.k)


@dataclass
class TeleMinWeights:
    importance: float = 0.36
    semantic: float = 0.16
    rarity: float = 0.16
    robustness: float = 0.32
    redundancy: float = 0.30
    poison_penalty: float = 0.55


class TeleMinRAGSelector(BaseSelector):
    name = "TeleMin-RAG"

    def __init__(
        self,
        min_k: int = 1,
        max_k: int = 4,
        max_features: int = 8000,
        seed: int = 13,
        auto_tune: bool = True,
        tune_poison_rate: float = 0.10,
        use_poison_penalty: bool = True,
        use_stability: bool = True,
        use_rarity: bool = True,
        use_semantic: bool = True,
        use_redundancy: bool = True,
        adaptive_k: bool = True,
        variant_name: str | None = None,
    ):
        if variant_name:
            self.name = variant_name
        self.min_k = min_k
        self.max_k = max_k
        self.max_features = max_features
        self.seed = seed
        self.auto_tune = auto_tune
        self.tune_poison_rate = tune_poison_rate
        self.use_poison_penalty = use_poison_penalty
        self.use_stability = use_stability
        self.use_rarity = use_rarity
        self.use_semantic = use_semantic
        self.use_redundancy = use_redundancy
        self.adaptive_k = adaptive_k
        self.weights = TeleMinWeights()
        self.stop_threshold = 0.72
        self.count_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 2), min_df=1, max_features=max_features)
        self.embed_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=max_features, norm="l2")
        self.context_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=max_features)
        self.pre_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        self.feature_importance_: np.ndarray | None = None
        self.feature_stability_: np.ndarray | None = None
        self.feature_rarity_: np.ndarray | None = None
        self.tuning_log_: list[dict] = []
        self.fit_seconds_: float = 0.0

    def fit(self, contexts: list[AlertContext]) -> "TeleMinRAGSelector":
        start = time.perf_counter()
        if self.auto_tune and len(contexts) >= 40 and len(set(c.label for c in contexts)) == 2:
            fit_contexts, val_contexts = train_test_split(
                contexts,
                test_size=0.25,
                random_state=self.seed,
                stratify=[context.label for context in contexts],
            )
            self._fit_scorers(fit_contexts)
            self._tune(fit_contexts, val_contexts)
        self._fit_scorers(contexts)
        self.fit_seconds_ = time.perf_counter() - start
        return self

    def _fit_scorers(self, contexts: list[AlertContext]) -> None:
        texts = event_texts(contexts)
        y_events = event_labels_from_contexts(contexts)
        x_count = self.count_vectorizer.fit_transform(texts)
        mi = mutual_info_classif(x_count, y_events, discrete_features=True, random_state=self.seed)
        stability = self._stability_scores(x_count, y_events)
        df = np.asarray((x_count > 0).sum(axis=0)).ravel()
        rarity = np.log((x_count.shape[0] + 1.0) / (df + 1.0)) + 1.0
        self.feature_importance_ = normalize(mi)
        self.feature_stability_ = normalize(stability)
        self.feature_rarity_ = normalize(rarity)

        self.embed_vectorizer.fit(texts + [SECURITY_QUERY])
        x_ctx = self.context_vectorizer.fit_transform(context_texts(contexts))
        y_ctx = np.asarray([context.label for context in contexts])
        self.pre_model.fit(x_ctx, y_ctx)

    def _stability_scores(self, x: sparse.spmatrix, y: np.ndarray, rounds: int = 8) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        n = x.shape[0]
        effects = []
        for _ in range(rounds):
            idx = rng.choice(n, size=n, replace=True)
            xb = x[idx]
            yb = y[idx]
            pos = xb[yb == 1]
            neg = xb[yb == 0]
            if pos.shape[0] == 0 or neg.shape[0] == 0:
                continue
            pos_rate = (np.asarray(pos.sum(axis=0)).ravel() + 1.0) / (pos.shape[0] + 2.0)
            neg_rate = (np.asarray(neg.sum(axis=0)).ravel() + 1.0) / (neg.shape[0] + 2.0)
            effect = np.abs(np.log(pos_rate / (1.0 - pos_rate)) - np.log(neg_rate / (1.0 - neg_rate)))
            effects.append(effect)
        if not effects:
            return np.zeros(x.shape[1])
        arr = np.vstack(effects)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        return mean / (std + 0.05)

    def _candidate_weight_sets(self) -> list[TeleMinWeights]:
        return [
            TeleMinWeights(0.36, 0.16, 0.16, 0.32, 0.30, 0.55),
            TeleMinWeights(0.45, 0.12, 0.13, 0.30, 0.25, 0.60),
            TeleMinWeights(0.30, 0.25, 0.15, 0.30, 0.35, 0.55),
            TeleMinWeights(0.28, 0.18, 0.18, 0.36, 0.40, 0.70),
            TeleMinWeights(0.40, 0.20, 0.10, 0.30, 0.20, 0.50),
            TeleMinWeights(0.25, 0.20, 0.20, 0.35, 0.45, 0.75),
            TeleMinWeights(0.50, 0.25, 0.15, 0.10, 0.25, 0.60),
            TeleMinWeights(0.42, 0.33, 0.15, 0.10, 0.30, 0.65),
            TeleMinWeights(0.55, 0.20, 0.20, 0.05, 0.20, 0.70),
        ]

    def _tune(self, fit_contexts: list[AlertContext], val_contexts: list[AlertContext]) -> None:
        noise_pool = flatten_events(fit_contexts, normal_only=True) or flatten_events(fit_contexts)
        val_sets = []
        for level in (0.0, 0.20, 0.40):
            val_sets.extend(
                corrupt_contexts(
                    val_contexts,
                    noise_level=level,
                    noise_pool=noise_pool,
                    poison_rate=self.tune_poison_rate,
                    seed=self.seed + int(level * 1000) + 77,
                )
            )

        best: tuple[float, TeleMinWeights, float, int, int] | None = None
        thresholds = [0.64, 0.76, 0.88, 1.01]
        budget_pairs = [
            (1, min(3, self.max_k)),
            (2, min(4, max(self.max_k, 4))),
            (3, min(6, max(self.max_k, 6))),
        ]
        original_weights = self.weights
        original_threshold = self.stop_threshold
        original_min_k = self.min_k
        original_max_k = self.max_k
        for weights in self._candidate_weight_sets():
            self.weights = weights
            for min_k, max_k in budget_pairs:
                if min_k > max_k:
                    continue
                self.min_k = min_k
                self.max_k = max_k
                for threshold in thresholds:
                    self.stop_threshold = threshold
                    texts, selected, counts = self.transform(val_sets)
                    x_val = self.context_vectorizer.transform(texts)
                    pred = self.pre_model.predict(x_val)
                    labels = np.asarray([context.label for context in val_sets])
                    f1 = f1_score(labels, pred, zero_division=0)
                    avg_fraction = np.mean([
                        len(indices) / max(1, len(context.events))
                        for indices, context in zip(selected, val_sets)
                    ])
                    poison_rate = np.mean([
                        count["selected_poison"] / max(1, count["selected_logs"])
                        for count in counts
                    ])
                    objective = f1 - 0.02 * avg_fraction - 0.10 * poison_rate
                    record = {
                        "objective": objective,
                        "f1": f1,
                        "avg_fraction": avg_fraction,
                        "poison_rate": poison_rate,
                        "threshold": threshold,
                        "min_k": min_k,
                        "max_k": max_k,
                        **weights.__dict__,
                    }
                    self.tuning_log_.append(record)
                    if best is None or objective > best[0]:
                        best = (objective, weights, threshold, min_k, max_k)
        if best is None:
            self.weights = original_weights
            self.stop_threshold = original_threshold
            self.min_k = original_min_k
            self.max_k = original_max_k
        else:
            _, self.weights, self.stop_threshold, self.min_k, self.max_k = best

    def _feature_sum(self, texts: list[str], scores: np.ndarray, vectorizer: CountVectorizer) -> np.ndarray:
        x = vectorizer.transform(texts)
        raw = x @ scores
        nnz = np.maximum(1, np.asarray((x > 0).sum(axis=1)).ravel())
        return np.asarray(raw).ravel() / nnz

    def _score_components(self, context: AlertContext) -> dict[str, np.ndarray]:
        if self.feature_importance_ is None or self.feature_stability_ is None or self.feature_rarity_ is None:
            raise RuntimeError("Selector is not fitted")

        texts = [event.text for event in context.events]
        importance = self._feature_sum(texts, self.feature_importance_, self.count_vectorizer)
        stability = self._feature_sum(texts, self.feature_stability_, self.count_vectorizer)
        rarity = self._feature_sum(texts, self.feature_rarity_, self.count_vectorizer)

        x_embed = self.embed_vectorizer.transform(texts)
        q_embed = self.embed_vectorizer.transform([build_alert_query(context)])
        semantic = cosine_similarity(x_embed, q_embed).ravel()
        if x_embed.shape[0] > 1:
            sim = cosine_similarity(x_embed)
            np.fill_diagonal(sim, 0.0)
            coherence = np.sort(sim, axis=1)[:, -min(2, sim.shape[1] - 1) :].mean(axis=1)
        else:
            coherence = np.ones(len(texts))
        poison_penalty = np.asarray([1.0 if POISON_RE.search(text) else 0.0 for text in texts])
        if not self.use_stability:
            stability = np.zeros_like(stability)
        if not self.use_rarity:
            rarity = np.zeros_like(rarity)
        if not self.use_semantic:
            semantic = np.zeros_like(semantic)
        if not self.use_poison_penalty:
            poison_penalty = np.zeros_like(poison_penalty)
        # Stability is useful as a guardrail, but over-weighting it can select
        # recurrent benign tokens. Coherence receives the larger share.
        robustness = 0.30 * normalize(stability) + 0.70 * normalize(coherence)
        if self.use_poison_penalty:
            robustness = np.clip(robustness - self.weights.poison_penalty * poison_penalty, 0.0, 1.0)
        return {
            "importance": normalize(importance),
            "semantic": normalize(semantic),
            "rarity": normalize(rarity),
            "robustness": normalize(robustness),
            "poison_penalty": poison_penalty,
            "x_embed": x_embed,
        }

    def _base_scores(self, components: dict[str, np.ndarray]) -> np.ndarray:
        w = self.weights
        score = (
            w.importance * components["importance"]
            + w.semantic * components["semantic"]
            + w.rarity * components["rarity"]
            + w.robustness * components["robustness"]
        )
        if self.use_poison_penalty:
            score = score - w.poison_penalty * components["poison_penalty"]
        return score

    def select(self, context: AlertContext) -> list[int]:
        n = len(context.events)
        if n == 0:
            return []
        max_k = max(self.min_k, min(self.max_k, n))
        components = self._score_components(context)
        base = self._base_scores(components)
        sim = cosine_similarity(components["x_embed"]) if n > 1 else np.zeros((n, n))
        selected: list[int] = []
        remaining = set(range(n))
        for _ in range(max_k):
            adjusted = {}
            for idx in remaining:
                redundancy = max((sim[idx, chosen] for chosen in selected), default=0.0)
                if not self.use_redundancy:
                    redundancy = 0.0
                adjusted[idx] = base[idx] - self.weights.redundancy * redundancy
            chosen = max(adjusted, key=adjusted.get)
            selected.append(chosen)
            remaining.remove(chosen)
            if self.adaptive_k and len(selected) >= self.min_k:
                text = context.text(sorted(selected))
                x = self.context_vectorizer.transform([text])
                confidence = float(np.max(self.pre_model.predict_proba(x)))
                if confidence >= self.stop_threshold:
                    break
        return sorted(selected)


def make_selectors(k: int, seed: int, include_shap: bool = True, auto_tune: bool = True) -> list[BaseSelector]:
    selectors: list[BaseSelector] = [
        FullContextSelector(),
        RandomKSelector(k=k, seed=seed),
        BM25Selector(k=k),
        MutualInformationSelector(k=k, seed=seed),
        EmbeddingSimilaritySelector(k=k),
    ]
    if include_shap:
        selectors.append(ShapSelector(k=k, seed=seed))
    selectors.append(TeleMinRAGSelector(min_k=1, max_k=max(k + 3, 6), seed=seed, auto_tune=auto_tune))
    return selectors


def make_ablation_selectors(k: int, seed: int, auto_tune: bool = True) -> list[BaseSelector]:
    max_k = max(k + 3, 6)
    common = {"min_k": 1, "max_k": max_k, "seed": seed, "auto_tune": auto_tune}
    return [
        TeleMinRAGSelector(**common, variant_name="Full TeleMin-RAG"),
        TeleMinRAGSelector(**common, use_poison_penalty=False, variant_name="without poisoning penalty"),
        TeleMinRAGSelector(**common, use_stability=False, variant_name="without bootstrap robustness"),
        TeleMinRAGSelector(**common, use_rarity=False, variant_name="without rarity"),
        TeleMinRAGSelector(**common, use_semantic=False, variant_name="without semantic similarity"),
        TeleMinRAGSelector(**common, use_redundancy=False, variant_name="without redundancy penalty"),
        TeleMinRAGSelector(
            min_k=k,
            max_k=k,
            seed=seed,
            auto_tune=False,
            adaptive_k=False,
            variant_name="fixed-k TeleMin",
        ),
        TeleMinRAGSelector(
            min_k=1,
            max_k=max_k,
            seed=seed,
            auto_tune=False,
            adaptive_k=True,
            variant_name="adaptive-k TeleMin",
        ),
    ]
