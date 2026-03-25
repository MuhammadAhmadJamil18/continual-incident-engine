from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np


class FeatureEncoder(abc.ABC):
    """Maps raw incident text to fixed-size float vectors for the MLP."""

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        """Vector length produced by ``transform``."""

    @abc.abstractmethod
    def transform(self, text: str) -> np.ndarray:
        """Embed a single document (shape ``(output_dim,)``, float32)."""

    def transform_batch(self, texts: list[str]) -> np.ndarray:
        """Stacked embeddings (shape ``(n, output_dim)``)."""
        rows = [self.transform(t) for t in texts]
        return np.stack(rows, axis=0).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        """Serializable state for checkpointing (override if stateful)."""
        return {"type": self.encoder_type}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        """Restore from ``state_dict``."""

    def observe_training_texts(self, texts: list[str]) -> None:
        """Optional hook when training on raw text (e.g. refit TF-IDF)."""

    @property
    def encoder_type(self) -> str:
        return self.__class__.__name__


@dataclass
class IdentityEncoder(FeatureEncoder):
    """Pass-through: client supplies precomputed ``features``; text path unused."""

    dim: int

    @property
    def output_dim(self) -> int:
        return self.dim

    def transform(self, text: str) -> np.ndarray:
        raise RuntimeError("IdentityEncoder does not encode text; send features.")

    def state_dict(self) -> dict[str, Any]:
        return {**super().state_dict(), "dim": self.dim}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        pass


@dataclass
class HashingTextEncoder(FeatureEncoder):
    """
    Streaming-friendly bag-of-words via hashing (no vocabulary fit step).

    Normalized L2 so Euclidean distance in buffer space is meaningful.
    """

    n_features: int = 128

    def __post_init__(self) -> None:
        from sklearn.feature_extraction.text import HashingVectorizer
        from sklearn.preprocessing import Normalizer

        self._hv = HashingVectorizer(
            n_features=self.n_features,
            alternate_sign=False,
            norm=None,
            ngram_range=(1, 2),
        )
        self._norm = Normalizer(copy=False)

    @property
    def output_dim(self) -> int:
        return self.n_features

    def transform(self, text: str) -> np.ndarray:
        x = self._hv.transform([text])
        x = self._norm.transform(x)
        return np.asarray(x.todense(), dtype=np.float32).reshape(-1)

    def state_dict(self) -> dict[str, Any]:
        return {**super().state_dict(), "n_features": self.n_features}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        pass


@dataclass
class TfidfTextEncoder(FeatureEncoder):
    """
    TF-IDF with a vocabulary fit from accumulated training text.

    First ``fit`` uses all strings seen so far; refit when corpus grows beyond
    ``refit_every`` new documents (bounded cost for demos).
    """

    max_features: int = 128
    refit_every: int = 200
    _corpus: list[str] = field(default_factory=list)
    _since_refit: int = 0
    _fitted: bool = False

    def __post_init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=1,
        )

    @property
    def output_dim(self) -> int:
        return self.max_features

    def observe_training_texts(self, texts: list[str]) -> None:
        """Append training texts and optionally refit the vectorizer."""
        self._corpus.extend(texts)
        self._since_refit += len(texts)
        if not self._fitted or self._since_refit >= self.refit_every:
            self._refit()

    def _refit(self) -> None:
        if not self._corpus:
            return
        self._vectorizer.fit(self._corpus)
        self._fitted = True
        self._since_refit = 0

    def transform(self, text: str) -> np.ndarray:
        if not self._fitted:
            self.observe_training_texts([text])
            self._refit()
        x = self._vectorizer.transform([text])
        out = np.asarray(x.todense(), dtype=np.float32).reshape(-1)
        if out.shape[0] < self.max_features:
            pad = np.zeros(self.max_features, dtype=np.float32)
            pad[: out.shape[0]] = out
            out = pad
        elif out.shape[0] > self.max_features:
            out = out[: self.max_features]
        return out

    def state_dict(self) -> dict[str, Any]:
        import pickle

        v_bytes: bytes | None = None
        if self._fitted:
            v_bytes = pickle.dumps(self._vectorizer, protocol=pickle.HIGHEST_PROTOCOL)
        return {
            **super().state_dict(),
            "max_features": self.max_features,
            "refit_every": self.refit_every,
            "corpus": list(self._corpus),
            "fitted": self._fitted,
            "vectorizer_pickle": v_bytes,
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        import pickle

        from sklearn.feature_extraction.text import TfidfVectorizer

        self._corpus = list(d.get("corpus", []))
        self._fitted = bool(d.get("fitted", False))
        self._since_refit = 0
        raw = d.get("vectorizer_pickle")
        if raw is not None:
            self._vectorizer = pickle.loads(raw)
            self._fitted = True
        elif d.get("vectorizer") is not None:
            self._vectorizer = d["vectorizer"]
            self._fitted = True
        else:
            self._vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=(1, 2),
                min_df=1,
            )
            if self._corpus:
                self._vectorizer.fit(self._corpus)
                self._fitted = True


@dataclass
class SentenceTransformerEncoder(FeatureEncoder):
    """Optional sentence-transformers backend (lazy import)."""

    model_name: str = "all-MiniLM-L6-v2"
    _model: Any = None

    def _ensure_model(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "Install sentence-transformers for SentenceTransformerEncoder"
                ) from e
            self._model = SentenceTransformer(self.model_name)

    @property
    def output_dim(self) -> int:
        self._ensure_model()
        return int(self._model.get_sentence_embedding_dimension())

    def transform(self, text: str) -> np.ndarray:
        self._ensure_model()
        v = self._model.encode(
            [text],
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return np.asarray(v, dtype=np.float32).reshape(-1)

    def state_dict(self) -> dict[str, Any]:
        return {**super().state_dict(), "model_name": self.model_name}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.model_name = d.get("model_name", self.model_name)
        self._model = None


def build_encoder(kind: str, feature_dim: int) -> FeatureEncoder:
    """
    Factory for encoders.

    Args:
        kind: ``identity`` | ``hashing`` | ``tfidf`` | ``sentence``.
        feature_dim: Used for ``identity`` / ``hashing`` / ``tfidf`` width.
    """
    k = kind.lower().strip()
    if k == "identity":
        return IdentityEncoder(dim=feature_dim)
    if k == "hashing":
        return HashingTextEncoder(n_features=feature_dim)
    if k == "tfidf":
        return TfidfTextEncoder(max_features=feature_dim)
    if k == "sentence":
        return SentenceTransformerEncoder()
    raise ValueError(f"Unknown encoder kind: {kind!r}")


def encoder_from_state(d: dict[str, Any], default_feature_dim: int) -> FeatureEncoder:
    """Rebuild encoder from a persisted state dict."""
    t = d.get("type", "IdentityEncoder")
    if t == "IdentityEncoder":
        enc = IdentityEncoder(dim=int(d.get("dim", default_feature_dim)))
        enc.load_state_dict(d)
        return enc
    if t == "HashingTextEncoder":
        enc = HashingTextEncoder(n_features=int(d.get("n_features", default_feature_dim)))
        enc.load_state_dict(d)
        return enc
    if t == "TfidfTextEncoder":
        enc = TfidfTextEncoder(
            max_features=int(d.get("max_features", default_feature_dim)),
            refit_every=int(d.get("refit_every", 200)),
        )
        enc.load_state_dict(d)
        return enc
    if t == "SentenceTransformerEncoder":
        enc = SentenceTransformerEncoder(model_name=str(d.get("model_name", "all-MiniLM-L6-v2")))
        enc.load_state_dict(d)
        return enc
    return IdentityEncoder(dim=default_feature_dim)
