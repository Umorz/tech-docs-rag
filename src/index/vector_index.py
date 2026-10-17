from __future__ import annotations

"""
Vector index abstraction for Query-time Step (5): embeddings (semantic search).

This module provides a small, clean API to:
- build an approximate nearest-neighbor index using `hnswlib` when available
- fall back to a NumPy brute-force implementation when `hnswlib` is not installed

Design goals
------------
- Keep project structure clean: one module that hides backend differences.
- Be explicit and explainable: store metadata alongside the index.
- Be robust for technical corpora: predictable metrics (cosine / l2), float32 vectors.
- Be future-friendly: compatible with a later "hybrid retrieval" (BM25 + vectors) and reranking.

Backends
--------
1) HNSW (preferred):
   - Requires: `hnswlib` (and numpy)
   - Fast for large corpora.

2) Brute force (fallback):
   - Requires: `numpy`
   - Works for small-to-medium corpora, or when you cannot install extra deps.

Persistence
-----------
We write two files:
- <base_path>.meta.json        (metadata, backend, dim, metric, counts)
- backend-specific payload:
    - hnsw:       <base_path>.hnsw
    - bruteforce: <base_path>.npz

Notes on distances
------------------
- metric="cosine":
    - HNSW uses cosine distance (1 - cosine_similarity) when configured accordingly.
    - Brute force computes distance = 1 - dot(normalized(q), normalized(x)).
- metric="l2":
    - HNSW uses squared L2 internally (typical behavior); brute force returns squared L2.

The KNN APIs return (labels, distances), where "labels" are the integer IDs you supplied.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence, Union, cast

try:  # optional dependency
    import hnswlib  # type: ignore
except Exception:  # pragma: no cover
    hnswlib = None  # type: ignore[assignment]

# NumPy is optional at runtime, but we want type checkers to treat `np` as the numpy module
# (not as None) to avoid a cascade of type errors.
if TYPE_CHECKING:  # pragma: no cover
    import numpy as np  # type: ignore

    _NUMPY_IMPORT_ERROR: Exception | None = None
else:  # runtime
    np: Any
    try:  # required for brute-force backend
        import numpy as np  # type: ignore
    except Exception as e:  # pragma: no cover
        np = None
        _NUMPY_IMPORT_ERROR = e
    else:
        _NUMPY_IMPORT_ERROR = None


Metric = Literal["cosine", "l2"]
Backend = Literal["auto", "hnsw", "bruteforce"]


class VectorIndexError(RuntimeError):
    """User-facing error for vector-index related problems."""


@dataclass(frozen=True)
class VectorIndexMeta:
    backend: Literal["hnsw", "bruteforce"]
    dim: int
    metric: Metric
    count: int

    # Backend tuning (informational)
    hnsw_space: Optional[str] = None
    hnsw_M: Optional[int] = None
    hnsw_ef_construction: Optional[int] = None


def _require_numpy() -> None:
    if np is None:  # pragma: no cover
        raise VectorIndexError(
            "NumPy is required for vector index operations but is not installed. "
            f"Original import error: {_NUMPY_IMPORT_ERROR}"
        )


def _as_float32_matrix(x, *, dim: int) -> "np.ndarray":
    _require_numpy()
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        if arr.shape[0] != dim:
            raise VectorIndexError(f"Vector has dim={arr.shape[0]}, expected dim={dim}")
        arr = arr.reshape(1, dim)
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise VectorIndexError(f"Expected shape (N, {dim}), got {arr.shape}")
    return arr


def _as_int64_labels(ids: Union[Sequence[int], "np.ndarray"]) -> "np.ndarray":
    _require_numpy()
    lab = np.asarray(ids, dtype=np.int64)
    if lab.ndim != 1:
        raise VectorIndexError(f"Expected 1D ids array, got shape {lab.shape}")
    return lab


def _normalize_rows(x: "np.ndarray") -> "np.ndarray":
    """
    Row-wise L2 normalization. Leaves zero vectors unchanged.
    """
    _require_numpy()
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0.0, 1.0, norms)
    return x / norms


def _topk_smallest(dist: "np.ndarray", k: int) -> "np.ndarray":
    """
    Return indices of k smallest values per row in dist matrix.
    dist shape: (Q, N)
    """
    _require_numpy()
    if k <= 0:
        raise VectorIndexError("k must be >= 1")
    q, n = dist.shape
    kk = min(k, n)
    # argpartition is O(N) average, then sort within top-k
    idx_part = np.argpartition(dist, kk - 1, axis=1)[:, :kk]
    # sort each row's candidates
    row_indices = np.arange(q)[:, None]
    idx_sorted = idx_part[row_indices, np.argsort(dist[row_indices, idx_part], axis=1)]
    return idx_sorted


class VectorIndex:
    """
    A vector index supporting two backends: HNSW (hnswlib) and brute-force (numpy).

    Typical usage:
      idx = VectorIndex(dim=384, metric="cosine", backend="auto")
      idx.build(max_elements=100_000)
      idx.add(vectors, ids)
      labels, distances = idx.knn(query_vectors, k=10)
      idx.save("data/vectors/my_index")
      idx2 = VectorIndex.load("data/vectors/my_index")
    """

    def __init__(
        self,
        *,
        dim: int,
        metric: Metric = "cosine",
        backend: Backend = "auto",
    ) -> None:
        if int(dim) <= 0:
            raise VectorIndexError("dim must be a positive integer")
        if metric not in ("cosine", "l2"):
            raise VectorIndexError(f"Unsupported metric: {metric!r}")

        self._dim = int(dim)
        self._metric: Metric = metric

        chosen = backend
        if chosen == "auto":
            chosen = "hnsw" if hnswlib is not None else "bruteforce"

        if chosen == "hnsw" and hnswlib is None:
            chosen = "bruteforce"

        if chosen not in ("hnsw", "bruteforce"):
            raise VectorIndexError(f"Unsupported backend: {backend!r}")

        self._backend: Literal["hnsw", "bruteforce"] = chosen  # type: ignore[assignment]

        # HNSW state
        self._hnsw = None
        self._hnsw_M: Optional[int] = None
        self._hnsw_ef_construction: Optional[int] = None
        self._hnsw_space: Optional[str] = None

        # Brute-force state
        self._bf_ids = None
        self._bf_vecs = None
        self._bf_normalized = False

        # Built flag
        self._built = False

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def metric(self) -> Metric:
        return self._metric

    @property
    def backend(self) -> Literal["hnsw", "bruteforce"]:
        return self._backend

    def meta(self) -> VectorIndexMeta:
        return VectorIndexMeta(
            backend=self._backend,
            dim=self._dim,
            metric=self._metric,
            count=self.count(),
            hnsw_space=self._hnsw_space,
            hnsw_M=self._hnsw_M,
            hnsw_ef_construction=self._hnsw_ef_construction,
        )

    def build(
        self,
        *,
        max_elements: int,
        ef_construction: int = 200,
        M: int = 16,
        num_threads: int = 4,
    ) -> None:
        """
        Initialize the index.

        For HNSW:
          - max_elements must be >= total items you plan to add
          - ef_construction and M control build quality/speed
        For brute-force:
          - max_elements is informational; we preallocate nothing
        """
        if int(max_elements) <= 0:
            raise VectorIndexError("max_elements must be a positive integer")

        if self._backend == "hnsw":
            if hnswlib is None:  # pragma: no cover
                raise VectorIndexError(
                    "hnswlib is not installed (cannot use HNSW backend)."
                )

            space = "cosine" if self._metric == "cosine" else "l2"
            idx = hnswlib.Index(space=space, dim=self._dim)
            idx.init_index(
                max_elements=int(max_elements),
                ef_construction=int(ef_construction),
                M=int(M),
            )
            idx.set_num_threads(int(max(1, num_threads)))
            # ef for query can be set later per query; set a reasonable default
            try:
                idx.set_ef(min(200, int(max_elements)))
            except Exception:
                pass

            self._hnsw = idx
            self._hnsw_space = space
            self._hnsw_M = int(M)
            self._hnsw_ef_construction = int(ef_construction)
            self._built = True
            return

        # bruteforce
        _require_numpy()
        self._bf_ids = np.empty((0,), dtype=np.int64)
        self._bf_vecs = np.empty((0, self._dim), dtype=np.float32)
        self._bf_normalized = False
        self._built = True

    def count(self) -> int:
        if not self._built:
            return 0
        if self._backend == "hnsw":
            if self._hnsw is None:
                return 0
            try:
                return int(self._hnsw.get_current_count())
            except Exception:
                return 0
        # bruteforce
        if self._bf_ids is None:
            return 0
        return int(self._bf_ids.shape[0])

    def add(self, vectors, ids: Union[Sequence[int], "np.ndarray"]) -> None:
        """
        Add items to the index.

        Args:
          vectors: array-like of shape (N, dim) or (dim,)
          ids: list/array of N integer labels (unique recommended)
        """
        if not self._built:
            raise VectorIndexError(
                "Index not built. Call build(max_elements=...) first."
            )

        X = _as_float32_matrix(vectors, dim=self._dim)
        lab = _as_int64_labels(ids)
        if X.shape[0] != lab.shape[0]:
            raise VectorIndexError(
                f"vectors N={X.shape[0]} does not match ids N={lab.shape[0]}"
            )

        if self._backend == "hnsw":
            assert self._hnsw is not None
            self._hnsw.add_items(X, lab)
            return

        # bruteforce
        _require_numpy()
        assert self._bf_ids is not None and self._bf_vecs is not None

        # If cosine metric, store normalized vectors to make queries faster/stable
        if self._metric == "cosine":
            X = _normalize_rows(X)
            self._bf_normalized = True

        self._bf_vecs = np.vstack([self._bf_vecs, X])
        self._bf_ids = np.concatenate([self._bf_ids, lab])

    def knn(
        self,
        query_vectors,
        *,
        k: int,
        ef: Optional[int] = None,
    ) -> tuple["np.ndarray", "np.ndarray"]:
        """
        Query K nearest neighbors.

        Returns:
          (labels, distances)
          labels shape: (Q, k)
          distances shape: (Q, k)
        """
        if not self._built:
            raise VectorIndexError("Index not built.")
        if int(k) <= 0:
            raise VectorIndexError("k must be >= 1")

        Q = _as_float32_matrix(query_vectors, dim=self._dim)

        if self._backend == "hnsw":
            assert self._hnsw is not None
            if ef is not None:
                try:
                    self._hnsw.set_ef(int(ef))
                except Exception:
                    pass
            labels, distances = self._hnsw.knn_query(Q, k=int(k))
            # hnswlib returns numpy arrays already
            return labels.astype(np.int64, copy=False), distances.astype(
                np.float32, copy=False
            )

        # bruteforce
        _require_numpy()
        assert self._bf_vecs is not None and self._bf_ids is not None
        if self._bf_vecs.shape[0] == 0:
            return np.empty((Q.shape[0], 0), dtype=np.int64), np.empty(
                (Q.shape[0], 0), dtype=np.float32
            )

        if self._metric == "cosine":
            # Normalize queries; indexed vectors are normalized on insert.
            Qn = _normalize_rows(Q)
            # similarity: (Q, N)
            sim = Qn @ self._bf_vecs.T
            dist = 1.0 - sim
        else:
            # squared L2 distances: ||q-x||^2 = ||q||^2 + ||x||^2 - 2 q·x
            q2 = np.sum(Q * Q, axis=1, keepdims=True)  # (Q,1)
            x2 = np.sum(self._bf_vecs * self._bf_vecs, axis=1, keepdims=True).T  # (1,N)
            dist = q2 + x2 - 2.0 * (Q @ self._bf_vecs.T)

        idx = _topk_smallest(dist, int(k))  # (Q, k)
        row = np.arange(Q.shape[0])[:, None]
        out_ids = self._bf_ids[idx]
        out_dist = dist[row, idx].astype(np.float32, copy=False)
        return out_ids.astype(np.int64, copy=False), out_dist

    def save(self, base_path: Union[str, Path]) -> None:
        """
        Persist the index to disk.

        Args:
          base_path: path without extension. We will write:
            - <base_path>.meta.json
            - <base_path>.hnsw   (if backend=hnsw)
            - <base_path>.npz    (if backend=bruteforce)
        """
        if not self._built:
            raise VectorIndexError("Index not built; nothing to save.")

        p = Path(base_path)
        meta_path = p.with_suffix(".meta.json")

        meta = self.meta()
        meta_path.write_text(
            json.dumps(meta.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        if self._backend == "hnsw":
            assert self._hnsw is not None
            idx_path = p.with_suffix(".hnsw")
            self._hnsw.save_index(str(idx_path))
            return

        # bruteforce
        _require_numpy()
        assert self._bf_vecs is not None and self._bf_ids is not None
        npz_path = p.with_suffix(".npz")
        np.savez_compressed(
            str(npz_path),
            ids=self._bf_ids.astype(np.int64, copy=False),
            vecs=self._bf_vecs.astype(np.float32, copy=False),
            dim=np.int32(self._dim),
            metric=np.array([self._metric], dtype=object),
        )

    @classmethod
    def load(cls, base_path: Union[str, Path]) -> "VectorIndex":
        """
        Load a saved index from disk (auto-detect backend).

        Requires:
          - <base_path>.meta.json
          - backend payload file (.hnsw or .npz)
        """
        p = Path(base_path)
        meta_path = p.with_suffix(".meta.json")
        if not meta_path.exists():
            raise VectorIndexError(f"Missing meta file: {meta_path}")

        meta_raw = json.loads(meta_path.read_text(encoding="utf-8"))
        backend = str(meta_raw.get("backend", "")).strip()
        dim = int(meta_raw.get("dim", 0))
        metric = str(meta_raw.get("metric", "cosine")).strip()
        if metric not in ("cosine", "l2"):
            raise VectorIndexError(f"Unsupported metric in meta: {metric!r}")

        backend_norm: Backend = "auto"
        if backend in ("auto", "hnsw", "bruteforce"):
            backend_norm = cast(Backend, backend)
        idx = cls(dim=dim, metric=metric, backend=backend_norm)

        if idx.backend == "hnsw":
            if hnswlib is None:  # pragma: no cover
                raise VectorIndexError(
                    "Cannot load HNSW index because hnswlib is not installed."
                )
            space = "cosine" if metric == "cosine" else "l2"
            h = hnswlib.Index(space=space, dim=dim)
            idx_path = p.with_suffix(".hnsw")
            if not idx_path.exists():
                raise VectorIndexError(f"Missing HNSW payload file: {idx_path}")
            h.load_index(str(idx_path))
            idx._hnsw = h
            idx._hnsw_space = space
            idx._hnsw_M = meta_raw.get("hnsw_M")
            idx._hnsw_ef_construction = meta_raw.get("hnsw_ef_construction")
            idx._built = True
            return idx

        # bruteforce
        _require_numpy()
        npz_path = p.with_suffix(".npz")
        if not npz_path.exists():
            raise VectorIndexError(f"Missing brute-force payload file: {npz_path}")

        data = np.load(str(npz_path), allow_pickle=True)
        ids = data["ids"].astype(np.int64, copy=False)
        vecs = data["vecs"].astype(np.float32, copy=False)
        if vecs.ndim != 2 or vecs.shape[1] != dim:
            raise VectorIndexError(
                f"Invalid vectors in {npz_path}: expected (N,{dim}), got {vecs.shape}"
            )

        idx._bf_ids = ids
        idx._bf_vecs = vecs
        idx._bf_normalized = metric == "cosine"
        idx._built = True
        return idx
