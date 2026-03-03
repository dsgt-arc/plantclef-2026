#!/usr/bin/env python3
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class CoGraph:
    num_classes: int
    indptr: np.ndarray          # shape [C+1]
    indices: np.ndarray         # shape [E]
    weights: np.ndarray         # shape [E], float32


def _build_csr(num_classes: int, neigh_i: np.ndarray, neigh_j: np.ndarray, neigh_w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    neigh_i = neigh_i.astype(np.int64, copy=False)
    neigh_j = neigh_j.astype(np.int64, copy=False)
    neigh_w = neigh_w.astype(np.float32, copy=False)

    if neigh_i.shape != neigh_j.shape or neigh_i.shape != neigh_w.shape:
        raise ValueError(f"Graph arrays must have same shape. Got {neigh_i.shape}, {neigh_j.shape}, {neigh_w.shape}")

    if neigh_i.ndim != 1:
        raise ValueError("Graph arrays must be 1D")

    if neigh_i.size == 0:
        raise ValueError("Graph is empty")

    if neigh_i.min() < 0 or neigh_i.max() >= num_classes:
        raise ValueError("neigh_i has out of range indices")
    if neigh_j.min() < 0 or neigh_j.max() >= num_classes:
        raise ValueError("neigh_j has out of range indices")

    order = np.argsort(neigh_i, kind="stable")
    neigh_i = neigh_i[order]
    neigh_j = neigh_j[order]
    neigh_w = neigh_w[order]

    indptr = np.zeros(num_classes + 1, dtype=np.int64)
    counts = np.bincount(neigh_i, minlength=num_classes)
    indptr[1:] = np.cumsum(counts, dtype=np.int64)

    indices = neigh_j.astype(np.int32, copy=False)
    weights = neigh_w.astype(np.float32, copy=False)
    return indptr, indices, weights


def load_cograph_npz(path: str, num_classes: int = 7806) -> CoGraph:
    z = np.load(path)
    for k in ["neigh_i", "neigh_j", "neigh_w"]:
        if k not in z:
            raise KeyError(f"Missing {k} in NPZ: {path}")

    indptr, indices, weights = _build_csr(num_classes, z["neigh_i"], z["neigh_j"], z["neigh_w"])
    return CoGraph(num_classes=num_classes, indptr=indptr, indices=indices, weights=weights)


def _pick_seeds(p: np.ndarray, seed_k: int, seed_alpha: float) -> np.ndarray:
    if seed_k <= 0:
        raise ValueError("seed_k must be > 0")

    k = min(seed_k, p.shape[0])
    idx = np.argpartition(p, -k)[-k:]
    idx = idx[np.argsort(p[idx])[::-1]]

    if seed_alpha > 0:
        pmax = float(p[idx[0]])
        keep = [int(i) for i in idx.tolist() if float(p[i]) >= seed_alpha * pmax]
        if keep:
            return np.asarray(keep, dtype=np.int32)

    return idx.astype(np.int32, copy=False)


def graph_rerank(
    p: np.ndarray,
    g: CoGraph,
    seed_k: int = 8,
    neigh_topn: int = 50,
    lam: float = 0.10,
    seed_alpha: float = 0.0,
    weight_transform: str = "log1p",
    w_cap: float = 5.0,
) -> np.ndarray:
    """
    One step graph boost:
      p' = clip(p + lam * sum_{i in seeds} p_i * f(w_ij), 0, 1)
    """
    if p.ndim != 1 or p.shape[0] != g.num_classes:
        raise ValueError(f"p must be shape [{g.num_classes}]")

    if lam <= 0:
        return p

    seeds = _pick_seeds(p, seed_k=seed_k, seed_alpha=seed_alpha)
    out = p.astype(np.float32, copy=True)

    for si in seeds:
        start = int(g.indptr[si])
        end = int(g.indptr[si + 1])
        if start == end:
            continue

        js = g.indices[start:end]
        ws = g.weights[start:end]

        if neigh_topn > 0 and js.shape[0] > neigh_topn:
            js = js[:neigh_topn]
            ws = ws[:neigh_topn]

        if weight_transform == "log1p":
            w_eff = np.log1p(np.maximum(ws, 0.0))
        elif weight_transform == "cap":
            w_eff = np.minimum(ws, w_cap)
        elif weight_transform == "none":
            w_eff = ws
        else:
            raise ValueError(f"Unknown weight_transform: {weight_transform}")

        out[js] += float(lam) * float(p[si]) * w_eff

    np.clip(out, 0.0, 1.0, out=out)
    return out


def graph_mask_expand(
    p: np.ndarray,
    g: CoGraph,
    seed_k: int = 8,
    neigh_topn: int = 50,
    seed_alpha: float = 0.0,
) -> np.ndarray:
    """
    Image specific mask expansion:
      allowed = seeds union neighbors(seeds)
      p' = p on allowed else 0
    """
    if p.ndim != 1 or p.shape[0] != g.num_classes:
        raise ValueError(f"p must be shape [{g.num_classes}]")

    seeds = _pick_seeds(p, seed_k=seed_k, seed_alpha=seed_alpha)

    allowed = np.zeros(g.num_classes, dtype=np.bool_)
    allowed[seeds] = True

    for si in seeds:
        start = int(g.indptr[si])
        end = int(g.indptr[si + 1])
        if start == end:
            continue

        js = g.indices[start:end]
        if neigh_topn > 0 and js.shape[0] > neigh_topn:
            js = js[:neigh_topn]
        allowed[js] = True

    out = p.astype(np.float32, copy=True)
    out[~allowed] = 0.0
    return out