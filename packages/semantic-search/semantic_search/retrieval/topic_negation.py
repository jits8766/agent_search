"""Topic / category negation in the residual embedding space (item 7).

"Domains not in tech and not in finance" cannot be answered by SLD-substring
exclusion — the constraint is semantic. This module builds per-topic centroids
from seed phrases (encoded with the same encoder as the residual) and subtracts
them from the query vector before ANN, pushing excluded topics away in cosine
space:

    q' = normalize(q - alpha * sum(centroid[t] for t in excluded_topics))

All numeric behaviour is parameterised (``alpha``); the seed phrases live in a
config file so topics are tunable without code change.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np


def _unit(vec: "np.ndarray") -> "np.ndarray":
    """Return the L2-normalised vector; unchanged when its norm is ~0.
    :param vec: np.ndarray - Input vector
    :return: np.ndarray - Unit vector (float32)
    """
    v = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if norm <= 1e-12:
        return v
    return v / norm


def build_category_centroids(seeds_by_topic: Dict[str, List[str]], encode_batch) -> Dict[str, "np.ndarray"]:
    """Build a unit centroid per topic by mean-pooling its encoded seed phrases.

    :param seeds_by_topic: Dict[str, List[str]] - Topic -> seed phrases
    :param encode_batch: Callable[[List[str]], Sequence[Sequence[float]]] - Encoder
    :return: Dict[str, np.ndarray] - Topic -> unit centroid vector
    """
    centroids: Dict[str, "np.ndarray"] = {}
    for topic, seeds in seeds_by_topic.items():
        phrases = [s for s in seeds if isinstance(s, str) and s.strip()]
        if not phrases:
            continue
        vecs = np.asarray(encode_batch(phrases), dtype=np.float32)
        if vecs.ndim != 2 or vecs.shape[0] == 0:
            continue
        centroids[str(topic).lower()] = _unit(vecs.mean(axis=0))
    return centroids


def subtract_topics(query_vec: Sequence[float], excluded: Sequence[str], centroids: Dict[str, "np.ndarray"], alpha: float) -> "np.ndarray":
    """Return the unit query vector with excluded-topic centroids subtracted.

    Topics absent from ``centroids`` are ignored. When nothing is subtracted the
    normalised input is returned unchanged.

    :param query_vec: Sequence[float] - Original query embedding
    :param excluded: Sequence[str] - Topic names to push away
    :param centroids: Dict[str, np.ndarray] - Topic -> unit centroid
    :param alpha: float - Subtraction strength (>= 0)
    :return: np.ndarray - Adjusted unit vector
    """
    q = np.asarray(query_vec, dtype=np.float32)
    if alpha <= 0.0 or not excluded or not centroids:
        return _unit(q)
    acc = np.zeros_like(q)
    used = False
    for topic in excluded:
        c = centroids.get(str(topic).lower())
        if c is not None:
            acc = acc + c
            used = True
    if not used:
        return _unit(q)
    return _unit(q - float(alpha) * acc)
