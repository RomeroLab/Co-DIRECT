
from __future__ import annotations

from typing import NamedTuple

import numpy as np

class Reduction(NamedTuple):

    features: np.ndarray
    explained_variance: float
    n_components: int

def reduce_intermediate_block(
    X: np.ndarray,
    *,
    width: int,
    train: np.ndarray,
    n_components: int,
) -> Reduction:

    from sklearn.decomposition import PCA

    if n_components <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}")
    n_train = int(train.sum())
    if n_train == 0:
        raise ValueError("train split is empty; cannot fit a reduction basis")
    if width <= 0 or width > X.shape[1]:
        raise ValueError(f"width {width} outside the {X.shape[1]} available columns")

    base, extra = X[:, :-width], X[:, -width:]
    reducer = PCA(n_components=min(n_components, n_train, width), random_state=0)
    reducer.fit(extra[train])
    return Reduction(
        features=np.concatenate([base, reducer.transform(extra)], axis=1),
        explained_variance=float(reducer.explained_variance_ratio_.sum()),
        n_components=int(reducer.n_components_),
    )
