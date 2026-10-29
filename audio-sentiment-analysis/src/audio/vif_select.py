"""Variance Inflation Factor feature pruning (threshold 10)."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor


def vif_select(df: pd.DataFrame, feature_cols: List[str], threshold: float = 10.0) -> Tuple[List[str], pd.DataFrame]:
    """Iteratively drop highest-VIF feature until all <= threshold."""
    cols = list(feature_cols)
    history = []
    while len(cols) > 2:
        x = df[cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        # standardize for numerical stability
        x = (x - x.mean(0)) / (x.std(0) + 1e-8)
        vifs = []
        for i in range(x.shape[1]):
            try:
                v = float(variance_inflation_factor(x, i))
            except Exception:
                v = float("inf")
            vifs.append(v)
        vmax = max(vifs)
        if vmax <= threshold:
            break
        drop_i = int(np.argmax(vifs))
        history.append({"feature": cols[drop_i], "vif": vmax})
        cols.pop(drop_i)
    hist_df = pd.DataFrame(history)
    return cols, hist_df
