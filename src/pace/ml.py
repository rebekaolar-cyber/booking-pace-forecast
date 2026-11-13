"""A gradient-boosting model on the same signals as the pace forecast.

It is trained only on days before the test period (a strict time split), so it never sees the
future it is scored on. One model covers both hotels and every horizon.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

FEATURES = [
    "horizon", "weekday", "month", "is_city",
    "weekday_average", "booked_now", "booked_normally", "pace_ratio", "same_day_last_year",
]


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    f = frame.copy()
    f["weekday"] = f.day.dt.day_of_week
    f["month"] = f.day.dt.month
    f["is_city"] = (f.hotel == "City Hotel").astype(int)
    f["pace_ratio"] = (f.booked_now + 1) / (f.booked_normally + 1)
    return f


def train(frame: pd.DataFrame, seed: int = 0) -> HistGradientBoostingRegressor:
    f = add_features(frame)
    model = HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=400, learning_rate=0.05,
        max_leaf_nodes=31, min_samples_leaf=40, random_state=seed,
    )
    model.fit(f[FEATURES], f.actual)
    return model


def predict(model: HistGradientBoostingRegressor, frame: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(add_features(frame)[FEATURES]), 0, None)
