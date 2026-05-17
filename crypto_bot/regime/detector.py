"""HMM and XGBoost regime classifiers."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from crypto_bot import config

logger = logging.getLogger(__name__)

REGIME_NAMES = ("trending", "ranging", "breakout")


def _label_regimes_rules(df: pd.DataFrame) -> np.ndarray:
    """Supervised labels: 0=trending, 1=ranging, 2=breakout."""
    labels = np.full(len(df), 1, dtype=int)
    adx = df["adx"].fillna(0)
    hurst = df["hurst"].fillna(0.5)
    vol_ratio = df["vol_ratio"].fillna(1.0)

    labels[adx > 25] = 0
    labels[(adx < 20) & (hurst < 0.48)] = 1
    labels[vol_ratio > 1.5] = 2
    return labels


class RegimeDetector:
    """Unified regime detector with predict_proba interface."""

    def __init__(self, model_type: str | None = None) -> None:
        self.model_type: Literal["hmm", "xgboost"] = (
            model_type or config.REGIME_MODEL
        )  # type: ignore[assignment]
        self._hmm: GaussianHMM | None = None
        self._xgb: XGBClassifier | None = None
        self._scaler = StandardScaler()
        self._state_map: dict[int, int] = {}
        self._fitted = False

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        close = df["close"]
        log_ret = np.log(close / close.shift(1)).fillna(0)
        realized_vol = log_ret.rolling(20).std().fillna(log_ret.std())
        if self.model_type == "xgboost":
            cols = ["adx", "hurst", "vol_ratio", "atr_pct", "bb_width", "rsi"]
            X = df[cols].ffill().fillna(0).values
        else:
            X = np.column_stack([log_ret.values, realized_vol.values])
        return X

    def fit(self, df: pd.DataFrame) -> None:
        X = self._build_features(df)
        if self.model_type == "hmm":
            X_scaled = self._scaler.fit_transform(X)
            self._hmm = GaussianHMM(
                n_components=3, covariance_type="full", n_iter=200, random_state=42
            )
            self._hmm.fit(X_scaled)
            means = self._hmm.means_[:, 0]
            covars = self._hmm.covars_
            vols = []
            for i in range(3):
                c = covars[i]
                vols.append(float(np.sqrt(c[1, 1]) if c.ndim == 2 else np.sqrt(c[1])))
            order = np.argsort(means)
            ranging_state = order[0]
            trending_state = order[2]
            breakout_state = int(np.argmax(vols))
            self._state_map = {
                trending_state: 0,
                ranging_state: 1,
                breakout_state: 2,
            }
            for i in range(3):
                if i not in self._state_map:
                    self._state_map[i] = 1
        else:
            y = _label_regimes_rules(df)
            self._xgb = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                objective="multi:softprob",
                num_class=3,
                random_state=42,
            )
            self._xgb.fit(X, y)
        self._fitted = True
        logger.info("Regime detector fitted (%s)", self.model_type)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return (N, 3) probability matrix [trending, ranging, breakout]."""
        if not self._fitted:
            raise RuntimeError("RegimeDetector must be fit before predict_proba")

        X = self._build_features(df)
        n = len(X)
        probs = np.zeros((n, 3))

        if self.model_type == "hmm" and self._hmm is not None:
            X_scaled = self._scaler.transform(X)
            raw = self._hmm.predict_proba(X_scaled)
            for state_idx in range(3):
                regime_idx = self._state_map.get(state_idx, 1)
                probs[:, regime_idx] += raw[:, state_idx]
        elif self._xgb is not None:
            raw = self._xgb.predict_proba(X)
            probs = np.zeros((n, 3))
            for col_idx, cls in enumerate(self._xgb.classes_):
                regime_idx = int(cls)
                if 0 <= regime_idx < 3:
                    probs[:, regime_idx] = raw[:, col_idx]

        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        return probs / row_sums

    def latest_regime_dict(self, df: pd.DataFrame) -> dict[str, float]:
        """Return regime probabilities for the latest bar."""
        proba = self.predict_proba(df)
        if len(proba) == 0:
            return {name: 1.0 / 3 for name in REGIME_NAMES}
        last = proba[-1]
        return {
            REGIME_NAMES[i]: float(last[i])
            for i in range(3)
        }
