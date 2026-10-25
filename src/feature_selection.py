"""Random Forest importance ranking to choose the control variables."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def rank_control_variables(X: pd.DataFrame, y_energy: pd.Series, y_yield: pd.Series,
                           n_estimators: int = 100, random_state: int = 42) -> pd.Series:
    """Average the RF feature importances for both objectives, sorted high to low."""
    rf_energy = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf_yield = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf_energy.fit(X, y_energy)
    rf_yield.fit(X, y_yield)

    imp_energy = pd.Series(rf_energy.feature_importances_, index=X.columns)
    imp_yield = pd.Series(rf_yield.feature_importances_, index=X.columns)
    return ((imp_energy + imp_yield) / 2).sort_values(ascending=False)


def select_control_variables(X, y_energy, y_yield, top_k: int = 10, **kwargs) -> list[str]:
    """Return the top-k features by combined importance."""
    return rank_control_variables(X, y_energy, y_yield, **kwargs).head(top_k).index.tolist()
