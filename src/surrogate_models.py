"""Gaussian Process surrogates and baseline model comparison (temporal split)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def make_kernel(n_features: int):
    """Anisotropic kernel: C * RBF(ARD) + WhiteKernel."""
    return (C(1.0, (1e-3, 1e3))
            * RBF(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e2))
            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-6, 1e1)))


def make_gp(n_features: int, random_state: int = 42) -> GaussianProcessRegressor:
    return GaussianProcessRegressor(
        kernel=make_kernel(n_features),
        n_restarts_optimizer=5,
        alpha=1e-6,
        normalize_y=True,
        random_state=random_state,
    )


def scale_controls(X_ctrl: pd.DataFrame):
    """Standardize the control variables. Returns (X_scaled, fitted_scaler)."""
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_ctrl), columns=X_ctrl.columns)
    return X_scaled, scaler


def temporal_split(X, y_energy, y_yield, train_frac: float = 0.8):
    """Chronological split: first 80% for training, last 20% for testing (no shuffling)."""
    idx = int(train_frac * len(X))
    return (X.iloc[:idx], X.iloc[idx:],
            y_energy.iloc[:idx], y_energy.iloc[idx:],
            y_yield.iloc[:idx], y_yield.iloc[idx:])


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def compare_models(X_train, X_test, yE_train, yE_test, yY_train, yY_test,
                   random_state: int = 42):
    """Benchmark LR, RF, GBR and GP on both targets.

    Returns (metrics_df, gp_energy, gp_yield) with the GP models fitted on the training split.
    """
    baselines = {
        "LinearRegression": lambda: LinearRegression(),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=200, random_state=random_state),
        "GradientBoosting": lambda: GradientBoostingRegressor(random_state=random_state),
    }
    targets = {
        "energy_index": (yE_train, yE_test),
        "yield_efficiency_index": (yY_train, yY_test),
    }

    rows = []
    for target, (y_tr, y_te) in targets.items():
        for name, factory in baselines.items():
            model = factory().fit(X_train, y_tr)
            rows.append({"Target": target, "Model": name,
                         **regression_metrics(y_te, model.predict(X_test))})

    n_features = X_train.shape[1]
    gp_energy = make_gp(n_features, random_state).fit(X_train, yE_train)
    gp_yield = make_gp(n_features, random_state).fit(X_train, yY_train)
    rows.append({"Target": "energy_index", "Model": "GaussianProcess",
                 **regression_metrics(yE_test, gp_energy.predict(X_test))})
    rows.append({"Target": "yield_efficiency_index", "Model": "GaussianProcess",
                 **regression_metrics(yY_test, gp_yield.predict(X_test))})

    return pd.DataFrame(rows), gp_energy, gp_yield


class SurrogatePair:
    """Wraps the two GP surrogates and the scaler so the optimizer can query raw control values."""

    def __init__(self, gp_energy, gp_yield, scaler, control_vars):
        self.gp_energy = gp_energy
        self.gp_yield = gp_yield
        self.scaler = scaler
        self.control_vars = list(control_vars)

    def _scale(self, x):
        return self.scaler.transform(pd.DataFrame([x], columns=self.control_vars))

    def predict(self, x):
        """Return (energy_mean, energy_std, yield_mean, yield_std) for one control vector."""
        xs = self._scale(x)
        e_mean, e_std = self.gp_energy.predict(xs, return_std=True)
        y_mean, y_std = self.gp_yield.predict(xs, return_std=True)
        return e_mean[0], e_std[0], y_mean[0], y_std[0]
