"""Decision support: scenario profiling of the Pareto front and result figures."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def profile_scenarios(df_pareto: pd.DataFrame) -> pd.DataFrame:
    """Pick three operating profiles from the Pareto front.

    - Conservative: lowest predicted energy (protects energy cost, accepts lower yield)
    - Balanced:     knee point, closest to the ideal point after min-max normalization
    - Aggressive:   highest predicted yield (maximizes output, accepts higher energy)

    Each profile reports its control settings, predictions and GP uncertainty.
    """
    df = df_pareto.reset_index(drop=True)
    e = (df["pred_energy"] - df["pred_energy"].min()) / (np.ptp(df["pred_energy"]) or 1)
    y = (df["pred_yield"].max() - df["pred_yield"]) / (np.ptp(df["pred_yield"]) or 1)

    picks = {
        "Conservative": int(df["pred_energy"].idxmin()),
        "Balanced": int(np.argmin(np.sqrt(e ** 2 + y ** 2))),
        "Aggressive": int(df["pred_yield"].idxmax()),
    }
    out = df.loc[list(picks.values())].copy()
    out.insert(0, "scenario", list(picks.keys()))
    return out.reset_index(drop=True)


# --------------------------------------------------------------------------- figures

def _save(fig_dir, name):
    if fig_dir:
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(os.path.join(fig_dir, name), dpi=300, bbox_inches="tight")


def plot_correlation_heatmap(df_fe, control_vars, fig_dir=None, show=False):
    cols = list(control_vars) + ["energy_index", "yield_efficiency_index"]
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_fe[cols].corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Correlation Heatmap: Control Variables and Targets")
    plt.tight_layout()
    _save(fig_dir, "correlation_heatmap.png")
    plt.show() if show else plt.close()


def plot_pareto_front(df_pareto, fig_dir=None, show=False):
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(df_pareto["pred_energy"], df_pareto["pred_yield"],
                     c=(df_pareto["energy_std"] + df_pareto["yield_std"]) / 2, s=60)
    plt.xlabel("Predicted Energy Index")
    plt.ylabel("Predicted Yield Efficiency")
    plt.title("Pareto Frontier from GP Surrogates")
    plt.colorbar(sc, label="Average Predictive Uncertainty")
    plt.tight_layout()
    _save(fig_dir, "pareto_frontier_gp.png")
    plt.show() if show else plt.close()


def plot_convergence(history, fig_dir=None, show=False):
    for key, label in [("best_energy", "Energy Index"), ("best_yield", "Yield Efficiency")]:
        plt.figure(figsize=(8, 5))
        plt.plot(history[key], label=f"Best {label.split()[0]}")
        plt.xlabel("Generation")
        plt.ylabel(label)
        plt.title(f"Convergence: Best {label.split()[0]} Over Generations")
        plt.legend()
        plt.tight_layout()
        _save(fig_dir, f"convergence_{key.split('_')[1]}.png")
        plt.show() if show else plt.close()


def plot_actual_vs_pred(y_true, y_pred, title, filename, xlab, ylab, fig_dir=None, show=False):
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, alpha=0.7)
    mn, mx = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
    plt.plot([mn, mx], [mn, mx], "--")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.tight_layout()
    _save(fig_dir, filename)
    plt.show() if show else plt.close()


def plot_scenario_uncertainty(df_pareto, fig_dir=None, show=False):
    df = df_pareto.reset_index(drop=True)
    x = np.arange(len(df))
    labels = [f"Scenario {i + 1}" for i in x]
    for col, std, name in [("pred_energy", "energy_std", "Energy"),
                           ("pred_yield", "yield_std", "Yield")]:
        plt.figure(figsize=(10, 6))
        plt.errorbar(x, df[col], yerr=df[std], fmt="o", capsize=5, label=name)
        plt.xticks(x[::10], labels[::10], rotation=45, ha="right")
        plt.ylabel(f"Predicted {name} {'Index' if name == 'Energy' else 'Efficiency'}")
        plt.title(f"Scenario Recommendations with GP Uncertainty: {name}")
        plt.tight_layout()
        _save(fig_dir, f"scenario_uncertainty_{name.lower()}.png")
        plt.show() if show else plt.close()
