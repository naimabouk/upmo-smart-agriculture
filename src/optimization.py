"""Pareto-based multi-objective evolutionary search over the GP surrogates.

Objectives (both minimized):
    f1 = energy_mean + beta * energy_std
    f2 = -(yield_mean - beta * yield_std)

With beta = 0 the search uses the GP posterior mean (this setting produced the published
results). With beta > 0 the fitness is uncertainty-penalized: solutions in poorly explored
regions of the control space are penalized in proportion to the GP predictive std.

The loop is an elitist, Pareto-dominance-based GA in the spirit of NSGA-II: parents are
drawn from the current non-dominated front, children come from blend crossover and Gaussian
mutation, and the next generation keeps the non-dominated set of parents plus offspring.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd


def dominates(a, b) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def pareto_front(pop, scores):
    """Return (front, front_scores): the non-dominated members of pop."""
    front, front_scores = [], []
    for i, s in enumerate(scores):
        if not any(dominates(scores[j], s) for j in range(len(scores)) if j != i):
            front.append(pop[i])
            front_scores.append(s)
    return front, front_scores


def create_individual(bounds):
    return [random.uniform(lo, hi) for lo, hi in bounds]


def crossover(p1, p2):
    """Blend (arithmetic) crossover with a random mixing weight."""
    alpha = random.random()
    return [alpha * a + (1 - alpha) * b for a, b in zip(p1, p2)]


def mutate(ind, bounds, rate: float = 0.15):
    """Gaussian mutation (sigma = 10% of the variable range), clipped to bounds."""
    child = ind.copy()
    for i in range(len(child)):
        if random.random() < rate:
            lo, hi = bounds[i]
            child[i] += np.random.normal(0, 0.1 * (hi - lo))
            child[i] = min(max(child[i], lo), hi)
    return child


def make_objective(surrogates, beta: float = 0.0):
    """Build the 2-objective fitness function from a SurrogatePair."""
    def objective(x):
        e_mean, e_std, y_mean, y_std = surrogates.predict(x)
        return [e_mean + beta * e_std, -(y_mean - beta * y_std)]
    return objective


def run_optimization(surrogates, X_ctrl: pd.DataFrame, pop_size: int = 80,
                     generations: int = 50, mutation_rate: float = 0.15,
                     beta: float = 0.0, seed: int = 42, verbose: bool = True):
    """Run the evolutionary search.

    Returns (df_pareto, history) where df_pareto holds the control settings of each
    non-dominated solution with its predicted energy/yield and GP std, and history holds the
    best energy and best yield per generation.
    """
    random.seed(seed)
    np.random.seed(seed)

    control_vars = surrogates.control_vars
    bounds = [(float(X_ctrl[c].min()), float(X_ctrl[c].max())) for c in control_vars]
    objective = make_objective(surrogates, beta)

    population = [create_individual(bounds) for _ in range(pop_size)]
    history = {"best_energy": [], "best_yield": []}

    for gen in range(generations):
        scores = [objective(ind) for ind in population]
        front, front_scores = pareto_front(population, scores)
        history["best_energy"].append(min(s[0] for s in front_scores))
        history["best_yield"].append(max(-s[1] for s in front_scores))

        offspring = []
        while len(offspring) < pop_size:
            pool = front if len(front) >= 2 else population
            p1, p2 = random.sample(pool, 2)
            offspring.append(mutate(crossover(p1, p2), bounds, rate=mutation_rate))

        combined = population + offspring
        front_combined, _ = pareto_front(combined, [objective(ind) for ind in combined])

        if len(front_combined) >= pop_size:
            population = front_combined[:pop_size]
        else:
            remaining = [ind for ind in combined if ind not in front_combined]
            random.shuffle(remaining)
            population = front_combined + remaining[:pop_size - len(front_combined)]

        if verbose:
            print(f"Generation {gen + 1}/{generations} complete")

    final_scores = [objective(ind) for ind in population]
    pareto_solutions, _ = pareto_front(population, final_scores)

    rows = []
    for sol in pareto_solutions:
        e_mean, e_std, y_mean, y_std = surrogates.predict(sol)
        row = dict(zip(control_vars, sol))
        row.update(pred_energy=e_mean, pred_yield=y_mean, energy_std=e_std, yield_std=y_std)
        rows.append(row)

    df_pareto = (pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
                 .sort_values(["pred_energy", "pred_yield"], ascending=[True, False]))
    return df_pareto, history


def compute_hypervolume_2d(df_pareto: pd.DataFrame) -> float:
    """2-D hypervolume in (energy, -yield) space; reference point = worst value + 1 on each axis."""
    pts = df_pareto[["pred_energy", "pred_yield"]].copy()
    pts["neg_yield"] = -pts["pred_yield"]
    points = pts[["pred_energy", "neg_yield"]].sort_values("pred_energy").values

    ref_energy = points[:, 0].max() + 1.0
    ref_neg_yield = points[:, 1].max() + 1.0

    hv, prev_energy = 0.0, ref_energy
    for e, ny in points[::-1]:
        hv += max(prev_energy - e, 0) * max(ref_neg_yield - ny, 0)
        prev_energy = e
    return hv


def compute_spacing(df_pareto: pd.DataFrame) -> float:
    """Schott's spacing: std of nearest-neighbour distances along the front (lower = more uniform)."""
    pts = df_pareto[["pred_energy", "pred_yield"]].values
    if len(pts) < 2:
        return 0.0
    dists = np.array([np.min(np.linalg.norm(np.delete(pts, i, axis=0) - pts[i], axis=1))
                      for i in range(len(pts))])
    return float(np.sqrt(np.mean((dists - dists.mean()) ** 2)))


def pareto_metrics(df_pareto: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        "hypervolume": compute_hypervolume_2d(df_pareto),
        "spacing": compute_spacing(df_pareto),
        "nondominated_count": len(df_pareto),
    }])
