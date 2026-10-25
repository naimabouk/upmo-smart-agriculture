"""End-to-end UPMO pipeline.

Usage:
    python run_pipeline.py                         # reproduce the published results
    python run_pipeline.py --beta 0.2              # uncertainty-penalized fitness
    python run_pipeline.py --out my_results --generations 100
"""

import argparse
import os

from src import decision_support as ds
from src.feature_selection import select_control_variables
from src.optimization import pareto_metrics, run_optimization
from src.preprocessing import prepare, split_features_targets
from src.surrogate_models import SurrogatePair, compare_models, scale_controls, temporal_split


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data", default="data/raw/synthetic_agritech_sensor_dataset.csv")
    p.add_argument("--out", default="results_reproduced")
    p.add_argument("--top-k", type=int, default=10, help="number of control variables")
    p.add_argument("--pop-size", type=int, default=80)
    p.add_argument("--generations", type=int, default=50)
    p.add_argument("--beta", type=float, default=0.0,
                   help="uncertainty penalty weight (0 = posterior mean, as in the paper)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    fig_dir = os.path.join(args.out, "figures")
    met_dir = os.path.join(args.out, "metrics")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(met_dir, exist_ok=True)

    # 1. Preprocessing
    df_fe = prepare(args.data)
    X, y_energy, y_yield = split_features_targets(df_fe)
    print(f"Engineered dataset: {df_fe.shape}")

    # 2. Control variable selection
    control_vars = select_control_variables(X, y_energy, y_yield, top_k=args.top_k)
    print("Selected control variables:", control_vars)
    X_ctrl = df_fe[control_vars]
    ds.plot_correlation_heatmap(df_fe, control_vars, fig_dir)

    # 3. Surrogate models on a chronological 80/20 split
    X_scaled, scaler = scale_controls(X_ctrl)
    X_tr, X_te, yE_tr, yE_te, yY_tr, yY_te = temporal_split(X_scaled, y_energy, y_yield)
    df_cmp, gp_energy, gp_yield = compare_models(X_tr, X_te, yE_tr, yE_te, yY_tr, yY_te)
    df_cmp.to_csv(os.path.join(met_dir, "model_comparison_temporal.csv"), index=False)
    print(df_cmp.sort_values(["Target", "MAE"]).to_string(index=False))

    ds.plot_actual_vs_pred(yE_te, gp_energy.predict(X_te),
                           "Actual vs Predicted Energy (GP, Temporal Split)",
                           "actual_vs_pred_energy_gp.png",
                           "Actual Energy Index", "Predicted Energy Index", fig_dir)
    ds.plot_actual_vs_pred(yY_te, gp_yield.predict(X_te),
                           "Actual vs Predicted Yield (GP, Temporal Split)",
                           "actual_vs_pred_yield_gp.png",
                           "Actual Yield Efficiency", "Predicted Yield Efficiency", fig_dir)

    # 4. Multi-objective optimization
    surrogates = SurrogatePair(gp_energy, gp_yield, scaler, control_vars)
    df_pareto, history = run_optimization(surrogates, X_ctrl, pop_size=args.pop_size,
                                          generations=args.generations, beta=args.beta,
                                          seed=args.seed, verbose=False)
    df_pareto.to_csv(os.path.join(met_dir, "pareto_solutions_gp.csv"), index=False)
    df_pm = pareto_metrics(df_pareto)
    df_pm.to_csv(os.path.join(met_dir, "pareto_metrics.csv"), index=False)
    print(df_pm.to_string(index=False))

    ds.plot_pareto_front(df_pareto, fig_dir)
    ds.plot_convergence(history, fig_dir)

    # 5. Decision support
    df_scen = ds.profile_scenarios(df_pareto)
    df_scen.to_csv(os.path.join(met_dir, "scenario_profiles.csv"), index=False)
    print(df_scen[["scenario", "pred_energy", "energy_std", "pred_yield", "yield_std"]]
          .to_string(index=False))
    ds.plot_scenario_uncertainty(df_pareto, fig_dir)

    print(f"\nDone. Results written to {args.out}/")


if __name__ == "__main__":
    main()
