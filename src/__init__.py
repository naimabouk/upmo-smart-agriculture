"""UPMO: Uncertainty-aware Pareto Multi-objective Optimization for smart agriculture.

Pipeline modules:
    preprocessing      -> KNN imputation and temporal feature engineering
    feature_selection  -> Random Forest importance ranking of control variables
    surrogate_models   -> Gaussian Process surrogates and baseline comparison
    optimization       -> Pareto-based evolutionary search (NSGA-II style) and front metrics
    decision_support   -> Scenario profiling and uncertainty figures
"""

__version__ = "1.0.0"
