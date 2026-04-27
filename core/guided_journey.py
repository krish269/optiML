"""
Story-first guided journey for OptiMLFlow.

Phase 1 provides a low-decision AutoML baseline.
Phase 2 routes users to business-oriented domain workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from core.chart_generator import (
    plot_cluster_scatter_2d,
    plot_confusion_matrix_generic,
    plot_elbow_curve,
    plot_feature_correlation_lollipop,
    plot_model_score_comparison,
    plot_precision_recall_curve_binary,
    plot_probability_distribution,
    plot_regression_predictions,
    plot_residuals_analysis,
    plot_roc_curve_binary,
    plot_silhouette_analysis,
    plot_target_distribution,
    plot_threshold_metric_curves,
)
from core.churn_analysis import plot_threshold_analysis
from core.domain_workflows import (
    render_customer_segmentation_tab,
    render_ecommerce_recommendation_tab,
    render_iot_anomaly_tab,
    render_upsell_probability_tab,
)
from core.model_training import (
    create_clustering_performance_dataframe,
    create_performance_dataframe,
    detect_task_type,
    get_available_models,
    train_clustering_models,
    train_multiple_models,
)
from core.preprocessing import identify_column_types, preprocess_and_split


@dataclass
class Phase2Module:
    """Metadata for a business workflow module."""

    label: str
    about: str
    business_problem: str
    requirements: list[tuple[str, str, str]]
    use_case: str
    output_focus: str


PHASE2_MODULES: dict[str, Phase2Module] = {
    "Customer Analytics": Phase2Module(
        label="Customer Analytics (Churn, Upsell, Segmentation)",
        about=(
            "Score customer behavior, identify churn risk, estimate upsell probability, "
            "and discover actionable segments."
        ),
        business_problem=(
            "Reduces revenue leakage from churn and improves campaign efficiency by "
            "prioritizing high-impact customer actions."
        ),
        requirements=[
            ("customer_id", "string or integer", "Entity key for ranking and segmentation"),
            ("target/churn/upsell", "binary label", "Supervised learning target"),
            ("usage / plan / tenure / spend", "numeric", "Behavior and value signals"),
            ("contract / channel / region", "categorical", "Context for model discrimination"),
        ],
        use_case="Telecom retention team prioritizes at-risk subscribers and next-best-offer campaigns.",
        output_focus=(
            "Business insights: who to retain first, who to upsell first, and what traits define each segment."
        ),
    ),
    "E-commerce": Phase2Module(
        label="E-commerce Recommendation Systems",
        about=(
            "Generate recommendation candidates from interaction data and optionally train "
            "purchase propensity models."
        ),
        business_problem=(
            "Increases conversion and average order value by surfacing products with the "
            "highest contextual relevance."
        ),
        requirements=[
            ("user_id or customer_id", "string or integer", "Primary shopper identity"),
            ("product_id / item_id", "string or integer", "Catalog item identity"),
            ("timestamp", "datetime", "Optional sequence and recency context"),
            ("purchase/converted", "binary label", "Optional supervised propensity training"),
        ],
        use_case="Online retailer builds top-N bundles and ranks high-probability buyers.",
        output_focus=(
            "Business insights: top recommendation candidates, confidence, and propensity-ranked customers."
        ),
    ),
    "IoT / Manufacturing": Phase2Module(
        label="IoT / Manufacturing Anomaly Detection",
        about=(
            "Detect anomalous sensor patterns on tabular or time-series data with adaptive "
            "feature engineering."
        ),
        business_problem=(
            "Prevents downtime and quality defects by identifying abnormal machine behavior earlier."
        ),
        requirements=[
            ("sensor values", "numeric", "Signals used for anomaly scoring"),
            ("timestamp", "datetime or epoch", "Required for timeline analysis"),
            ("machine_id / line_id", "string or integer", "Optional asset segmentation"),
        ],
        use_case="Factory operations team flags abnormal runs before maintenance incidents escalate.",
        output_focus=(
            "Business insights: anomaly rate, most affected signals, and timeline of critical events."
        ),
    ),
}


def _inject_styles() -> None:
    """Inject lightweight product styling for guided pages."""
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');

            html, body, [class*="css"], [data-testid="stAppViewContainer"] {
                font-family: 'Manrope', sans-serif;
            }

            .guided-hero {
                border-radius: 18px;
                padding: 18px 20px;
                background: linear-gradient(120deg, #12353a 0%, #1f6f78 55%, #4ea37f 100%);
                color: #f3fbf7;
                margin-bottom: 12px;
            }

            .guided-hero h3 {
                margin: 0 0 4px 0;
                font-size: 1.35rem;
                letter-spacing: 0.2px;
            }

            .guided-hero p {
                margin: 0;
                opacity: 0.95;
            }

            .insight-card {
                border-radius: 14px;
                padding: 12px 14px;
                border: 1px solid rgba(36, 61, 64, 0.18);
                background: linear-gradient(180deg, #f9fcfb 0%, #eef8f4 100%);
                min-height: 108px;
            }

            .insight-title {
                font-size: 0.78rem;
                color: #31575d;
                text-transform: uppercase;
                letter-spacing: 0.55px;
                margin-bottom: 5px;
                font-weight: 700;
            }

            .insight-value {
                font-size: 1.2rem;
                color: #102d30;
                font-weight: 800;
                line-height: 1.15;
                margin-bottom: 4px;
            }

            .insight-detail {
                font-size: 0.82rem;
                color: #3d5f66;
                line-height: 1.25;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_insight_cards(cards: list[dict[str, str]]) -> None:
    """Render compact insight cards."""
    if not cards:
        return

    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        col.markdown(
            (
                "<div class='insight-card'>"
                f"<div class='insight-title'>{card.get('title', '')}</div>"
                f"<div class='insight-value'>{card.get('value', '')}</div>"
                f"<div class='insight-detail'>{card.get('detail', '')}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def _auto_detect_target_column(df: pd.DataFrame) -> str:
    """Best-effort target detection with name and distribution heuristics."""
    keyword_hits = [
        "target",
        "label",
        "class",
        "outcome",
        "churn",
        "converted",
        "conversion",
        "purchase",
        "upsell",
        "y",
    ]
    id_pattern = re.compile(r"(^|[_\s\-])id([_\s\-]|$)|_id$|^id_", re.IGNORECASE)

    n_rows = max(1, len(df))
    best_col = str(df.columns[-1])
    best_score = -1e9

    for col in df.columns:
        series = df[col]
        name = str(col).lower()

        non_null = series.dropna()
        unique = max(1, int(non_null.nunique(dropna=True)))
        unique_ratio = unique / max(1, len(non_null))
        missing_ratio = float(series.isna().mean())
        is_numeric = bool(pd.api.types.is_numeric_dtype(series))
        is_id_like = bool(id_pattern.search(name))

        score = 0.0
        if any(token in name for token in keyword_hits):
            score += 4.0

        if unique_ratio <= 0.25:
            score += 2.5
        elif unique_ratio <= 0.6:
            score += 1.0

        if unique_ratio >= 0.95 and n_rows > 80:
            score -= 4.0

        if is_numeric:
            score += 0.4

        if is_id_like:
            score -= 3.0

        score -= 2.0 * missing_ratio

        if score > best_score:
            best_score = score
            best_col = str(col)

    return best_col


def _safe_numeric_target(series: pd.Series) -> pd.Series:
    """Convert target to numeric for correlation analysis."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    encoded, _ = pd.factorize(series.astype(str), sort=True)
    encoded = pd.Series(encoded, index=series.index, dtype=float)
    return encoded.replace(-1, np.nan)


def _compute_top_correlations(df: pd.DataFrame, target_col: str, top_n: int = 10) -> pd.DataFrame:
    """Compute top absolute correlations against target for numeric features."""
    if target_col not in df.columns:
        return pd.DataFrame(columns=["feature", "correlation", "abs_correlation"])

    y = _safe_numeric_target(df[target_col])
    if y.nunique(dropna=True) <= 1:
        return pd.DataFrame(columns=["feature", "correlation", "abs_correlation"])

    rows: list[dict[str, float | str]] = []
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == target_col:
            continue
        merged = pd.concat([pd.to_numeric(df[col], errors="coerce"), y], axis=1).dropna()
        if len(merged) < 8:
            continue
        corr = merged.iloc[:, 0].corr(merged.iloc[:, 1])
        if pd.isna(corr):
            continue
        rows.append(
            {
                "feature": str(col),
                "correlation": float(corr),
                "abs_correlation": float(abs(corr)),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["feature", "correlation", "abs_correlation"])

    corr_df = pd.DataFrame(rows).sort_values("abs_correlation", ascending=False)
    return corr_df.head(top_n)


def _resolve_feature_names(preprocessor: Any, feature_count: int) -> list[str]:
    """Resolve readable post-preprocessing feature names."""
    if preprocessor is not None:
        try:
            names = list(preprocessor.get_feature_names_out())
            if len(names) == feature_count:
                return [str(name) for name in names]
        except Exception:
            pass

    return [f"feature_{i + 1}" for i in range(feature_count)]


def _extract_model_importance(model: Any, feature_names: list[str], top_n: int = 15) -> pd.DataFrame:
    """Extract tree or linear importance for narrative driver analysis."""
    importance: np.ndarray | None = None

    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_).reshape(-1)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        if coef.ndim > 1:
            coef = np.mean(np.abs(coef), axis=0)
        else:
            coef = np.abs(coef)
        importance = coef.reshape(-1)

    if importance is None or importance.size == 0:
        return pd.DataFrame(columns=["feature", "importance"])

    if len(feature_names) != len(importance):
        feature_names = [f"feature_{i + 1}" for i in range(len(importance))]

    imp_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": np.asarray(importance, dtype=float),
        }
    )
    imp_df["importance"] = imp_df["importance"].abs()
    imp_df = imp_df.sort_values("importance", ascending=False)
    return imp_df.head(top_n)


def _derive_binary_probability(model: Any, X_test: Any) -> np.ndarray | None:
    """Get binary probabilities using predict_proba or decision_function fallback."""
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return np.asarray(proba[:, 1], dtype=float)
        except Exception:
            return None

    if hasattr(model, "decision_function"):
        try:
            raw = np.asarray(model.decision_function(X_test), dtype=float)
            if raw.ndim == 1:
                return (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        except Exception:
            return None

    return None


def _infer_primary_metric(task_type: str, metrics: dict[str, float]) -> tuple[str, float]:
    """Pick a display metric for model highlight cards."""
    if task_type == "classification":
        if "test_roc_auc" in metrics:
            return "ROC-AUC", float(metrics["test_roc_auc"])
        return "F1", float(metrics.get("test_f1", 0.0))
    if task_type == "regression":
        return "R2", float(metrics.get("test_r2", 0.0))
    return "Silhouette", float(metrics.get("silhouette", 0.0))


def _compute_profit_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    offer_cost: float = 500.0,
    customer_value: float = 10000.0,
    success_rate: float = 1.0,
) -> tuple[float, float]:
    """Find threshold that maximizes net profit under simple business economics."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.50
    best_profit = -np.inf

    for threshold in thresholds:
        pred = (y_proba >= threshold).astype(int)
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        revenue_saved = tp * customer_value * success_rate
        campaign_cost = (tp + fp) * offer_cost
        profit = revenue_saved - campaign_cost
        if profit > best_profit:
            best_profit = profit
            best_threshold = float(threshold)

    return best_threshold, float(best_profit)


def _auto_phase1_defaults(task_type: str) -> dict[str, Any]:
    """Low-decision defaults for fully automated mode."""
    cls_models, reg_models = get_available_models()

    if task_type == "classification":
        model_names = list(cls_models.keys())
        defaults = [
            name
            for name in ["LGBMClassifier", "XGBClassifier", "RandomForestClassifier", "LogisticRegression"]
            if name in model_names
        ]
        if not defaults:
            defaults = model_names[: min(3, len(model_names))]
        return {
            "selected_metric": "roc_auc",
            "selected_models": defaults,
            "test_size": 0.2,
            "cv_fold_option": "Auto",
            "hyperparameter_tuning": False,
            "tree_max_depth": None,
            "n_estimators": 200,
            "SVM_C": 1.0,
            "KNN_neighbors": 5,
        }

    if task_type == "regression":
        model_names = list(reg_models.keys())
        defaults = [
            name
            for name in ["LGBMRegressor", "XGBRegressor", "RandomForestRegressor", "LinearRegression"]
            if name in model_names
        ]
        if not defaults:
            defaults = model_names[: min(3, len(model_names))]
        return {
            "selected_metric": "r2",
            "selected_models": defaults,
            "test_size": 0.2,
            "cv_fold_option": "Auto",
            "hyperparameter_tuning": False,
            "tree_max_depth": None,
            "n_estimators": 200,
            "SVM_C": 1.0,
            "KNN_neighbors": 5,
        }

    return {
        "selected_metric": "silhouette",
        "selected_models": ["KMeans", "DBSCAN", "AgglomerativeClustering"],
        "n_clusters": 4,
        "hyperparameter_tuning": False,
    }


def _render_phase1_controls(task_type: str, control_mode: str) -> dict[str, Any]:
    """Render phase-1 controls with progressive disclosure."""
    config = _auto_phase1_defaults(task_type)

    if control_mode == "Fully Automated":
        with st.expander("Auto configuration details", expanded=False):
            st.write(config)
        return config

    st.caption("Advanced controls expose model and tuning options while keeping the same pipeline.")

    if task_type == "classification":
        model_names = list(get_available_models()[0].keys())
        config["selected_metric"] = st.selectbox(
            "Optimization metric",
            ["roc_auc", "f1", "balanced_accuracy", "precision", "recall", "accuracy"],
            index=0,
            help="Metric used to rank models and choose the best baseline.",
            key="guided_adv_cls_metric",
        )
        config["selected_models"] = st.multiselect(
            "Models",
            model_names,
            default=[m for m in config["selected_models"] if m in model_names],
            key="guided_adv_cls_models",
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            config["test_size"] = st.slider(
                "Test split",
                0.1,
                0.4,
                float(config["test_size"]),
                step=0.05,
                key="guided_adv_cls_test",
            )
        with col2:
            config["cv_fold_option"] = st.selectbox(
                "CV folds",
                ["Auto", "3", "5", "10"],
                index=0,
                key="guided_adv_cls_cv",
            )
        with col3:
            config["n_estimators"] = st.slider(
                "Tree estimators",
                50,
                500,
                int(config["n_estimators"]),
                step=50,
                key="guided_adv_cls_estimators",
            )
        config["hyperparameter_tuning"] = st.checkbox(
            "Enable hyperparameter tuning",
            value=False,
            key="guided_adv_cls_tuning",
        )
        return config

    if task_type == "regression":
        model_names = list(get_available_models()[1].keys())
        config["selected_metric"] = st.selectbox(
            "Optimization metric",
            ["r2", "mae", "rmse", "mse", "explained_variance"],
            index=0,
            key="guided_adv_reg_metric",
        )
        config["selected_models"] = st.multiselect(
            "Models",
            model_names,
            default=[m for m in config["selected_models"] if m in model_names],
            key="guided_adv_reg_models",
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            config["test_size"] = st.slider(
                "Test split",
                0.1,
                0.4,
                float(config["test_size"]),
                step=0.05,
                key="guided_adv_reg_test",
            )
        with col2:
            config["cv_fold_option"] = st.selectbox(
                "CV folds",
                ["Auto", "3", "5", "10"],
                index=0,
                key="guided_adv_reg_cv",
            )
        with col3:
            config["n_estimators"] = st.slider(
                "Tree estimators",
                50,
                500,
                int(config["n_estimators"]),
                step=50,
                key="guided_adv_reg_estimators",
            )
        config["hyperparameter_tuning"] = st.checkbox(
            "Enable hyperparameter tuning",
            value=False,
            key="guided_adv_reg_tuning",
        )
        return config

    config["selected_metric"] = st.selectbox(
        "Optimization metric",
        ["silhouette", "davies_bouldin", "calinski_harabasz"],
        index=0,
        key="guided_adv_cluster_metric",
    )
    config["selected_models"] = st.multiselect(
        "Clustering models",
        ["KMeans", "DBSCAN", "AgglomerativeClustering", "MeanShift", "SpectralClustering"],
        default=[m for m in config["selected_models"] if m in ["KMeans", "DBSCAN", "AgglomerativeClustering", "MeanShift", "SpectralClustering"]],
        key="guided_adv_cluster_models",
    )
    config["n_clusters"] = st.slider(
        "Default K for KMeans",
        2,
        12,
        int(config["n_clusters"]),
        key="guided_adv_cluster_k",
    )
    config["hyperparameter_tuning"] = st.checkbox(
        "Enable hyperparameter tuning",
        value=False,
        key="guided_adv_cluster_tuning",
    )
    return config


def _run_supervised_phase1(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
    config: dict[str, Any],
    progress_callback,
) -> dict[str, Any]:
    """Run supervised phase-1 pipeline."""
    X_train, X_test, y_train_raw, y_test_raw, preprocessor, column_info = preprocess_and_split(
        df,
        target_col,
        test_size=float(config["test_size"]),
    )

    y_train = y_train_raw
    y_test = y_test_raw
    label_mapping: dict[int, str] | None = None

    if task_type == "classification":
        encoder = LabelEncoder()
        y_train = pd.Series(encoder.fit_transform(y_train_raw), index=y_train_raw.index)
        y_test = pd.Series(encoder.transform(y_test_raw), index=y_test_raw.index)
        label_mapping = {int(i): str(label) for i, label in enumerate(encoder.classes_)}

    performance, best_model_name, use_cv, cv_folds, current_task, trained_models, best_params, all_metrics = train_multiple_models(
        X_train,
        X_test,
        y_train,
        y_test,
        task_type=task_type,
        selected_models_cls=config["selected_models"] if task_type == "classification" else [],
        selected_models_reg=config["selected_models"] if task_type == "regression" else [],
        selected_metric_cls=config["selected_metric"] if task_type == "classification" else "accuracy",
        selected_metric_reg=config["selected_metric"] if task_type == "regression" else "r2",
        hyperparameter_tuning=bool(config["hyperparameter_tuning"]),
        cv_fold_option=str(config["cv_fold_option"]),
        progress_callback=progress_callback,
        tree_max_depth=config.get("tree_max_depth"),
        n_estimators=int(config.get("n_estimators", 200)),
        SVM_C=float(config.get("SVM_C", 1.0)),
        KNN_neighbors=int(config.get("KNN_neighbors", 5)),
    )

    feature_names = _resolve_feature_names(preprocessor, int(X_train.shape[1]))

    return {
        "task_type": current_task,
        "target_col": target_col,
        "config": config,
        "performance": performance,
        "best_model_name": best_model_name,
        "use_cv": use_cv,
        "cv_folds": cv_folds,
        "trained_models": trained_models,
        "best_params": best_params,
        "all_metrics": all_metrics,
        "X_test": X_test,
        "y_test": np.asarray(y_test),
        "label_mapping": label_mapping,
        "feature_names": feature_names,
        "column_info": column_info,
    }


def _run_clustering_phase1(
    df: pd.DataFrame,
    target_col: str,
    config: dict[str, Any],
    progress_callback,
) -> dict[str, Any]:
    """Run clustering phase-1 pipeline."""
    X_all = df.drop(columns=[target_col], errors="ignore").copy()
    numeric_features = X_all.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_all.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    transformers = []
    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features))

    if not transformers:
        raise ValueError("No usable features found for clustering.")

    preprocessor = ColumnTransformer(transformers=transformers)
    X_proc = preprocessor.fit_transform(X_all)

    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()

    performance, best_model_name, trained_models, best_params, all_metrics, cluster_labels = train_clustering_models(
        X_data=X_proc,
        selected_models_clustering=config["selected_models"],
        selected_metric_clustering=config["selected_metric"],
        hyperparameter_tuning=bool(config["hyperparameter_tuning"]),
        n_clusters=int(config.get("n_clusters", 4)),
        progress_callback=progress_callback,
    )

    return {
        "task_type": "clustering",
        "target_col": target_col,
        "config": config,
        "performance": performance,
        "best_model_name": best_model_name,
        "trained_models": trained_models,
        "best_params": best_params,
        "all_metrics": all_metrics,
        "cluster_labels": cluster_labels,
        "X_all": X_proc,
        "feature_names": _resolve_feature_names(preprocessor, int(X_proc.shape[1])),
    }


def _render_overview_step(df: pd.DataFrame, target_col: str, task_type: str) -> None:
    """Step 1: overview narrative."""
    st.markdown("### Step 1: Overview")

    missing_rate = float(df.isna().sum().sum() / max(1, df.shape[0] * df.shape[1]))
    cards = [
        {
            "title": "Total records",
            "value": f"{len(df):,}",
            "detail": "Entities available for modeling.",
        },
        {
            "title": "Columns",
            "value": f"{df.shape[1]}",
            "detail": "Detected feature space before preprocessing.",
        },
        {
            "title": "Missing data",
            "value": f"{missing_rate:.2%}",
            "detail": "Used to estimate preprocessing complexity.",
        },
    ]

    if task_type == "classification":
        y = df[target_col]
        positive_rate = float((y.astype(str).str.lower().isin(["1", "yes", "true", "churn", "positive"]).mean()))
        cards.append(
            {
                "title": "Positive-class hint",
                "value": f"{positive_rate:.2%}",
                "detail": "Quick signal for class imbalance risk.",
            }
        )
    else:
        cards.append(
            {
                "title": "Target",
                "value": str(target_col),
                "detail": "Auto-selected target for this run.",
            }
        )

    _render_insight_cards(cards)

    viz_task = "classification" if task_type == "classification" else "regression"
    plot_target_distribution(df[target_col], viz_task)


def _render_drivers_step(df: pd.DataFrame, target_col: str, results: dict[str, Any]) -> None:
    """Step 2: key drivers and insight narrative."""
    st.markdown("### Step 2: Key Drivers and Insights")

    corr_df = _compute_top_correlations(df, target_col)
    if not corr_df.empty:
        top = corr_df.iloc[0]
        _render_insight_cards(
            [
                {
                    "title": "Top driver (correlation)",
                    "value": str(top["feature"]),
                    "detail": f"Correlation {float(top['correlation']):+.3f} to target.",
                },
                {
                    "title": "Strongest absolute signal",
                    "value": f"{float(top['abs_correlation']):.3f}",
                    "detail": "Higher absolute values indicate stronger linear influence.",
                },
            ]
        )
        plot_feature_correlation_lollipop(corr_df, title="Top Correlation Drivers")
    else:
        st.info("Not enough numeric signal to compute reliable target correlations.")

    best_model = results["trained_models"].get(results["best_model_name"])
    importance_df = _extract_model_importance(best_model, results.get("feature_names", []), top_n=15)

    if not importance_df.empty:
        top_feature = str(importance_df.iloc[0]["feature"])
        top_importance = float(importance_df.iloc[0]["importance"])
        _render_insight_cards(
            [
                {
                    "title": "Model-specific top feature",
                    "value": top_feature,
                    "detail": f"Importance score {top_importance:.4f}.",
                }
            ]
        )
        fig = px.scatter(
            importance_df.sort_values("importance", ascending=True),
            x="importance",
            y="feature",
            size="importance",
            size_max=22,
            title="Model Importance View",
            color="importance",
            color_continuous_scale="Tealgrn",
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)


def _render_performance_step(results: dict[str, Any]) -> dict[str, Any]:
    """Step 3: model performance narrative and visuals."""
    st.markdown("### Step 3: Model Performance")

    task_type = str(results["task_type"])
    best_model_name = str(results["best_model_name"])

    if task_type == "clustering":
        best_metrics = results["all_metrics"].get(best_model_name, {})
    else:
        best_metrics = results["all_metrics"].get(best_model_name, {})

    metric_name, metric_value = _infer_primary_metric(task_type, best_metrics)
    _render_insight_cards(
        [
            {
                "title": "Best model",
                "value": best_model_name,
                "detail": f"Selected by {results['config']['selected_metric']}.",
            },
            {
                "title": f"Best {metric_name}",
                "value": f"{metric_value:.4f}",
                "detail": "Primary quality signal for this task.",
            },
        ]
    )

    if task_type == "clustering":
        perf_df = create_clustering_performance_dataframe(
            results["performance"],
            results["all_metrics"],
            results["best_params"],
        )
        st.dataframe(perf_df, use_container_width=True)

        labels = results["cluster_labels"].get(best_model_name)
        if labels is not None:
            col1, col2 = st.columns(2)
            with col1:
                plot_cluster_scatter_2d(results["X_all"], labels, best_model_name)
            with col2:
                try:
                    plot_silhouette_analysis(results["X_all"], labels, best_model_name)
                except Exception:
                    st.info("Silhouette analysis is unavailable for this clustering output.")
            plot_elbow_curve(results["X_all"], max_clusters=10)

        return {}

    perf_df = create_performance_dataframe(results["performance"], results["best_params"], results["all_metrics"])
    st.dataframe(perf_df, use_container_width=True)
    plot_model_score_comparison(results["performance"], title="Model Leaderboard (Test vs CV)")

    model = results["trained_models"][best_model_name]
    y_test = np.asarray(results["y_test"])
    y_pred = np.asarray(model.predict(results["X_test"]))

    if task_type == "classification":
        plot_confusion_matrix_generic(y_test, y_pred, title=f"Confusion Matrix - {best_model_name}")

        binary = len(np.unique(y_test)) == 2
        y_proba = _derive_binary_probability(model, results["X_test"]) if binary else None
        if binary and y_proba is not None:
            col1, col2 = st.columns(2)
            with col1:
                plot_roc_curve_binary(y_test, y_proba, best_model_name)
            with col2:
                plot_precision_recall_curve_binary(y_test, y_proba, best_model_name)
            plot_probability_distribution(y_proba, title="Predicted Positive-Class Probability")
            plot_threshold_metric_curves(y_test, y_proba, best_model_name)
        else:
            st.info("ROC, PR, and threshold curves require binary probabilities.")

        return {
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
            "best_model_name": best_model_name,
        }

    col1, col2 = st.columns(2)
    with col1:
        plot_regression_predictions(y_test, y_pred, best_model_name)
    with col2:
        plot_residuals_analysis(y_test, y_pred, best_model_name)

    return {
        "y_test": y_test,
        "y_pred": y_pred,
        "best_model_name": best_model_name,
    }


def _render_decision_step(results: dict[str, Any], model_artifacts: dict[str, Any]) -> None:
    """Step 4: business decision layer and impact simulation."""
    st.markdown("### Step 4: Decision Layer")

    task_type = str(results["task_type"])

    if task_type == "classification":
        y_test = model_artifacts.get("y_test")
        y_proba = model_artifacts.get("y_proba")
        best_model_name = model_artifacts.get("best_model_name", "Best model")

        if y_test is None or y_proba is None:
            st.info("Decision optimization is available when binary probability scores are available.")
            return

        optimal_thr, optimal_profit = _compute_profit_optimal_threshold(y_test, y_proba)
        _render_insight_cards(
            [
                {
                    "title": "Optimal threshold",
                    "value": f"{optimal_thr:.2f}",
                    "detail": "Default profit assumptions maximize net benefit at this cutoff.",
                },
                {
                    "title": "Estimated net profit",
                    "value": f"${optimal_profit:,.0f}",
                    "detail": "Before applying your custom economics below.",
                },
            ]
        )

        plot_threshold_analysis(np.asarray(y_test), np.asarray(y_proba), str(best_model_name))
        return

    if task_type == "regression":
        y_test = np.asarray(model_artifacts.get("y_test", []), dtype=float)
        y_pred = np.asarray(model_artifacts.get("y_pred", []), dtype=float)
        if y_test.size == 0 or y_pred.size == 0:
            st.info("No regression predictions available for decision simulation.")
            return

        mae = float(mean_absolute_error(y_test, y_pred))
        error = np.abs(y_test - y_pred)

        c1, c2 = st.columns(2)
        with c1:
            cost_per_error_unit = st.number_input(
                "Cost per unit prediction error",
                min_value=0.0,
                value=120.0,
                step=10.0,
                key="guided_reg_cost_per_error",
            )
        with c2:
            decision_volume = st.number_input(
                "Decisions per cycle",
                min_value=1,
                value=1000,
                step=100,
                key="guided_reg_decision_volume",
            )

        expected_cost = float(mae * cost_per_error_unit * decision_volume)
        _render_insight_cards(
            [
                {
                    "title": "Mean absolute error",
                    "value": f"{mae:.3f}",
                    "detail": "Average miss size in target units.",
                },
                {
                    "title": "Expected impact cost",
                    "value": f"${expected_cost:,.0f}",
                    "detail": "Estimated operational impact from prediction error.",
                },
            ]
        )

        fig = px.histogram(
            x=error,
            nbins=35,
            title="Absolute Error Distribution",
            labels={"x": "|Actual - Predicted|", "y": "Count"},
            color_discrete_sequence=["#d95f02"],
        )
        st.plotly_chart(fig, use_container_width=True)
        return

    labels = results.get("cluster_labels", {}).get(results.get("best_model_name"))
    if labels is None:
        st.info("No clustering labels available for decision summary.")
        return

    counts = pd.Series(labels).value_counts().sort_values(ascending=False)
    largest_share = float(counts.iloc[0] / max(1, counts.sum()))
    _render_insight_cards(
        [
            {
                "title": "Largest segment share",
                "value": f"{largest_share:.2%}",
                "detail": "Indicates concentration risk or dominant customer profile.",
            },
            {
                "title": "Detected segments",
                "value": str(len(counts)),
                "detail": "Use these groups for differentiated campaign or operations policy.",
            },
        ]
    )

    decision_df = counts.rename_axis("segment").reset_index(name="records")
    decision_df["share_pct"] = (decision_df["records"] / max(1, decision_df["records"].sum()) * 100).round(2)
    decision_df["recommended_action"] = decision_df["share_pct"].apply(
        lambda pct: "Create dedicated playbook" if pct >= 30 else "Include in targeted cohort"
    )
    st.dataframe(decision_df, use_container_width=True, hide_index=True)


def _render_phase1_story(df: pd.DataFrame, target_col: str, results: dict[str, Any]) -> None:
    """Render full phase-1 story in four connected steps."""
    st.markdown("---")
    st.markdown("## Phase 1 Storyboard")
    st.progress(0.25)
    _render_overview_step(df, target_col, str(results["task_type"]))

    st.progress(0.50)
    if str(results["task_type"]) in {"classification", "regression"}:
        _render_drivers_step(df, target_col, results)
    else:
        st.markdown("### Step 2: Key Drivers and Insights")
        st.info("Driver-level analysis is limited for unsupervised clustering without a target signal.")

    st.progress(0.75)
    model_artifacts = _render_performance_step(results)

    st.progress(1.0)
    _render_decision_step(results, model_artifacts)


def _render_phase1(df: pd.DataFrame, target_col: str, task_type: str) -> None:
    """Render phase 1 controls and execute the baseline pipeline."""
    st.markdown("### Phase 1: General AI Auto-Modelling")
    st.caption(
        "Fast baseline mode with automatic preprocessing, feature engineering, model training, and evaluation."
    )

    control_mode = st.radio(
        "Control mode",
        ["Fully Automated", "Advanced Controls"],
        horizontal=True,
        key="guided_phase1_control_mode",
        help=(
            "Fully Automated keeps decisions minimal. Advanced Controls exposes model, metric, and tuning choices."
        ),
    )

    config = _render_phase1_controls(task_type, control_mode)
    run_label = "Run Automated Pipeline" if control_mode == "Fully Automated" else "Run Pipeline"

    run_button = st.button(run_label, type="primary", use_container_width=True, key="guided_phase1_run_button")

    if "guided_phase1_story_results" not in st.session_state:
        st.session_state.guided_phase1_story_results = None

    if run_button:
        if not config.get("selected_models"):
            st.error("Please select at least one model before running the pipeline.")
            return

        working_df = df.dropna(subset=[target_col]).copy()
        if len(working_df) < 20:
            st.error("At least 20 valid rows are required after removing missing target values.")
            return

        progress_text = st.empty()
        progress_bar = st.progress(0)

        def _progress(current: int, total: int, model_name: str | None) -> None:
            if total <= 0:
                return
            progress_bar.progress(min(1.0, float(current) / float(total)))
            if model_name:
                progress_text.text(f"Training {model_name} ({current}/{total})")
            else:
                progress_text.text("Training complete")

        try:
            with st.spinner("Running end-to-end AutoML pipeline..."):
                if task_type == "clustering":
                    results = _run_clustering_phase1(working_df, target_col, config, _progress)
                else:
                    results = _run_supervised_phase1(working_df, target_col, task_type, config, _progress)

            st.session_state.guided_phase1_story_results = results
            st.success(f"Phase 1 complete. Best model: {results['best_model_name']}")
        except Exception as exc:
            st.error(f"Phase 1 failed: {exc}")
        finally:
            progress_text.empty()
            progress_bar.empty()

    results = st.session_state.guided_phase1_story_results
    if not results:
        return

    if results.get("target_col") != target_col or results.get("task_type") != task_type:
        st.info("Inputs changed since the last run. Execute Phase 1 again to refresh insights.")
        return

    _render_phase1_story(df, target_col, results)


def _render_module_frame(module: Phase2Module) -> None:
    """Render About, Requirements, Use Case, and Output Focus sections."""
    st.markdown("### About this module")
    st.write(module.about)
    st.info(module.business_problem)

    st.markdown("### Dataset Requirements")
    req_df = pd.DataFrame(module.requirements, columns=["Required column", "Expected type", "Why it matters"])
    st.dataframe(req_df, use_container_width=True, hide_index=True)

    st.markdown("### Example Use Case")
    st.write(module.use_case)

    st.markdown("### Output Focus")
    st.write(module.output_focus)


def _render_phase2(df: pd.DataFrame) -> None:
    """Render phase 2 business applications."""
    st.markdown("### Phase 2: Industry-Specific Analysis")
    st.caption("Use Phase 1 baseline context, then switch into business workflows.")

    module_key = st.selectbox(
        "Industry module",
        list(PHASE2_MODULES.keys()),
        key="guided_phase2_module_selector",
    )
    module = PHASE2_MODULES[module_key]
    _render_module_frame(module)
    st.markdown("---")

    if module_key == "Customer Analytics":
        workflow = st.radio(
            "Customer workflow",
            ["Upsell Probability", "Customer Segmentation", "Customer Churn"],
            horizontal=True,
            key="guided_phase2_customer_workflow",
        )

        if workflow == "Upsell Probability":
            render_upsell_probability_tab(df)
        elif workflow == "Customer Segmentation":
            render_customer_segmentation_tab(df)
        else:
            st.warning(
                "Customer Churn Studio is available in Advanced Workspace mode where full churn controls, "
                "drift checks, and model export are enabled."
            )
    elif module_key == "E-commerce":
        render_ecommerce_recommendation_tab(df)
    else:
        render_iot_anomaly_tab(df)


def render_guided_journey(
    df: pd.DataFrame,
    target_col: str | None = None,
    auto_detected_task: str | None = None,
) -> None:
    """Public entry point for the two-phase guided product flow."""
    _inject_styles()

    st.markdown(
        """
        <div class='guided-hero'>
            <h3>OptiMLFlow AI Product Journey</h3>
            <p>Phase 1 builds a strong AutoML baseline. Phase 2 translates that baseline into industry decisions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if df is None or df.empty:
        st.warning("Dataset is empty. Upload a valid dataset to begin.")
        return

    detected_target = target_col if target_col in df.columns else _auto_detect_target_column(df)
    target_idx = list(df.columns).index(detected_target)

    selected_target = st.selectbox(
        "Target column",
        list(df.columns),
        index=target_idx,
        key="guided_detected_target",
        help="Auto-detected target. You can override it before running the pipeline.",
    )

    resolved_task = auto_detected_task if auto_detected_task and selected_target == target_col else detect_task_type(df[selected_target])
    col_info = identify_column_types(df.drop(columns=[selected_target], errors="ignore"))

    _render_insight_cards(
        [
            {
                "title": "Detected target",
                "value": selected_target,
                "detail": "You can override this selection any time.",
            },
            {
                "title": "Detected problem type",
                "value": str(resolved_task).upper(),
                "detail": "Classification, regression, or clustering pipeline will be applied.",
            },
            {
                "title": "Data profile",
                "value": f"{len(col_info['num_cols'])} numeric / {len(col_info['cat_cols'])} categorical",
                "detail": "Used for automated preprocessing and feature handling.",
            },
        ]
    )

    phase = st.radio(
        "Product phase",
        ["Phase 1: General Auto-Modelling", "Phase 2: Industry-Specific Analysis"],
        horizontal=True,
        key="guided_phase_selector",
    )

    if phase.startswith("Phase 1"):
        _render_phase1(df, selected_target, str(resolved_task))
    else:
        if st.session_state.get("guided_phase1_story_results") is None:
            st.info("Run Phase 1 first to establish a baseline before entering business modules.")
        _render_phase2(df)
