"""
Domain-specific workflow components for OptiMLFlow.

This module adds business-first Streamlit workflows for:
- Customer Analytics (upsell probability and segmentation)
- E-commerce recommendation systems
- Manufacturing / IoT anomaly detection
"""

from __future__ import annotations

import itertools
import re
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core.churn_analysis import (
    apply_imbalance_handling,
    build_churn_metrics_dataframe,
    encode_churn_target,
    get_churn_models,
    plot_churn_metrics_comparison,
    plot_confusion_matrix_churn,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_threshold_analysis,
    preprocess_for_churn,
    train_churn_models,
)
from core.entity_aggregator import EntityAggregator
from core.model_training import (
    create_clustering_performance_dataframe,
    create_performance_dataframe,
    get_available_models,
    train_clustering_models,
    train_multiple_models,
)
from core.preprocessing import get_preprocessing_summary, identify_column_types, preprocess_and_split


def _default_column_by_keywords(columns: list[str], keywords: list[str]) -> str | None:
    """Return the first column that matches any keyword."""
    for column in columns:
        name = str(column).lower()
        if any(keyword in name for keyword in keywords):
            return column
    return None


def _id_like_columns(df: pd.DataFrame) -> list[str]:
    """Best-effort detection of ID-like columns for ranking outputs."""
    id_pattern = re.compile(
        r"(^|[_\s\-])id([_\s\-]|$)|_id$|^id_|customer|account|member|user|client",
        re.IGNORECASE,
    )
    return [col for col in df.columns if id_pattern.search(str(col))]


def _try_binary_encode_target(series: pd.Series) -> pd.Series:
    """Encode common binary targets with robust handling."""
    encoded = encode_churn_target(series)
    unique_values = set(pd.Series(encoded).dropna().unique().tolist())
    if not unique_values.issubset({0, 1}):
        raise ValueError(
            "Target column could not be encoded to a binary label (0/1). "
            "Please provide a binary target column."
        )
    return encoded


def render_solution_hub() -> str:
    """Render a lightweight solution hub selector."""
    st.subheader("Solution Hub")
    st.markdown(
        "Choose the business workflow you want to prioritize. "
        "All workflows remain available below in dedicated tabs."
    )

    options = [
        "Customer Analytics (SaaS / Telecom / Banking)",
        "E-commerce Recommendation Systems",
        "Manufacturing / IoT Anomaly Detection",
        "General AutoML",
    ]

    selected = st.radio(
        "Workflow focus",
        options,
        horizontal=True,
        key="solution_hub_focus",
    )

    descriptions = {
        options[0]: "Predict churn, estimate upsell probability, and discover customer segments.",
        options[1]: "Handle noisy product and behavior data, then generate recommendations quickly.",
        options[2]: "Preprocess timestamped sensor data and detect anomalous events.",
        options[3]: "Use the original general-purpose model training flow.",
    }
    st.info(descriptions[selected])
    return selected


def render_dataset_health_panel(df: pd.DataFrame) -> None:
    """Render shared data quality diagnostics for messy business datasets."""
    st.markdown("#### Dataset Health")

    if df.empty:
        st.warning("Dataset is empty.")
        return

    column_info = identify_column_types(df)

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isna().sum().sum())
    missing_pct = (missing_cells / total_cells * 100) if total_cells else 0.0

    duplicate_rows = int(df.duplicated().sum())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", f"{df.shape[1]}")
    col3.metric("Missing", f"{missing_pct:.2f}%")
    col4.metric("Duplicates", f"{duplicate_rows:,}")

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("ID-like columns", len(column_info["id_cols"]))
    summary_col2.metric("Low-card categories", len(column_info["low_card_cols"]))
    summary_col3.metric("High-card categories", len(column_info["high_card_cols"]))

    with st.expander("Show quality diagnostics", expanded=False):
        missing_by_col = (
            (df.isna().mean() * 100)
            .sort_values(ascending=False)
            .reset_index()
        )
        missing_by_col.columns = ["column", "missing_pct"]
        st.markdown("Top columns by missingness")
        st.dataframe(missing_by_col.head(10), use_container_width=True)

        cardinality = pd.DataFrame(
            {
                "column": df.columns,
                "dtype": [str(df[c].dtype) for c in df.columns],
                "nunique": [int(df[c].nunique(dropna=True)) for c in df.columns],
            }
        ).sort_values("nunique", ascending=False)
        st.markdown("Column cardinality")
        st.dataframe(cardinality.head(15), use_container_width=True)


def render_upsell_probability_tab(df: pd.DataFrame) -> None:
    """Render end-to-end upsell probability workflow."""
    st.markdown("### Upsell Probability")
    st.markdown(
        "Train probabilistic models that score which customers are most likely "
        "to accept an upsell or upgrade offer."
    )

    all_columns = list(df.columns)
    default_target = _default_column_by_keywords(
        all_columns,
        ["upsell", "upgrade", "conversion", "converted", "accepted_offer", "purchase_next", "target"],
    )
    default_idx = all_columns.index(default_target) if default_target in all_columns else 0

    target_col = st.selectbox(
        "Upsell target column (binary)",
        all_columns,
        index=default_idx,
        key="upsell_target_col",
    )

    try:
        encoded_preview = _try_binary_encode_target(df[target_col])
        class_counts = encoded_preview.value_counts().sort_index()
        positives = int(class_counts.get(1, 0))
        negatives = int(class_counts.get(0, 0))
        ratio = positives / max(1, positives + negatives)
        st.caption(
            f"Target preview: negatives={negatives:,}, positives={positives:,}, "
            f"positive_rate={ratio:.2%}"
        )
    except Exception as exc:
        st.error(str(exc))
        return

    model_options = list(get_churn_models().keys())
    default_models = [m for m in ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"] if m in model_options]
    if not default_models:
        default_models = model_options[: min(3, len(model_options))]

    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test split", 0.1, 0.4, 0.2, step=0.05, key="upsell_test_size")
        imbalance_method = st.selectbox(
            "Imbalance handling",
            ["none", "class_weight", "smote", "adasyn", "undersample"],
            index=2,
            key="upsell_imbalance",
        )
    with col2:
        optimize_metric = st.selectbox(
            "Optimize metric",
            ["roc_auc", "avg_precision", "f1", "recall", "precision", "balanced_accuracy", "accuracy"],
            index=0,
            key="upsell_metric",
        )
        cv_folds = st.selectbox("CV folds", [3, 5, 10], index=1, key="upsell_cv_folds")
    with col3:
        n_estimators = st.slider("Tree estimators", 50, 500, 200, step=50, key="upsell_estimators")
        max_depth_option = st.selectbox(
            "Max tree depth",
            ["None"] + [str(i) for i in range(2, 21)],
            index=0,
            key="upsell_max_depth",
        )
        max_depth = None if max_depth_option == "None" else int(max_depth_option)

    selected_models = st.multiselect(
        "Models",
        model_options,
        default=default_models,
        key="upsell_models",
    )

    train_button = st.button(
        "Train upsell models",
        type="primary",
        use_container_width=True,
        key="upsell_train_btn",
    )

    if "upsell_results" not in st.session_state:
        st.session_state.upsell_results = None

    if train_button:
        if not selected_models:
            st.error("Please select at least one model.")
            return

        working_df = df.copy()
        try:
            working_df[target_col] = _try_binary_encode_target(working_df[target_col])
        except Exception as exc:
            st.error(str(exc))
            return

        with st.spinner("Preprocessing customer data..."):
            try:
                X_train, X_test, y_train, y_test, feature_names = preprocess_for_churn(
                    working_df,
                    target_col,
                    test_size=test_size,
                )
            except Exception as exc:
                st.error(f"Preprocessing failed: {exc}")
                return

        class_weight = None
        with st.spinner("Applying imbalance handling..."):
            X_train_bal, y_train_bal, class_weight = apply_imbalance_handling(
                X_train,
                y_train,
                imbalance_method,
            )

        progress_text = st.empty()
        progress = st.progress(0)

        def _progress(current: int, total: int, model_name: str | None) -> None:
            if total <= 0:
                return
            progress.progress(min(1.0, current / total))
            if model_name:
                progress_text.text(f"Training {model_name} ({current}/{total})")
            else:
                progress_text.text("Training complete")

        try:
            metrics_dict, proba_dict, pred_dict, trained_models, best_model = train_churn_models(
                X_train_bal,
                X_test,
                y_train_bal,
                y_test,
                selected_model_names=selected_models,
                class_weight=class_weight,
                n_estimators=n_estimators,
                max_depth=max_depth,
                svm_c=1.0,
                cv_folds=int(cv_folds),
                optimize_metric=optimize_metric,
                progress_callback=_progress,
            )
        except Exception as exc:
            progress.empty()
            progress_text.empty()
            st.error(f"Training failed: {exc}")
            return

        progress.empty()
        progress_text.empty()

        st.session_state.upsell_results = {
            "target_col": target_col,
            "metrics_dict": metrics_dict,
            "proba_dict": proba_dict,
            "pred_dict": pred_dict,
            "trained_models": trained_models,
            "best_model": best_model,
            "y_test": y_test,
            "test_indices": list(y_test.index),
            "optimize_metric": optimize_metric,
        }

    if st.session_state.upsell_results is None:
        return

    results = st.session_state.upsell_results
    best_model = results["best_model"]
    metrics_dict = results["metrics_dict"]
    proba_dict = results["proba_dict"]
    pred_dict = results["pred_dict"]
    y_test = results["y_test"]

    st.markdown("---")
    st.success(
        f"Best model: {best_model} | "
        f"ROC-AUC={metrics_dict[best_model].get('roc_auc', np.nan):.4f}"
    )

    metrics_df = build_churn_metrics_dataframe(metrics_dict)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    res_tab1, res_tab2, res_tab3, res_tab4 = st.tabs(
        [
            "Metric Comparison",
            "ROC / PR Curves",
            "Confusion Matrix",
            "Upsell Priority List",
        ]
    )

    with res_tab1:
        plot_churn_metrics_comparison(metrics_dict)

    with res_tab2:
        plot_roc_curves(y_test, proba_dict)
        plot_precision_recall_curves(y_test, proba_dict)

    with res_tab3:
        selected_model_cm = st.selectbox(
            "Model",
            list(pred_dict.keys()),
            key="upsell_cm_model",
        )
        plot_confusion_matrix_churn(y_test, pred_dict[selected_model_cm], selected_model_cm)

    with res_tab4:
        chosen_model = st.selectbox(
            "Probability model",
            [m for m, prob in proba_dict.items() if prob is not None],
            key="upsell_priority_model",
        )
        if chosen_model:
            probs = np.asarray(proba_dict[chosen_model])
            ranking_df = df.loc[results["test_indices"]].copy()
            ranking_df["upsell_probability"] = probs
            ranking_df["predicted_upsell"] = (ranking_df["upsell_probability"] >= 0.5).astype(int)
            ranking_df = ranking_df.sort_values("upsell_probability", ascending=False)

            st.markdown("Top customers by upsell probability")
            max_rows = max(1, min(500, len(ranking_df)))
            default_rows = min(50, max_rows)
            top_n = st.slider("Top N", 1, max_rows, default_rows, key="upsell_top_n")

            display_cols = _id_like_columns(ranking_df)[:2]
            for candidate in [results["target_col"], "upsell_probability", "predicted_upsell"]:
                if candidate in ranking_df.columns and candidate not in display_cols:
                    display_cols.append(candidate)
            if not display_cols:
                display_cols = ["upsell_probability", "predicted_upsell"]

            st.dataframe(ranking_df[display_cols].head(top_n), use_container_width=True)

            fig = px.histogram(
                ranking_df,
                x="upsell_probability",
                nbins=30,
                title="Upsell Probability Distribution",
            )
            st.plotly_chart(fig, use_container_width=True)

            plot_threshold_analysis(y_test, probs, chosen_model)


def _prepare_entity_clustering_data(
    df: pd.DataFrame,
    exclude_target_col: str | None,
    prefer_entity_col: str | None,
) -> tuple[pd.DataFrame, str | None, bool, pd.DataFrame, Any]:
    """Aggregate entity-level rows and create clustering matrix."""
    working_df = df.copy()
    if exclude_target_col and exclude_target_col in working_df.columns:
        working_df = working_df.drop(columns=[exclude_target_col])

    aggregator = EntityAggregator(
        target_col=None,
        prefer_col=prefer_entity_col if prefer_entity_col and prefer_entity_col != "(auto)" else None,
        verbose=False,
    )
    entity_df, entity_col, aggregated = aggregator.fit_transform(working_df)

    feature_df = entity_df.copy()
    if entity_col and entity_col in feature_df.columns:
        feature_df = feature_df.drop(columns=[entity_col])

    if feature_df.empty:
        raise ValueError("No features available for clustering after entity processing.")

    numeric_features = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = feature_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    transformers = []
    if numeric_features:
        transformers.append(("num", StandardScaler(), numeric_features))
    if categorical_features:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features))

    if not transformers:
        raise ValueError("No supported numeric/categorical columns found for clustering.")

    preprocessor = ColumnTransformer(transformers=transformers)
    X_proc = preprocessor.fit_transform(feature_df)

    return entity_df, entity_col, aggregated, feature_df, X_proc


def _build_segment_profile(df_with_segments: pd.DataFrame) -> pd.DataFrame:
    """Build profile table with segment sizes and numeric means."""
    segment_counts = df_with_segments["segment"].value_counts(dropna=False).sort_index()
    total = max(1, len(df_with_segments))

    profile = pd.DataFrame(
        {
            "segment": segment_counts.index,
            "segment_size": segment_counts.values,
            "segment_share_pct": (segment_counts.values / total * 100).round(2),
        }
    )

    numeric_cols = df_with_segments.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "segment"]

    if numeric_cols:
        means = (
            df_with_segments.groupby("segment")[numeric_cols]
            .mean(numeric_only=True)
            .round(3)
            .reset_index()
        )
        profile = profile.merge(means, on="segment", how="left")

    return profile.sort_values("segment_size", ascending=False)


def _project_clusters(X_proc: Any, labels: np.ndarray) -> pd.DataFrame:
    """Project cluster features into 2D for quick visual diagnostics."""
    X_dense = X_proc.toarray() if hasattr(X_proc, "toarray") else np.asarray(X_proc)

    if X_dense.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=42)
        projected = pca.fit_transform(X_dense)
        variance_ratio = pca.explained_variance_ratio_.sum()
    else:
        projected = np.hstack([X_dense, np.zeros((X_dense.shape[0], 1))])
        variance_ratio = 1.0

    projection_df = pd.DataFrame(
        {
            "component_1": projected[:, 0],
            "component_2": projected[:, 1],
            "segment": labels.astype(str),
        }
    )
    projection_df.attrs["variance_ratio"] = variance_ratio
    return projection_df


def render_customer_segmentation_tab(df: pd.DataFrame) -> None:
    """Render customer segmentation workflow."""
    st.markdown("### Customer Segmentation")
    st.markdown(
        "Automatically derive customer-level features from messy records and "
        "group similar customers into actionable segments."
    )

    entity_default = _default_column_by_keywords(list(df.columns), ["customer_id", "user_id", "client_id", "account_id", "customer"])

    col1, col2, col3 = st.columns(3)
    with col1:
        exclude_target = st.selectbox(
            "Optional target column to exclude",
            ["(none)"] + list(df.columns),
            index=0,
            key="segment_exclude_target",
        )
    with col2:
        entity_override_values = ["(auto)"] + list(df.columns)
        default_entity_idx = entity_override_values.index(entity_default) if entity_default in entity_override_values else 0
        entity_override = st.selectbox(
            "Entity ID column",
            entity_override_values,
            index=default_entity_idx,
            key="segment_entity_override",
        )
    with col3:
        n_clusters = st.slider("Default K for KMeans", 2, 12, 4, key="segment_n_clusters")

    selected_models = st.multiselect(
        "Clustering models",
        ["KMeans", "DBSCAN", "AgglomerativeClustering", "MeanShift", "SpectralClustering"],
        default=["KMeans", "DBSCAN"],
        key="segment_models",
    )

    metric = st.selectbox(
        "Optimization metric",
        ["silhouette", "davies_bouldin", "calinski_harabasz"],
        index=0,
        key="segment_metric",
    )
    tune_hyperparams = st.checkbox("Enable clustering hyperparameter tuning", key="segment_tuning")

    run_btn = st.button("Build customer segments", type="primary", use_container_width=True, key="segment_run_btn")

    if "segmentation_results" not in st.session_state:
        st.session_state.segmentation_results = None

    if run_btn:
        if not selected_models:
            st.error("Please select at least one clustering model.")
            return

        with st.spinner("Preparing customer-level feature matrix..."):
            try:
                entity_df, entity_col, aggregated, feature_df, X_proc = _prepare_entity_clustering_data(
                    df,
                    None if exclude_target == "(none)" else exclude_target,
                    entity_override,
                )
            except Exception as exc:
                st.error(f"Feature preparation failed: {exc}")
                return

        progress_text = st.empty()
        progress = st.progress(0)

        def _progress(current: int, total: int, model_name: str | None) -> None:
            if total <= 0:
                return
            progress.progress(min(1.0, current / total))
            if model_name:
                progress_text.text(f"Training {model_name} ({current}/{total})")
            else:
                progress_text.text("Clustering complete")

        performance, best_model, trained_models, best_params, all_metrics, labels_dict = train_clustering_models(
            X_data=X_proc,
            selected_models_clustering=selected_models,
            selected_metric_clustering=metric,
            hyperparameter_tuning=tune_hyperparams,
            n_clusters=n_clusters,
            progress_callback=_progress,
        )

        progress.empty()
        progress_text.empty()

        if not best_model:
            st.error("No clustering model completed successfully.")
            return

        labels = labels_dict.get(best_model)
        if labels is None:
            st.error("Best model did not produce cluster labels.")
            return

        segmented_df = entity_df.copy()
        segmented_df["segment"] = labels
        profile_df = _build_segment_profile(segmented_df)
        projection_df = _project_clusters(X_proc, labels)

        st.session_state.segmentation_results = {
            "entity_col": entity_col,
            "aggregated": aggregated,
            "feature_count": feature_df.shape[1],
            "performance": performance,
            "best_model": best_model,
            "best_params": best_params,
            "all_metrics": all_metrics,
            "segmented_df": segmented_df,
            "profile_df": profile_df,
            "projection_df": projection_df,
        }

    results = st.session_state.segmentation_results
    if results is None:
        return

    st.markdown("---")
    aggregation_status = "aggregated to entity-level" if results["aggregated"] else "used as-is (already entity-level)"
    st.success(
        f"Best model: {results['best_model']} | data prep: {aggregation_status} | "
        f"features: {results['feature_count']}"
    )

    perf_df = create_clustering_performance_dataframe(
        results["performance"],
        results["all_metrics"],
        results["best_params"],
    )
    st.dataframe(perf_df, use_container_width=True)

    tab1, tab2, tab3 = st.tabs(["Segment Distribution", "2D Projection", "Segment Profiles"])

    with tab1:
        counts = (
            results["segmented_df"]["segment"]
            .value_counts()
            .rename_axis("segment")
            .reset_index(name="customers")
            .sort_values("customers", ascending=False)
        )
        fig = px.bar(counts, x="segment", y="customers", title="Customers per Segment")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        projection_df = results["projection_df"]
        variance = projection_df.attrs.get("variance_ratio", 1.0)
        st.caption(f"2D projection explained variance: {variance:.1%}")
        fig = px.scatter(
            projection_df,
            x="component_1",
            y="component_2",
            color="segment",
            title="Customer Segment Map (PCA Projection)",
            opacity=0.8,
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.dataframe(results["profile_df"], use_container_width=True)
        st.download_button(
            "Download segment assignments",
            results["segmented_df"].to_csv(index=False).encode("utf-8"),
            "customer_segments.csv",
            "text/csv",
            use_container_width=True,
            key="segment_download_csv",
        )


def _compute_item_cooccurrence(
    df: pd.DataFrame,
    user_col: str,
    item_col: str,
    min_support: int = 2,
) -> tuple[dict[str, list[tuple[str, int, float]]], dict[str, int], int]:
    """Build item-to-item co-occurrence recommendations."""
    interactions = df[[user_col, item_col]].dropna().copy()
    interactions[user_col] = interactions[user_col].astype(str)
    interactions[item_col] = interactions[item_col].astype(str)

    grouped = interactions.groupby(user_col)[item_col].apply(lambda s: list(pd.unique(s)))

    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    item_counts: dict[str, int] = defaultdict(int)

    for item_list in grouped:
        unique_items = sorted(set(item_list))
        if len(unique_items) > 80:
            unique_items = unique_items[:80]

        for item in unique_items:
            item_counts[item] += 1

        for item_a, item_b in itertools.combinations(unique_items, 2):
            pair_counts[(item_a, item_b)] += 1

    adjacency: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
    for (item_a, item_b), support in pair_counts.items():
        if support < min_support:
            continue
        conf_ab = support / max(1, item_counts[item_a])
        conf_ba = support / max(1, item_counts[item_b])
        adjacency[item_a].append((item_b, support, conf_ab))
        adjacency[item_b].append((item_a, support, conf_ba))

    for item, neighbors in adjacency.items():
        neighbors.sort(key=lambda row: (row[1], row[2]), reverse=True)

    return adjacency, dict(item_counts), len(interactions)


def render_ecommerce_recommendation_tab(df: pd.DataFrame) -> None:
    """Render e-commerce recommendation workflows."""
    st.markdown("### E-commerce Recommendation Systems")
    st.markdown(
        "Use product + user behavior data to produce recommendation candidates and "
        "optional model-based purchase propensity scores."
    )

    topn_tab, model_tab = st.tabs(["Top-N Co-occurrence", "Model-based Propensity"])

    with topn_tab:
        user_default = _default_column_by_keywords(list(df.columns), ["user_id", "customer_id", "client_id", "user", "customer"])
        item_default = _default_column_by_keywords(list(df.columns), ["product_id", "item_id", "sku", "product", "item"])

        user_idx = list(df.columns).index(user_default) if user_default in df.columns else 0
        item_idx = list(df.columns).index(item_default) if item_default in df.columns else min(1, len(df.columns) - 1)

        col1, col2, col3 = st.columns(3)
        with col1:
            user_col = st.selectbox("User column", list(df.columns), index=user_idx, key="reco_user_col")
        with col2:
            item_col = st.selectbox("Item column", list(df.columns), index=item_idx, key="reco_item_col")
        with col3:
            min_support = st.slider("Minimum co-occurrence support", 1, 20, 2, key="reco_min_support")

        top_n = st.slider("Recommendations per item", 3, 30, 10, key="reco_top_n")
        build_btn = st.button("Build recommendation candidates", type="primary", use_container_width=True, key="reco_build_btn")

        if "reco_candidates" not in st.session_state:
            st.session_state.reco_candidates = None

        if build_btn:
            with st.spinner("Mining co-occurrence patterns..."):
                adjacency, item_counts, interaction_count = _compute_item_cooccurrence(
                    df,
                    user_col=user_col,
                    item_col=item_col,
                    min_support=min_support,
                )

            st.session_state.reco_candidates = {
                "adjacency": adjacency,
                "item_counts": item_counts,
                "interaction_count": interaction_count,
                "user_col": user_col,
                "item_col": item_col,
                "top_n": top_n,
            }

        reco_state = st.session_state.reco_candidates
        if reco_state:
            st.info(
                f"Interactions: {reco_state['interaction_count']:,} | "
                f"Items: {len(reco_state['item_counts']):,} | "
                f"Items with neighbors: {len(reco_state['adjacency']):,}"
            )

            popular_items = sorted(
                reco_state["item_counts"].items(),
                key=lambda pair: pair[1],
                reverse=True,
            )
            candidate_items = [item for item, _ in popular_items if item in reco_state["adjacency"]]

            if not candidate_items:
                st.warning("No item co-occurrence neighbors found. Lower min support or check data sparsity.")
            else:
                anchor_item = st.selectbox("Anchor item", candidate_items, key="reco_anchor_item")
                recs = reco_state["adjacency"][anchor_item][: reco_state["top_n"]]
                rec_df = pd.DataFrame(recs, columns=["recommended_item", "support", "confidence"])
                st.dataframe(rec_df, use_container_width=True)

    with model_tab:
        st.caption(
            "This path uses automatic category encoding, class-imbalance handling, "
            "and quick multi-model selection for purchase propensity modeling."
        )

        target_default = _default_column_by_keywords(
            list(df.columns),
            ["purchased", "purchase", "converted", "conversion", "clicked", "target", "label"],
        )
        target_idx = list(df.columns).index(target_default) if target_default in df.columns else 0

        target_col = st.selectbox("Binary target column", list(df.columns), index=target_idx, key="reco_target_col")

        cls_models, _ = get_available_models()
        model_names = list(cls_models.keys())
        default_model_names = [m for m in ["RandomForestClassifier", "LogisticRegression", "XGBClassifier"] if m in model_names]
        if not default_model_names:
            default_model_names = model_names[: min(3, len(model_names))]

        col1, col2, col3 = st.columns(3)
        with col1:
            test_size = st.slider("Test split", 0.1, 0.4, 0.2, step=0.05, key="reco_test_size")
            imbalance = st.selectbox(
                "Imbalance handling",
                ["none", "class_weight", "smote", "adasyn", "undersample"],
                index=2,
                key="reco_imbalance",
            )
        with col2:
            metric = st.selectbox(
                "Optimize metric",
                ["accuracy", "balanced_accuracy", "f1", "roc_auc", "precision", "recall"],
                index=3,
                key="reco_metric",
            )
            cv_fold = st.selectbox("CV folds", ["Auto", "3", "5", "10"], index=0, key="reco_cv")
        with col3:
            n_estimators = st.slider("Tree estimators", 50, 500, 200, step=50, key="reco_estimators")
            tree_depth_raw = st.selectbox(
                "Max tree depth",
                ["None"] + [str(i) for i in range(2, 21)],
                index=0,
                key="reco_tree_depth",
            )
            tree_depth = None if tree_depth_raw == "None" else int(tree_depth_raw)

        selected_models = st.multiselect(
            "Classification models",
            model_names,
            default=default_model_names,
            key="reco_models",
        )

        train_btn = st.button("Train propensity models", type="primary", use_container_width=True, key="reco_train_btn")

        if "reco_model_results" not in st.session_state:
            st.session_state.reco_model_results = None

        if train_btn:
            if not selected_models:
                st.error("Please select at least one model.")
                return

            working_df = df.copy()
            try:
                working_df[target_col] = _try_binary_encode_target(working_df[target_col])
            except Exception as exc:
                st.error(str(exc))
                return

            with st.spinner("Preprocessing product + behavior data..."):
                try:
                    X_train_proc, X_test_proc, y_train, y_test, _, column_info = preprocess_and_split(
                        working_df,
                        target_col,
                        test_size=test_size,
                    )
                except Exception as exc:
                    st.error(f"Preprocessing failed: {exc}")
                    return

            with st.expander("Preprocessing summary", expanded=False):
                st.markdown(get_preprocessing_summary(column_info))

            class_weight = None
            X_train_bal = X_train_proc
            y_train_bal = y_train
            if imbalance != "none":
                X_train_bal, y_train_bal, class_weight = apply_imbalance_handling(X_train_proc, y_train, imbalance)

            tuned_models = selected_models
            if class_weight == "balanced":
                st.info(
                    "class_weight=balanced requested. Models with native class weights "
                    "are recommended in this mode (RandomForestClassifier, LogisticRegression)."
                )

            progress_text = st.empty()
            progress = st.progress(0)

            def _progress(current: int, total: int, model_name: str | None) -> None:
                if total <= 0:
                    return
                progress.progress(min(1.0, current / total))
                if model_name:
                    progress_text.text(f"Training {model_name} ({current}/{total})")
                else:
                    progress_text.text("Training complete")

            try:
                performance, best_model, use_cv, cv_folds, current_task, trained_models, best_params, all_metrics = train_multiple_models(
                    X_train_bal,
                    X_test_proc,
                    y_train_bal,
                    y_test,
                    task_type="classification",
                    selected_models_cls=tuned_models,
                    selected_models_reg=[],
                    selected_metric_cls=metric,
                    hyperparameter_tuning=False,
                    cv_fold_option=cv_fold,
                    progress_callback=_progress,
                    tree_max_depth=tree_depth,
                    n_estimators=n_estimators,
                    SVM_C=1.0,
                    KNN_neighbors=5,
                )
            except Exception as exc:
                progress.empty()
                progress_text.empty()
                st.error(f"Training failed: {exc}")
                return

            progress.empty()
            progress_text.empty()

            st.session_state.reco_model_results = {
                "target_col": target_col,
                "performance": performance,
                "best_model": best_model,
                "use_cv": use_cv,
                "cv_folds": cv_folds,
                "current_task": current_task,
                "trained_models": trained_models,
                "best_params": best_params,
                "all_metrics": all_metrics,
                "X_test": X_test_proc,
                "y_test": y_test,
                "test_indices": list(y_test.index),
                "metric": metric,
            }

        reco_results = st.session_state.reco_model_results
        if reco_results:
            st.success(
                f"Best model: {reco_results['best_model']} | "
                f"optimized metric: {reco_results['metric']}"
            )
            perf_df = create_performance_dataframe(
                reco_results["performance"],
                reco_results["best_params"],
                reco_results["all_metrics"],
            )
            st.dataframe(perf_df, use_container_width=True)

            model = reco_results["trained_models"].get(reco_results["best_model"])
            if model is not None:
                if hasattr(model, "predict_proba"):
                    probability = model.predict_proba(reco_results["X_test"])[:, 1]
                elif hasattr(model, "decision_function"):
                    raw = model.decision_function(reco_results["X_test"])
                    probability = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
                else:
                    probability = model.predict(reco_results["X_test"]).astype(float)

                rank_df = df.loc[reco_results["test_indices"]].copy()
                rank_df["purchase_probability"] = probability
                rank_df = rank_df.sort_values("purchase_probability", ascending=False)

                display_cols = _id_like_columns(rank_df)[:2]
                for col in [reco_results["target_col"], "purchase_probability"]:
                    if col in rank_df.columns and col not in display_cols:
                        display_cols.append(col)
                if not display_cols:
                    display_cols = ["purchase_probability"]

                st.markdown("Top propensity rows")
                max_rows = max(1, min(500, len(rank_df)))
                default_rows = min(50, max_rows)
                top_n = st.slider("Top N rows", 1, max_rows, default_rows, key="reco_top_rows")
                st.dataframe(rank_df[display_cols].head(top_n), use_container_width=True)


def _parse_timestamp_series(series: pd.Series) -> pd.Series:
    """Parse timestamps with numeric fallback units for robust auto-detection."""
    candidates: list[pd.Series] = [pd.to_datetime(series, errors="coerce")]

    if pd.api.types.is_numeric_dtype(series):
        for unit in ("s", "ms", "us", "ns"):
            try:
                candidates.append(pd.to_datetime(series, errors="coerce", unit=unit))
            except Exception:
                continue

    best = candidates[0]
    best_ratio = float(best.notna().mean())
    for parsed in candidates[1:]:
        ratio = float(parsed.notna().mean())
        if ratio > best_ratio:
            best_ratio = ratio
            best = parsed

    return best


def _evaluate_timestamp_quality(series: pd.Series) -> dict[str, Any]:
    """Evaluate timestamp validity and return quality diagnostics."""
    parsed = _parse_timestamp_series(series)
    total = int(len(parsed))
    valid_mask = parsed.notna()
    valid_count = int(valid_mask.sum())
    valid_ratio = valid_count / max(1, total)

    if valid_count == 0:
        return {
            "is_valid": False,
            "valid_ratio": 0.0,
            "unique_ratio": 0.0,
            "unique_count": 0,
            "span_seconds": 0.0,
            "epoch_1970_ratio": 1.0,
            "quality_score": 0.0,
            "parsed": parsed,
            "reason": "No parseable timestamps",
        }

    valid_values = parsed[valid_mask]
    unique_count = int(valid_values.nunique(dropna=True))
    unique_ratio = unique_count / max(1, valid_count)
    span_seconds = 0.0
    if unique_count > 1:
        span_seconds = float((valid_values.max() - valid_values.min()).total_seconds())

    epoch_1970_ratio = float((valid_values.dt.year == 1970).mean())
    min_unique = 10 if valid_count >= 200 else 3

    is_valid = (
        valid_ratio >= 0.60
        and unique_count >= min_unique
        and unique_ratio >= 0.01
        and epoch_1970_ratio < 0.80
        and span_seconds > 0
    )

    quality_score = (
        (0.50 * valid_ratio)
        + (0.25 * min(1.0, unique_ratio * 10))
        + (0.15 * (1.0 - min(1.0, epoch_1970_ratio)))
        + (0.10 if span_seconds > 0 else 0.0)
    )

    reason = "Valid temporal signal" if is_valid else "Weak or suspicious timestamp quality"

    return {
        "is_valid": bool(is_valid),
        "valid_ratio": float(valid_ratio),
        "unique_ratio": float(unique_ratio),
        "unique_count": int(unique_count),
        "span_seconds": float(span_seconds),
        "epoch_1970_ratio": float(epoch_1970_ratio),
        "quality_score": float(quality_score),
        "parsed": parsed,
        "reason": reason,
    }


def _detect_time_axis(df: pd.DataFrame) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Find the best timestamp candidate and return ranked diagnostics."""
    keyword_tokens = [
        "timestamp",
        "datetime",
        "event_time",
        "event_date",
        "date",
        "time",
        "created",
        "occurred",
    ]

    candidates: list[dict[str, Any]] = []
    for col in df.columns:
        series = df[col]
        col_name = str(col).lower()
        name_hint = any(token in col_name for token in keyword_tokens)

        if not (
            name_hint
            or pd.api.types.is_datetime64_any_dtype(series)
            or pd.api.types.is_numeric_dtype(series)
            or series.dtype == object
        ):
            continue

        quality = _evaluate_timestamp_quality(series)
        score = quality["quality_score"] + (0.10 if name_hint else 0.0)
        candidates.append(
            {
                "column": col,
                "name_hint": name_hint,
                "score": float(score),
                **quality,
            }
        )

    if not candidates:
        return None, []

    candidates.sort(key=lambda item: item["score"], reverse=True)
    best = candidates[0]
    return best, candidates[:5]


def _resolve_detection_mode(df: pd.DataFrame, requested_mode: str) -> dict[str, Any]:
    """Resolve auto/time-series/tabular mode using timestamp quality checks."""
    best_ts, ranked_candidates = _detect_time_axis(df)
    ts_valid = bool(best_ts and best_ts.get("is_valid"))

    if requested_mode == "Auto":
        resolved_mode = "Time-series" if ts_valid else "Tabular"
        fallback = False
    elif requested_mode == "Time-series":
        resolved_mode = "Time-series" if ts_valid else "Tabular"
        fallback = not ts_valid
    else:
        resolved_mode = "Tabular"
        fallback = False

    return {
        "requested_mode": requested_mode,
        "resolved_mode": resolved_mode,
        "timestamp": best_ts,
        "timestamp_candidates": ranked_candidates,
        "fallback_to_tabular": fallback,
    }


def _select_numeric_features(
    df: pd.DataFrame,
    max_features: int = 50,
    exclude_cols: list[str] | None = None,
) -> tuple[list[str], dict[str, list[str]]]:
    """Pick robust numeric features and drop ID-like / constant columns."""
    exclude = set(exclude_cols or [])
    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns.tolist() if col not in exclude
    ]

    dropped = {
        "id_like": [],
        "constant": [],
        "near_constant": [],
        "all_missing": [],
    }
    valid_cols: list[str] = []
    col_variance: dict[str, float] = {}

    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        non_null = series.dropna()

        if non_null.empty:
            dropped["all_missing"].append(col)
            continue

        nunique = int(non_null.nunique(dropna=True))
        unique_ratio = nunique / max(1, len(non_null))
        mode_ratio = float(non_null.value_counts(normalize=True, dropna=True).iloc[0])

        if unique_ratio > 0.995 and nunique > min(100, max(20, len(non_null) // 5)):
            dropped["id_like"].append(col)
            continue

        if nunique <= 1:
            dropped["constant"].append(col)
            continue

        if mode_ratio >= 0.995:
            dropped["near_constant"].append(col)
            continue

        valid_cols.append(col)
        col_variance[col] = float(non_null.var(ddof=0))

    valid_cols.sort(key=lambda col: col_variance.get(col, 0.0), reverse=True)
    selected = valid_cols[:max_features]
    return selected, dropped


def _impute_numeric_frame(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Median imputation with inf handling for numeric frames."""
    cleaned = feature_df.replace([np.inf, -np.inf], np.nan)
    medians = cleaned.median(numeric_only=True)
    cleaned = cleaned.fillna(medians)
    cleaned = cleaned.fillna(0)
    return cleaned


def _build_tabular_feature_frame(
    df: pd.DataFrame,
    base_numeric_cols: list[str],
    max_features: int,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Create non-temporal statistical feature frame."""
    base_cols = base_numeric_cols[:max_features]
    feature_df = df[base_cols].apply(pd.to_numeric, errors="coerce") if base_cols else pd.DataFrame(index=df.index)
    feature_df = _impute_numeric_frame(feature_df)

    remaining_slots = max(0, max_features - len(feature_df.columns))
    if remaining_slots > 0 and len(feature_df.columns) > 0:
        stats: dict[str, pd.Series] = {
            "row_mean": feature_df.mean(axis=1),
            "row_std": feature_df.std(axis=1, ddof=0),
            "row_min": feature_df.min(axis=1),
            "row_max": feature_df.max(axis=1),
            "row_abs_mean": feature_df.abs().mean(axis=1),
            "row_l2_norm": np.sqrt((feature_df ** 2).sum(axis=1)),
        }
        for name in list(stats.keys())[:remaining_slots]:
            feature_df[name] = stats[name]

    return feature_df, list(feature_df.columns), base_cols


def _build_timeseries_feature_frame_adaptive(
    df: pd.DataFrame,
    timestamp_col: str,
    parsed_timestamp: pd.Series,
    base_numeric_cols: list[str],
    window: int,
    max_features: int,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Create lag/rolling temporal features with a hard feature budget."""
    max_base = max(1, min(len(base_numeric_cols), max_features // 5))
    base_cols = base_numeric_cols[:max_base]

    working = df[base_cols].copy()
    working[timestamp_col] = parsed_timestamp
    working = working.dropna(subset=[timestamp_col]).sort_values(timestamp_col)

    if working.empty:
        raise ValueError("No valid timestamp rows available for time-series mode.")

    feature_parts: dict[str, pd.Series] = {}
    for col in base_cols:
        values = pd.to_numeric(working[col], errors="coerce")
        feature_parts[col] = values
        feature_parts[f"{col}_lag1"] = values.shift(1)
        feature_parts[f"{col}_diff1"] = values.diff(1)
        feature_parts[f"{col}_roll_mean"] = values.rolling(window=window, min_periods=2).mean()
        feature_parts[f"{col}_roll_std"] = values.rolling(window=window, min_periods=2).std()

    feature_parts["time_delta_seconds"] = working[timestamp_col].diff().dt.total_seconds()

    feature_df = pd.DataFrame(feature_parts, index=working.index)
    feature_df = _impute_numeric_frame(feature_df)

    if len(feature_df.columns) > max_features:
        variances = feature_df.var(numeric_only=True).sort_values(ascending=False)
        selected_cols = variances.index[:max_features].tolist()
        feature_df = feature_df[selected_cols]

    return feature_df, list(feature_df.columns), base_cols


def _run_anomaly_model(
    feature_df: pd.DataFrame,
    method: str,
    contamination: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Run a lightweight anomaly model in one pass."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df)

    if method == "IsolationForest":
        model = IsolationForest(
            contamination=contamination,
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        )
        labels = model.fit_predict(X_scaled)
        scores = -model.score_samples(X_scaled)
    elif method == "LocalOutlierFactor":
        n_neighbors = max(5, min(35, max(6, X_scaled.shape[0] // 20)))
        n_neighbors = min(n_neighbors, max(2, X_scaled.shape[0] - 1))
        model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        labels = model.fit_predict(X_scaled)
        scores = -model.negative_outlier_factor_
    else:
        raise ValueError(f"Unsupported method: {method}")

    return labels, scores, X_scaled, model


def _compute_feature_importance_proxy(
    feature_df: pd.DataFrame,
    labels: np.ndarray,
    model: Any,
) -> pd.DataFrame:
    """Create feature-importance view using model or anomaly-separation proxy."""
    if hasattr(model, "feature_importances_"):
        raw = np.asarray(model.feature_importances_).reshape(-1)
        importance = pd.Series(raw, index=feature_df.columns)
    else:
        anomalies = feature_df[labels == -1]
        normal = feature_df[labels == 1]

        if anomalies.empty or normal.empty:
            importance = feature_df.std(ddof=0).fillna(0)
        else:
            spread = (feature_df.quantile(0.75) - feature_df.quantile(0.25)).abs() + 1e-9
            separation = (anomalies.median() - normal.median()).abs()
            importance = (separation / spread).fillna(0)

    importance = importance.sort_values(ascending=False)
    return pd.DataFrame({"feature": importance.index, "importance": importance.values})


def _project_for_anomaly_scatter(X_scaled: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Create 2D PCA projection for anomaly separation plot."""
    if X_scaled.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=42)
        projected = pca.fit_transform(X_scaled)
        var_ratio = pca.explained_variance_ratio_.sum()
    else:
        projected = np.column_stack([X_scaled[:, 0], np.zeros(X_scaled.shape[0])])
        var_ratio = 1.0

    pca_df = pd.DataFrame(
        {
            "component_1": projected[:, 0],
            "component_2": projected[:, 1],
            "label": np.where(labels == -1, "Anomaly", "Normal"),
        }
    )
    pca_df.attrs["variance_ratio"] = float(var_ratio)
    return pca_df


def render_iot_anomaly_tab(df: pd.DataFrame) -> None:
    """Render generic, self-adaptive anomaly detection workflow."""
    st.markdown("### Generic Adaptive Anomaly Detection")
    st.markdown(
        "Automatically detects whether your data is time-series or tabular, "
        "adapts feature engineering, and runs a lightweight anomaly model."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        requested_mode = st.selectbox(
            "Mode selector",
            ["Auto", "Time-series", "Tabular"],
            index=0,
            key="adaptive_anomaly_mode",
        )
    with col2:
        method = st.selectbox(
            "Detection model",
            ["IsolationForest", "LocalOutlierFactor"],
            index=0,
            key="adaptive_anomaly_model",
        )
    with col3:
        contamination = st.slider(
            "Expected anomaly rate",
            min_value=0.01,
            max_value=0.20,
            value=0.05,
            step=0.01,
            key="adaptive_anomaly_contamination",
        )

    rolling_window = st.slider(
        "Rolling window (used in time-series mode)",
        min_value=3,
        max_value=60,
        value=12,
        key="adaptive_anomaly_window",
    )

    run_btn = st.button(
        "Run adaptive anomaly detection",
        type="primary",
        use_container_width=True,
        key="adaptive_anomaly_run",
    )

    if "adaptive_anomaly_results" not in st.session_state:
        st.session_state.adaptive_anomaly_results = None

    if run_btn:
        resolution = _resolve_detection_mode(df, requested_mode)
        timestamp_info = resolution["timestamp"]
        resolved_mode = resolution["resolved_mode"]

        exclude_cols = []
        if resolved_mode == "Time-series" and timestamp_info is not None:
            exclude_cols.append(timestamp_info["column"])

        selected_numeric, dropped = _select_numeric_features(
            df,
            max_features=50,
            exclude_cols=exclude_cols,
        )

        if not selected_numeric:
            st.error(
                "No usable numeric columns after automatic filtering. "
                "Please provide at least one non-ID numeric feature."
            )
            return

        try:
            if resolved_mode == "Time-series":
                assert timestamp_info is not None
                feature_df, feature_cols, base_features = _build_timeseries_feature_frame_adaptive(
                    df=df,
                    timestamp_col=timestamp_info["column"],
                    parsed_timestamp=timestamp_info["parsed"],
                    base_numeric_cols=selected_numeric,
                    window=rolling_window,
                    max_features=50,
                )
                aligned_df = df.loc[feature_df.index].copy()
                aligned_df["_resolved_timestamp"] = timestamp_info["parsed"].loc[feature_df.index]
                timestamp_col = "_resolved_timestamp"
            else:
                feature_df, feature_cols, base_features = _build_tabular_feature_frame(
                    df=df,
                    base_numeric_cols=selected_numeric,
                    max_features=50,
                )
                aligned_df = df.loc[feature_df.index].copy()
                timestamp_col = None
        except Exception as exc:
            st.error(f"Feature engineering failed: {exc}")
            return

        if feature_df.empty:
            st.error("Feature frame is empty after preprocessing.")
            return

        try:
            labels, scores, X_scaled, model = _run_anomaly_model(
                feature_df=feature_df,
                method=method,
                contamination=contamination,
            )
        except Exception as exc:
            st.error(f"Anomaly model failed: {exc}")
            return

        output = aligned_df.copy()
        output["anomaly_score"] = scores
        output["anomaly_label"] = np.where(labels == -1, "Anomaly", "Normal")
        output["is_anomaly"] = (labels == -1).astype(int)

        importance_df = _compute_feature_importance_proxy(feature_df, labels, model)
        pca_df = _project_for_anomaly_scatter(X_scaled, labels)

        st.session_state.adaptive_anomaly_results = {
            "requested_mode": requested_mode,
            "resolved_mode": resolved_mode,
            "fallback_to_tabular": resolution["fallback_to_tabular"],
            "timestamp_info": timestamp_info,
            "timestamp_candidates": resolution["timestamp_candidates"],
            "dropped": dropped,
            "model": method,
            "contamination": contamination,
            "feature_cols": feature_cols,
            "base_features": base_features,
            "feature_frame": feature_df,
            "output": output,
            "pca_df": pca_df,
            "importance_df": importance_df,
            "timestamp_col": timestamp_col,
        }

    results = st.session_state.adaptive_anomaly_results
    if results is None:
        return

    output = results["output"]
    anomaly_count = int(output["is_anomaly"].sum())
    total_count = int(len(output))
    anomaly_rate = anomaly_count / max(1, total_count)

    st.markdown("---")
    if results["requested_mode"] == "Auto":
        st.info(f"Auto mode selected: **{results['resolved_mode']}**")
    elif results["fallback_to_tabular"]:
        st.warning("Requested Time-series mode, but timestamp quality was weak. Switched to **Tabular** automatically.")
    else:
        st.info(f"Selected mode: **{results['resolved_mode']}**")

    st.success(
        f"Model: {results['model']} | anomalies: {anomaly_count:,}/{total_count:,} ({anomaly_rate:.2%})"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total records", f"{total_count:,}")
    m2.metric("Anomalies", f"{anomaly_count:,}")
    m3.metric("Anomaly rate", f"{anomaly_rate:.2%}")
    m4.metric("Used features", len(results["feature_cols"]))

    ts_info = results.get("timestamp_info")
    if ts_info is not None:
        with st.expander("Timestamp quality diagnostics", expanded=False):
            st.write(
                {
                    "column": ts_info.get("column"),
                    "valid_ratio": round(ts_info.get("valid_ratio", 0.0), 4),
                    "unique_ratio": round(ts_info.get("unique_ratio", 0.0), 4),
                    "unique_count": ts_info.get("unique_count", 0),
                    "span_seconds": round(ts_info.get("span_seconds", 0.0), 2),
                    "epoch_1970_ratio": round(ts_info.get("epoch_1970_ratio", 0.0), 4),
                    "quality_score": round(ts_info.get("quality_score", 0.0), 4),
                    "is_valid": ts_info.get("is_valid", False),
                    "reason": ts_info.get("reason", ""),
                }
            )

    with st.expander("Auto feature filtering summary", expanded=False):
        st.write(
            {
                "base_features_used": results["base_features"],
                "total_feature_count": len(results["feature_cols"]),
                "dropped_columns": results["dropped"],
            }
        )

    tabs: list[str] = [
        "Anomaly Distribution",
        "PCA Separation",
        "Feature Importance",
        "Top Anomalous Records",
    ]
    include_timeline = results["resolved_mode"] == "Time-series" and results.get("timestamp_col") is not None
    if include_timeline:
        tabs.insert(3, "Timeline")

    tab_objects = st.tabs(tabs)
    tab_index = {name: idx for idx, name in enumerate(tabs)}

    with tab_objects[tab_index["Anomaly Distribution"]]:
        dist_fig = px.histogram(
            output,
            x="anomaly_score",
            color="anomaly_label",
            nbins=50,
            title="Anomaly Score Distribution",
            barmode="overlay",
            color_discrete_map={"Normal": "#5d7a6a", "Anomaly": "#c13c44"},
        )
        dist_fig.update_layout(legend_title_text="Label")
        st.plotly_chart(dist_fig, use_container_width=True)

        ratio_df = pd.DataFrame(
            {
                "label": ["Anomaly", "Normal"],
                "count": [anomaly_count, total_count - anomaly_count],
            }
        )
        pie_fig = px.pie(
            ratio_df,
            values="count",
            names="label",
            title="Anomaly vs Normal Share",
            color="label",
            color_discrete_map={"Normal": "#5d7a6a", "Anomaly": "#c13c44"},
        )
        st.plotly_chart(pie_fig, use_container_width=True)

    with tab_objects[tab_index["PCA Separation"]]:
        pca_df = results["pca_df"]
        variance_ratio = pca_df.attrs.get("variance_ratio", 1.0)
        st.caption(f"PCA explained variance (2D): {variance_ratio:.1%}")
        scatter_fig = px.scatter(
            pca_df,
            x="component_1",
            y="component_2",
            color="label",
            opacity=0.75,
            title="Anomaly Separation (PCA Projection)",
            color_discrete_map={"Normal": "#5d7a6a", "Anomaly": "#c13c44"},
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

    with tab_objects[tab_index["Feature Importance"]]:
        importance_df = results["importance_df"].head(20)
        importance_fig = px.bar(
            importance_df,
            x="importance",
            y="feature",
            orientation="h",
            title="Top Features Driving Anomaly Separation",
        )
        importance_fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(importance_fig, use_container_width=True)

        if not importance_df.empty:
            top_feature = str(importance_df.iloc[0]["feature"])
            if top_feature in results["feature_frame"].columns:
                box_df = pd.DataFrame(
                    {
                        "value": results["feature_frame"][top_feature],
                        "label": output["anomaly_label"].values,
                    }
                )
                box_fig = px.box(
                    box_df,
                    x="label",
                    y="value",
                    color="label",
                    title=f"Distribution Shift for Top Feature: {top_feature}",
                    color_discrete_map={"Normal": "#5d7a6a", "Anomaly": "#c13c44"},
                )
                st.plotly_chart(box_fig, use_container_width=True)

    if include_timeline:
        with tab_objects[tab_index["Timeline"]]:
            timestamp_col = results["timestamp_col"]
            timeline_df = output.sort_values(timestamp_col)
            signal_options = [col for col in results["base_features"] if col in timeline_df.columns]

            if signal_options:
                selected_signal = st.selectbox(
                    "Signal for timeline",
                    signal_options,
                    index=0,
                    key="adaptive_anomaly_timeline_signal",
                )

                timeline_fig = go.Figure()
                timeline_fig.add_trace(
                    go.Scatter(
                        x=timeline_df[timestamp_col],
                        y=timeline_df[selected_signal],
                        mode="lines",
                        name="Signal",
                        line=dict(color="#4b6f61", width=1.5),
                    )
                )

                anom_df = timeline_df[timeline_df["is_anomaly"] == 1]
                if not anom_df.empty:
                    timeline_fig.add_trace(
                        go.Scatter(
                            x=anom_df[timestamp_col],
                            y=anom_df[selected_signal],
                            mode="markers",
                            name="Anomaly",
                            marker=dict(color="#c13c44", size=8, symbol="x"),
                        )
                    )

                timeline_fig.update_layout(
                    title="Anomaly Timeline",
                    xaxis_title="Time",
                    yaxis_title=selected_signal,
                    legend=dict(orientation="h"),
                )
                st.plotly_chart(timeline_fig, use_container_width=True)
            else:
                st.info("No original numeric signal available for timeline plotting.")

    with tab_objects[tab_index["Top Anomalous Records"]]:
        top_n = min(200, len(output))
        ranked = output.sort_values("anomaly_score", ascending=False).head(top_n)
        st.dataframe(ranked, use_container_width=True)

        st.download_button(
            "Download anomaly results",
            output.to_csv(index=False).encode("utf-8"),
            "adaptive_anomaly_results.csv",
            "text/csv",
            use_container_width=True,
            key="adaptive_anomaly_download",
        )
