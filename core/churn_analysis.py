# ==============================
# ./core/churn_analysis.py
# ==============================
"""
Customer Churn Specialization Module
Provides end-to-end pipeline: data preview, imbalance handling,
model training, and churn-specific visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# Persistence & drift extras
import joblib
import io
import json
import hashlib
from datetime import datetime
from scipy import stats as _scipy_stats

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# COLUMN DETECTION
# ─────────────────────────────────────────────────────────────

CHURN_KEYWORDS = [
    "churn", "churned", "attrition", "retention", "cancelled", "canceled",
    "subscription_status", "customer_status", "status", "active", "inactive",
    "left", "exited", "exit", "defected", "defection", "stayed", "loyal"
]

def detect_churn_column(df: pd.DataFrame) -> str | None:
    """
    Heuristically identify the most likely churn/target column.
    Returns column name or None if not found.
    """
    cols_lower = {c.lower().replace(" ", "_"): c for c in df.columns}

    # Exact keyword matches first
    for kw in CHURN_KEYWORDS:
        if kw in cols_lower:
            return cols_lower[kw]

    # Partial match on column names
    for kw in ["churn", "attrition", "exit", "cancel", "active"]:
        for col_l, col in cols_lower.items():
            if kw in col_l:
                return col

    # Fallback: binary columns that look like flags
    for col in df.columns:
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) == 2:
            vals_set = set(str(v).lower() for v in unique_vals)
            binary_pairs = [
                {"0", "1"}, {"yes", "no"}, {"true", "false"},
                {"active", "inactive"}, {"active", "cancelled"},
                {"churned", "not churned"}, {"1.0", "0.0"}
            ]
            for pair in binary_pairs:
                if vals_set == pair:
                    return col
    return None


def encode_churn_target(series: pd.Series) -> pd.Series:
    """
    Encode any churn column representation to binary int (0 = retained, 1 = churned).
    Handles: Yes/No, 1/0, True/False, active/cancelled/paused, etc.

    Multi-class columns are collapsed: known "churned" labels → 1, all others → 0.
    """
    s = series.copy()

    churn_positive = {
        "yes", "1", "1.0", "true", "churned", "cancelled", "canceled",
        "inactive", "left", "exited", "exit", "defected"
    }
    retain_positive = {
        "no", "0", "0.0", "false", "retained", "active", "staying",
        "loyal", "not churned", "stayed", "paused", "pending"
    }

    vals_norm = s.dropna().astype(str).str.strip().str.lower().unique()
    vals_set = set(vals_norm)

    # If all values fall into known buckets, map directly
    if vals_set <= (churn_positive | retain_positive):
        return (
            s.astype(str).str.strip().str.lower()
            .map(lambda v: 1 if v in churn_positive else 0)
            .astype(int)
        )

    # If ANY value is in churn_positive (e.g. mixed multi-class), churn = known
    # churn keywords, retain = everything else
    if vals_set & churn_positive:
        return (
            s.astype(str).str.strip().str.lower()
            .map(lambda v: 1 if v in churn_positive else 0)
            .astype(int)
        )

    # Pure numeric column
    if pd.api.types.is_numeric_dtype(s):
        unique_sorted = sorted(s.dropna().unique())
        if len(unique_sorted) == 2:
            # Map lower value → 0 (retained), higher → 1 (churned)
            return s.map({unique_sorted[0]: 0, unique_sorted[1]: 1}).fillna(0).astype(int)
        # Multi-value numeric: use minority class as 1 (churn)
        vc = s.value_counts()
        minority = vc.index[-1]
        return (s == minority).astype(int)

    # Fallback: minority string class → churned (1)
    vc = s.astype(str).str.lower().value_counts()
    minority_label = vc.index[-1]
    return (s.astype(str).str.lower() == minority_label).astype(int)


# ─────────────────────────────────────────────────────────────
# DATA PREVIEW CHARTS
# ─────────────────────────────────────────────────────────────

def plot_churn_rate_overview(df: pd.DataFrame, churn_col: str):
    """Donut chart + key KPI metrics for overall churn rate."""
    y = encode_churn_target(df[churn_col])
    n_total = len(y)
    n_churned = int(y.sum())
    n_retained = n_total - n_churned
    churn_rate = n_churned / n_total * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{n_total:,}")
    col2.metric("Churned", f"{n_churned:,}", delta=f"{churn_rate:.1f}%", delta_color="inverse")
    col3.metric("Retained", f"{n_retained:,}", delta=f"{100 - churn_rate:.1f}%")

    fig = go.Figure(go.Pie(
        labels=["Retained", "Churned"],
        values=[n_retained, n_churned],
        hole=0.55,
        marker_colors=["#22c55e", "#ef4444"],
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>"
    ))
    fig.update_layout(
        title_text="Churn vs Retention Overview",
        annotations=[{
            "text": f"{churn_rate:.1f}%<br>Churn",
            "x": 0.5, "y": 0.5,
            "font_size": 18,
            "showarrow": False,
            "font_color": "#ef4444"
        }],
        legend=dict(orientation="h", y=-0.05),
        margin=dict(t=60, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Imbalance warning
    ratio = n_churned / max(n_retained, 1)
    if ratio < 0.15:
        st.error(f"⚠️ Severe class imbalance detected! Churn rate = {churn_rate:.1f}%. "
                 "Consider enabling SMOTE or class weighting.")
    elif ratio < 0.3:
        st.warning(f"⚠️ Moderate class imbalance (churn rate = {churn_rate:.1f}%). "
                   "SMOTE or class weighting is recommended.")
    else:
        st.success(f"✅ Class balance is acceptable (churn rate = {churn_rate:.1f}%).")


def plot_churn_by_numeric_features(df: pd.DataFrame, churn_col: str, max_features: int = 8):
    """Box/violin plots of numeric features split by churn status."""
    y = encode_churn_target(df[churn_col])
    df_viz = df.drop(columns=[churn_col]).copy()
    df_viz["__churn__"] = y.map({0: "Retained", 1: "Churned"})

    numeric_cols = df_viz.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "__churn__"][:max_features]

    if not numeric_cols:
        st.info("No numeric features found for distribution analysis.")
        return

    n = len(numeric_cols)
    cols_per_row = 2
    rows = (n + cols_per_row - 1) // cols_per_row

    fig = make_subplots(rows=rows, cols=cols_per_row,
                        subplot_titles=numeric_cols,
                        vertical_spacing=0.1, horizontal_spacing=0.07)

    colors = {"Retained": "#22c55e", "Churned": "#ef4444"}

    for i, col in enumerate(numeric_cols):
        row = i // cols_per_row + 1
        col_idx = i % cols_per_row + 1
        for status in ["Retained", "Churned"]:
            vals = df_viz[df_viz["__churn__"] == status][col].dropna()
            fig.add_trace(
                go.Box(
                    y=vals,
                    name=status,
                    marker_color=colors[status],
                    showlegend=(i == 0),
                    legendgroup=status,
                    boxpoints="outliers"
                ),
                row=row, col=col_idx
            )

    fig.update_layout(
        height=250 * rows,
        title_text="Numeric Feature Distributions by Churn Status",
        legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"),
        margin=dict(t=60, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_churn_by_categorical_features(df: pd.DataFrame, churn_col: str, max_features: int = 6):
    """Stacked percentage bar charts for categorical features vs churn."""
    y = encode_churn_target(df[churn_col])
    df_viz = df.drop(columns=[churn_col]).copy()
    df_viz["__churn__"] = y

    cat_cols = df_viz.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != "__churn__" and df_viz[c].nunique() <= 20][:max_features]

    if not cat_cols:
        st.info("No suitable categorical features found (must have ≤ 20 unique values).")
        return

    figs = []
    for col in cat_cols:
        grouped = (
            df_viz.groupby([col, "__churn__"])
            .size()
            .reset_index(name="count")
        )
        totals = grouped.groupby(col)["count"].transform("sum")
        grouped["pct"] = grouped["count"] / totals * 100
        churn_df = grouped[grouped["__churn__"] == 1].sort_values("pct", ascending=False)

        fig = px.bar(
            churn_df,
            x=col,
            y="pct",
            text=churn_df["pct"].apply(lambda v: f"{v:.1f}%"),
            title=f"Churn Rate by {col}",
            color="pct",
            color_continuous_scale=["#22c55e", "#facc15", "#ef4444"],
            range_color=[0, 100],
            labels={"pct": "Churn Rate (%)"},
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            coloraxis_showscale=False,
            margin=dict(t=50, b=40),
            height=350,
            showlegend=False
        )
        figs.append(fig)

    cols_layout = st.columns(min(2, len(figs)))
    for i, fig in enumerate(figs):
        cols_layout[i % 2].plotly_chart(fig, use_container_width=True)


def plot_churn_correlations(df: pd.DataFrame, churn_col: str):
    """Bar chart of feature correlations with the churn label."""
    y = encode_churn_target(df[churn_col])
    df_enc = df.drop(columns=[churn_col]).copy()

    # Encode categoricals numerically for correlation
    for col in df_enc.select_dtypes(include=["object", "category"]).columns:
        try:
            df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))
        except Exception:
            df_enc.drop(columns=[col], inplace=True)

    df_enc = df_enc.select_dtypes(include=["int64", "float64", "int32", "float32"])
    if df_enc.empty:
        st.info("No numeric features available for correlation analysis.")
        return

    corrs = df_enc.apply(lambda c: c.corr(y.rename(None))).dropna()
    corrs = corrs.sort_values(key=lambda x: x.abs(), ascending=False).head(20)

    colors = ["#ef4444" if v >= 0 else "#3b82f6" for v in corrs.values]
    fig = go.Figure(go.Bar(
        x=corrs.values,
        y=corrs.index,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in corrs.values],
        textposition="outside",
        hovertemplate="%{y}: %{x:.4f}<extra></extra>"
    ))
    fig.update_layout(
        title_text="Feature Correlation with Churn (Pearson)",
        xaxis_title="Correlation coefficient",
        height=max(300, 25 * len(corrs)),
        margin=dict(t=50, l=200)
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_churn_missing_values(df: pd.DataFrame):
    """Bar chart of missing values percentage per column."""
    missing_pct = df.isnull().mean() * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)

    if missing_pct.empty:
        st.success("✅ No missing values detected.")
        return

    fig = px.bar(
        x=missing_pct.values,
        y=missing_pct.index,
        orientation="h",
        text=[f"{v:.1f}%" for v in missing_pct.values],
        color=missing_pct.values,
        color_continuous_scale=["#facc15", "#ef4444"],
        labels={"x": "Missing %", "y": "Column"},
        title="Missing Values by Column"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(coloraxis_showscale=False, margin=dict(t=50))
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# POST-TRAINING CHARTS
# ─────────────────────────────────────────────────────────────

def plot_roc_curves(y_test, proba_dict: dict):
    """Overlay ROC curves for all models."""
    fig = go.Figure()

    colors = px.colors.qualitative.Plotly

    for i, (model_name, y_proba) in enumerate(proba_dict.items()):
        if y_proba is None:
            continue
        try:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_score = roc_auc_score(y_test, y_proba)
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                name=f"{model_name} (AUC={auc_score:.3f})",
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        except Exception:
            pass

    # Random classifier baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random Classifier",
        line=dict(color="gray", width=1, dash="dash")
    ))

    fig.update_layout(
        title_text="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate (Recall)",
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02]),
        height=500,
        margin=dict(t=60, b=60)
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_precision_recall_curves(y_test, proba_dict: dict):
    """Overlay Precision-Recall curves for all models."""
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    baseline_recall = y_test.mean()

    for i, (model_name, y_proba) in enumerate(proba_dict.items()):
        if y_proba is None:
            continue
        try:
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            ap = average_precision_score(y_test, y_proba)
            fig.add_trace(go.Scatter(
                x=recall, y=precision,
                mode="lines",
                name=f"{model_name} (AP={ap:.3f})",
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        except Exception:
            pass

    fig.add_hline(
        y=baseline_recall,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Baseline ({baseline_recall:.3f})",
        annotation_position="bottom right"
    )
    fig.update_layout(
        title_text="Precision–Recall Curves — All Models",
        xaxis_title="Recall (Sensitivity)",
        yaxis_title="Precision",
        legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02]),
        height=500,
        margin=dict(t=60, b=60)
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_confusion_matrix_churn(y_test, y_pred, model_name: str):
    """Annotated heatmap confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Retained (0)", "Churned (1)"]

    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    text = [[f"{cm[r][c]:,}<br>({cm_pct[r][c]:.1f}%)" for c in range(2)] for r in range(2)]

    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        text=text,
        texttemplate="%{text}",
        colorscale=[[0, "#1e293b"], [0.5, "#3b82f6"], [1.0, "#ef4444"]],
        showscale=False
    ))
    fig.update_layout(
        title_text=f"Confusion Matrix — {model_name}",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        width=420,
        height=380,
        margin=dict(t=60)
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_feature_importance_churn(model, feature_names: list[str], model_name: str, top_n: int = 20):
    """Feature importance bar chart for tree-based or linear models."""
    importance = None
    source = ""

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        source = "Feature Importance (Gini)"
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0]) if model.coef_.ndim == 2 else np.abs(model.coef_)
        source = "|Coefficient| (Linear)"
    else:
        st.info(f"Feature importance not available for {model_name}.")
        return

    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    fi_df = fi_df.sort_values("Importance", ascending=False).head(top_n)

    fig = px.bar(
        fi_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text=fi_df["Importance"].apply(lambda v: f"{v:.4f}"),
        color="Importance",
        color_continuous_scale="Blues",
        title=f"Top {top_n} Feature Importances — {model_name} ({source})",
        labels={"Importance": source}
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        height=max(300, 22 * len(fi_df)),
        margin=dict(t=60, l=200)
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_threshold_analysis(y_test, y_proba, model_name: str):
    """
    Interactive threshold analyzer with Business Cost & ROI Simulation.

    Shows precision/recall/F1/accuracy curves, an interactive threshold
    slider, and a collapsible Business Impact Simulation section that
    re-computes KPIs and a Profit-vs-Threshold curve whenever any input
    changes.
    """
    thresholds = np.linspace(0.01, 0.99, 98)
    precisions, recalls, f1s, accuracies = [], [], [], []
    tps, fps, fns, tns = [], [], [], []

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        precisions.append(precision_score(y_test, y_pred_t, zero_division=0))
        recalls.append(recall_score(y_test, y_pred_t, zero_division=0))
        f1s.append(f1_score(y_test, y_pred_t, zero_division=0))
        accuracies.append(accuracy_score(y_test, y_pred_t))
        cm = confusion_matrix(y_test, y_pred_t, labels=[0, 1])
        tns.append(int(cm[0, 0]))
        fps.append(int(cm[0, 1]))
        fns.append(int(cm[1, 0]))
        tps.append(int(cm[1, 1]))

    # ── Sensitivity Chart ────────────────────────────────────────────────
    st.markdown(f"#### 📊 Threshold Sensitivity — {model_name}")

    # Interactive threshold slider
    sel_threshold = st.slider(
        "Classification Threshold",
        min_value=0.01,
        max_value=0.99,
        value=0.50,
        step=0.01,
        key=f"thr_slider_{model_name}",
    )

    # Find nearest pre-computed index
    idx = int(np.argmin(np.abs(thresholds - sel_threshold)))
    sel_tp = tps[idx]
    sel_fp = fps[idx]
    sel_fn = fns[idx]
    sel_tn = tns[idx]

    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(x=thresholds, y=recalls,    name="Recall (Detect Churn)", line=dict(color="#ef4444", width=2)))
    fig_sens.add_trace(go.Scatter(x=thresholds, y=precisions, name="Precision",               line=dict(color="#3b82f6", width=2)))
    fig_sens.add_trace(go.Scatter(x=thresholds, y=f1s,        name="F1 Score",                line=dict(color="#a855f7", width=2, dash="dash")))
    fig_sens.add_trace(go.Scatter(x=thresholds, y=accuracies, name="Accuracy",                line=dict(color="#22c55e", width=2, dash="dot")))
    fig_sens.add_vline(x=sel_threshold, line_dash="dash", line_color="#f59e0b",
                       annotation_text=f"Selected ({sel_threshold:.2f})",
                       annotation_font_color="#f59e0b")

    fig_sens.update_layout(
        xaxis_title="Classification Threshold",
        yaxis_title="Score",
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02]),
        height=430,
        margin=dict(t=40, b=60),
    )
    st.plotly_chart(fig_sens, use_container_width=True)
    st.caption(
        "⬆ High recall = fewer missed churners. For retention campaigns, "
        "favour higher recall even at some precision cost."
    )

    # ── Quick confusion matrix at selected threshold ─────────────────────
    with st.expander("🔢 Confusion Matrix at Selected Threshold", expanded=False):
        cm_col1, cm_col2, cm_col3, cm_col4 = st.columns(4)
        cm_col1.metric("True Positives (TP)",  f"{sel_tp:,}")
        cm_col2.metric("False Positives (FP)", f"{sel_fp:,}")
        cm_col3.metric("False Negatives (FN)", f"{sel_fn:,}")
        cm_col4.metric("True Negatives (TN)",  f"{sel_tn:,}")

    # ── Business Impact Simulation ───────────────────────────────────────
    with st.expander("💰 Business Impact Simulation", expanded=True):
        st.markdown(
            "Configure your retention programme economics to estimate the "
            "financial impact of deploying this model at the selected threshold."
        )

        bi_col1, bi_col2, bi_col3 = st.columns(3)
        with bi_col1:
            offer_cost = st.number_input(
                "Retention Offer Cost (per customer, $)",
                min_value=0,
                value=500,
                step=50,
                key=f"offer_cost_{model_name}",
                help="Cost of the incentive / offer sent to each flagged customer.",
            )
        with bi_col2:
            customer_value = st.number_input(
                "Estimated Customer Value Saved (per churn prevented, $)",
                min_value=0,
                value=10_000,
                step=500,
                key=f"cust_value_{model_name}",
                help="Expected revenue / lifetime value retained for each churner "
                     "successfully kept.",
            )
        with bi_col3:
            success_rate = st.slider(
                "Retention Success Rate (%)",
                min_value=0,
                max_value=100,
                value=100,
                step=1,
                key=f"success_rate_{model_name}",
                help="Percentage of correctly identified churners (TP) who actually "
                     "accept the retention offer.",
            ) / 100.0

        # ── KPIs at selected threshold ───────────────────────────────────
        rev_saved     = sel_tp * customer_value * success_rate
        campaign_cost = (sel_tp + sel_fp) * offer_cost
        net_profit    = rev_saved - campaign_cost
        targeted      = sel_tp + sel_fp

        st.markdown("##### KPIs at Selected Threshold")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("💵 Revenue Saved",       f"${rev_saved:,.0f}")
        k2.metric("📤 Campaign Cost",        f"${campaign_cost:,.0f}")
        k3.metric(
            "📈 Net Profit",
            f"${net_profit:,.0f}",
            delta=f"${net_profit:,.0f}",
            delta_color="normal",
        )
        k4.metric("🎯 Customers Targeted",  f"{targeted:,}")

        # ── Profit-vs-Threshold curve ────────────────────────────────────
        st.markdown("##### Profit vs Threshold")

        profits = [
            (tp * customer_value * success_rate) - ((tp + fp) * offer_cost)
            for tp, fp in zip(tps, fps)
        ]

        best_idx    = int(np.argmax(profits))
        best_thr    = float(thresholds[best_idx])
        best_profit = profits[best_idx]

        fig_profit = go.Figure()
        fig_profit.add_trace(go.Scatter(
            x=thresholds,
            y=profits,
            name="Net Profit",
            fill="tozeroy",
            fillcolor="rgba(34,197,94,0.12)",
            line=dict(color="#22c55e", width=2.5),
        ))
        fig_profit.add_vline(
            x=sel_threshold, line_dash="dash", line_color="#f59e0b",
            annotation_text=f"Selected ({sel_threshold:.2f})",
            annotation_font_color="#f59e0b",
        )
        fig_profit.add_vline(
            x=best_thr, line_dash="dot", line_color="#a855f7",
            annotation_text=f"Optimal ({best_thr:.2f})",
            annotation_position="top left",
            annotation_font_color="#a855f7",
        )
        fig_profit.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
        fig_profit.update_layout(
            xaxis_title="Classification Threshold",
            yaxis_title="Net Profit ($)",
            height=380,
            margin=dict(t=30, b=50),
            legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_profit, use_container_width=True)

        # Optimal threshold callout
        profit_col1, profit_col2 = st.columns(2)
        profit_col1.info(
            f"**Profit-maximising threshold:** {best_thr:.2f}  \n"
            f"**Expected net profit at optimum:** ${best_profit:,.0f}"
        )
        profit_col2.caption(
            "The **optimal threshold** (purple dashed line) maximises net profit "
            "given the business parameters above. Lowering the threshold captures "
            "more churners but increases false-positive campaign spend."
        )


def plot_churn_metrics_comparison(metrics_dict: dict):
    """Grouped bar chart comparing churn-specific metrics across models."""
    churn_metrics = ["roc_auc", "f1", "precision", "recall", "balanced_accuracy", "avg_precision"]
    metric_labels = {
        "roc_auc": "ROC-AUC",
        "f1": "F1",
        "precision": "Precision",
        "recall": "Recall",
        "balanced_accuracy": "Balanced Acc.",
        "avg_precision": "Avg Precision"
    }

    rows = []
    for model, m in metrics_dict.items():
        for mk, ml in metric_labels.items():
            if mk in m:
                rows.append({"Model": model, "Metric": ml, "Score": m[mk]})

    if not rows:
        st.info("No churn metrics available.")
        return

    df_plot = pd.DataFrame(rows)
    fig = px.bar(
        df_plot,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        text=df_plot["Score"].apply(lambda v: f"{v:.3f}"),
        title="Churn Model Performance Comparison",
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_traces(textposition="outside", textfont_size=10)
    fig.update_layout(
        yaxis=dict(range=[0, 1.15]),
        height=450,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=60)
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────

def preprocess_for_churn(df: pd.DataFrame, churn_col: str, test_size: float = 0.2):
    """
    Encode features, impute missing values, scale/encode, split, and return
    (X_train, X_test, y_train, y_test, feature_names).

    Missing value strategy
    ----------------------
    - Numeric  : median imputation  (robust to outliers)
    - Categorical / bool : most-frequent imputation

    Imputation is fitted on the training split only and applied to both
    splits, preventing any leakage.  Because every NaN is filled before
    resampling, SMOTE / ADASYN receive fully-dense arrays.
    """
    y = encode_churn_target(df[churn_col])
    X = df.drop(columns=[churn_col]).copy()

    # Drop columns that are clearly IDs / dates with too many uniques
    cols_to_drop = []
    for col in X.columns:
        if X[col].dtype == object and X[col].nunique() > 100:
            cols_to_drop.append(col)
        elif "id" in col.lower() or "date" in col.lower():
            cols_to_drop.append(col)
    X.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    transformers = []
    if numeric_cols:
        # Pipeline: median impute → standard scale
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
        transformers.append(("num", numeric_pipeline, numeric_cols))

    if categorical_cols:
        # Pipeline: most-frequent impute → one-hot encode
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    if not transformers:
        raise ValueError("No usable features found after preprocessing.")

    preprocessor = ColumnTransformer(transformers)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Fit on train only → transform both (no leakage, NaNs fully resolved)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test  = preprocessor.transform(X_test_raw)

    # Build feature names
    feature_names = list(numeric_cols)
    if categorical_cols:
        ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        feature_names += list(ohe.get_feature_names_out(categorical_cols))

    return X_train, X_test, y_train, y_test, feature_names


def apply_imbalance_handling(X_train, y_train, method: str):
    """
    Apply imbalance correction to the training set.
    method: 'none', 'smote', 'adasyn', 'undersample', 'class_weight'
    Returns (X_resampled, y_resampled, class_weight_param)
    """
    class_weight_param = None

    if method == "class_weight":
        class_weight_param = "balanced"
        return X_train, y_train, class_weight_param

    if not IMBLEARN_AVAILABLE or method == "none":
        return X_train, y_train, None

    try:
        if method == "smote":
            sampler = SMOTE(random_state=42)
        elif method == "adasyn":
            sampler = ADASYN(random_state=42)
        elif method == "undersample":
            sampler = RandomUnderSampler(random_state=42)
        else:
            return X_train, y_train, None

        X_res, y_res = sampler.fit_resample(X_train, y_train)
        return X_res, y_res, None
    except Exception as e:
        st.warning(f"Imbalance sampler failed ({e}). Proceeding without resampling.")
        return X_train, y_train, None


# ─────────────────────────────────────────────────────────────
# MODEL DEFINITIONS & TRAINING
# ─────────────────────────────────────────────────────────────

def get_churn_models(class_weight=None, n_estimators: int = 200, max_depth=None, svm_c: float = 1.0):
    """
    Return dict of classification models tuned for churn detection.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=1.0, class_weight=class_weight, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            class_weight=class_weight, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth else 4,
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=42
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "AdaBoost": AdaBoostClassifier(n_estimators=n_estimators, random_state=42),
    }

    if XGBOOST_AVAILABLE:
        scale_pos_weight = None
        models["XGBoost"] = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth else 6,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0
        )

    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth else -1,
            class_weight=class_weight,
            random_state=42,
            verbose=-1
        )

    return models


def train_churn_models(
    X_train, X_test, y_train, y_test,
    selected_model_names: list[str],
    class_weight: str | None,
    n_estimators: int,
    max_depth,
    svm_c: float,
    cv_folds: int,
    optimize_metric: str,
    progress_callback=None
) -> tuple[dict, dict, dict, dict, str]:
    """
    Train selected churn models and return metrics, probabilities, and trained models.

    Returns:
        metrics_dict, proba_dict, pred_dict, trained_models, best_model_name
    """
    all_models = get_churn_models(class_weight, n_estimators, max_depth, svm_c)
    models_to_train = {k: v for k, v in all_models.items() if k in selected_model_names}

    if not models_to_train:
        raise ValueError("No valid models selected.")

    metrics_dict = {}
    proba_dict = {}
    pred_dict = {}
    trained_models = {}

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    metric_fns = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy"
    }

    cv_metric = metric_fns.get(optimize_metric, "roc_auc")

    for idx, (m_name, model) in enumerate(models_to_train.items()):
        if progress_callback:
            progress_callback(idx, len(models_to_train), m_name)

        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Probabilities for ROC/PR
            y_proba = None
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            elif hasattr(model, "decision_function"):
                raw = model.decision_function(X_test)
                y_proba = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

            # CV score
            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=skf,
                                            scoring=cv_metric, n_jobs=-1)
                cv_mean = float(cv_scores.mean())
                cv_std = float(cv_scores.std())
            except Exception:
                cv_mean, cv_std = np.nan, np.nan

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "mcc": matthews_corrcoef(y_test, y_pred),
                "cv_mean": cv_mean,
                "cv_std": cv_std
            }
            if y_proba is not None:
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
                metrics["avg_precision"] = average_precision_score(y_test, y_proba)
            else:
                metrics["roc_auc"] = np.nan
                metrics["avg_precision"] = np.nan

            metrics_dict[m_name] = metrics
            proba_dict[m_name] = y_proba
            pred_dict[m_name] = y_pred
            trained_models[m_name] = model

        except Exception as e:
            st.warning(f"Model {m_name} failed: {e}")
            continue

    if progress_callback:
        progress_callback(len(models_to_train), len(models_to_train), None)

    if not metrics_dict:
        raise ValueError("All models failed to train.")

    # Find best model by chosen metric
    metric_key_map = {
        "roc_auc": "roc_auc", "f1": "f1", "precision": "precision",
        "recall": "recall", "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy", "avg_precision": "avg_precision"
    }
    sort_key = metric_key_map.get(optimize_metric, "roc_auc")
    best_model_name = max(
        metrics_dict,
        key=lambda m: metrics_dict[m].get(sort_key, 0) if not np.isnan(metrics_dict[m].get(sort_key, 0)) else 0
    )

    return metrics_dict, proba_dict, pred_dict, trained_models, best_model_name


# ─────────────────────────────────────────────────────────────
# CONFIGURATION UI
# ─────────────────────────────────────────────────────────────

_NOVICE_DEFAULTS = {
    "selected_models": ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"],
    "imbalance_method": "smote",
    "test_size": 0.2,
    "optimize_metric": "roc_auc",
    "n_estimators": 200,
    "max_depth": None,
    "svm_c": 1.0,
    "cv_folds": 5
}


def _available_default_models():
    """Filter default novice models to only those actually available."""
    all_available = list(get_churn_models().keys())
    return [m for m in _NOVICE_DEFAULTS["selected_models"] if m in all_available]


def render_churn_config() -> dict:
    """
    Render the Expert/Novice configuration panel for customer churn training.
    Returns a config dict with all parameters.
    """
    st.markdown("---")
    st.markdown("### ⚙️ Configuration")

    all_models = list(get_churn_models().keys())

    mode = st.radio(
        "User Mode",
        ["🟢 Novice (Auto-configured)", "🔧 Expert (Manual control)"],
        horizontal=True,
        help="Novice mode uses best-practice defaults. Expert mode exposes all parameters."
    )
    novice = mode.startswith("🟢")

    config = {}

    if novice:
        # ── NOVICE: show what's auto-selected, no editable controls
        defaults = _NOVICE_DEFAULTS.copy()
        defaults["selected_models"] = [m for m in defaults["selected_models"] if m in all_models]
        if not defaults["selected_models"]:
            defaults["selected_models"] = all_models[:3]

        col1, col2, col3, col4 = st.columns(4)
        col1.info(f"**Models:** {', '.join(defaults['selected_models'])}")
        col2.info(f"**Imbalance:** {defaults['imbalance_method'].upper()}")
        col3.info(f"**Metric:** {defaults['optimize_metric'].upper()}")
        col4.info(f"**Test size:** {int(defaults['test_size']*100)}%")
        config.update(defaults)

    else:
        # ── EXPERT: full controls
        st.markdown("#### Model Selection")
        default_selected = [m for m in _NOVICE_DEFAULTS["selected_models"] if m in all_models]
        config["selected_models"] = st.multiselect(
            "Models to train",
            all_models,
            default=default_selected or all_models[:3],
            help="Select one or more classification models for churn detection"
        )

        st.markdown("#### Class Imbalance Handling")
        imbalance_options = ["none", "class_weight"]
        imbalance_labels = ["None", "Class Weight"]

        if IMBLEARN_AVAILABLE:
            imbalance_options += ["smote", "adasyn", "undersample"]
            imbalance_labels += ["SMOTE", "ADASYN", "Random Undersample"]

        config["imbalance_method"] = st.selectbox(
            "Imbalance correction",
            imbalance_options,
            index=imbalance_options.index("smote") if "smote" in imbalance_options else 1,
            format_func=lambda x: dict(zip(imbalance_options, imbalance_labels)).get(x, x),
            help="Method to handle class imbalance in the training set"
        )

        st.markdown("#### Training Parameters")
        col1, col2 = st.columns(2)
        with col1:
            config["test_size"] = st.slider(
                "Test set fraction", 0.1, 0.4, 0.2, step=0.05,
                help="Fraction of data reserved for evaluation"
            )
            config["n_estimators"] = st.slider(
                "Estimators (trees)", 50, 500, 200, step=50,
                help="Number of trees for ensemble models"
            )
        with col2:
            cv_folds_str = st.selectbox(
                "Cross-validation folds", ["3", "5", "10"], index=1,
                help="StratifiedKFold folds for CV score estimation"
            )
            config["cv_folds"] = int(cv_folds_str)

            max_depth_opt = st.selectbox(
                "Max tree depth",
                ["None (unlimited)"] + list(range(2, 21)),
                index=0,
                help="Depth limit for tree-based models"
            )
            config["max_depth"] = None if max_depth_opt == "None (unlimited)" else int(max_depth_opt)

        st.markdown("#### Evaluation Metric")
        metric_options = {
            "roc_auc": "ROC-AUC (best overall)",
            "f1": "F1 Score (balanced)",
            "recall": "Recall (minimize missed churners)",
            "precision": "Precision (minimize false alarms)",
            "balanced_accuracy": "Balanced Accuracy",
            "avg_precision": "Average Precision (PR-AUC)"
        }
        config["optimize_metric"] = st.selectbox(
            "Optimization metric",
            list(metric_options.keys()),
            format_func=lambda k: metric_options[k],
            index=0,
            help="Metric used for CV scoring and model ranking"
        )

        config["svm_c"] = 1.0  # SVM not a churn default but keep param

    return config


# ─────────────────────────────────────────────────────────────
# RESULTS SUMMARY TABLE
# ─────────────────────────────────────────────────────────────

def build_churn_metrics_dataframe(metrics_dict: dict) -> pd.DataFrame:
    """
    Build a clean summary DataFrame of churn metrics for display.
    """
    rows = []
    for model, m in metrics_dict.items():
        rows.append({
            "Model": model,
            "ROC-AUC": f"{m.get('roc_auc', np.nan):.4f}" if not np.isnan(m.get("roc_auc", np.nan)) else "N/A",
            "Avg Precision": f"{m.get('avg_precision', np.nan):.4f}" if not np.isnan(m.get("avg_precision", np.nan)) else "N/A",
            "F1": f"{m.get('f1', 0):.4f}",
            "Precision": f"{m.get('precision', 0):.4f}",
            "Recall": f"{m.get('recall', 0):.4f}",
            "Balanced Acc.": f"{m.get('balanced_accuracy', 0):.4f}",
            "MCC": f"{m.get('mcc', 0):.4f}",
            "CV Score": f"{m.get('cv_mean', np.nan):.4f} ± {m.get('cv_std', np.nan):.4f}"
            if not np.isnan(m.get("cv_mean", np.nan)) else "N/A"
        })
    return pd.DataFrame(rows)


# ============================================================
# MODEL PERSISTENCE & REGISTRY
# ============================================================

def compute_dataset_hash(df: pd.DataFrame) -> str:
    """Return an MD5 hex-digest fingerprint of the DataFrame contents."""
    raw = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.md5(raw).hexdigest()


def create_model_bundle(
    model,
    model_name: str,
    feature_names: list,
    metrics: dict,
    config: dict,
    df_hash: str,
) -> dict:
    """Package a trained model with full provenance metadata."""
    # Serialise only JSON-safe subset of metrics
    safe_metrics = {
        k: (float(v) if isinstance(v, (float, np.floating)) else
            int(v) if isinstance(v, (int, np.integer)) else
            str(v))
        for k, v in metrics.items()
    }
    # Serialise config (drop non-serialisable values gracefully)
    safe_config = {}
    for k, v in config.items():
        try:
            json.dumps(v)
            safe_config[k] = v
        except (TypeError, ValueError):
            safe_config[k] = str(v)

    return {
        "model": model,
        "model_name": model_name,
        "feature_names": list(feature_names),
        "metrics": safe_metrics,
        "config": safe_config,
        "dataset_hash": df_hash,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "module_version": "1.0.0",
    }


def create_model_download_bytes(bundle: dict) -> bytes:
    """Serialise a model bundle to bytes using joblib (for st.download_button)."""
    buf = io.BytesIO()
    joblib.dump(bundle, buf)
    return buf.getvalue()


def render_model_save_section(
    trained_models: dict,
    metrics_dict: dict,
    feature_names: list,
    best_model_name: str,
    config: dict,
    df_hash: str,
    key_prefix: str = "churn",
) -> None:
    """
    Streamlit expander with:
    - Model selector (default = best model)
    - Key metric preview
    - Custom filename input
    - Download button (.joblib bundle with metadata)
    - Inline JSON config viewer
    """
    with st.expander("💾 Save / Export Trained Model", expanded=False):
        st.markdown(
            "Download any trained model as a `.joblib` bundle that includes the "
            "fitted estimator, feature names, metrics, training config, and a "
            "dataset fingerprint for traceability."
        )

        model_names = list(trained_models.keys())
        default_idx = model_names.index(best_model_name) if best_model_name in model_names else 0

        selected = st.selectbox(
            "Select model to export",
            model_names,
            index=default_idx,
            key=f"{key_prefix}_save_model_selector",
        )

        # Metric preview
        m = metrics_dict.get(selected, {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ROC-AUC", f"{m.get('roc_auc', 0):.4f}")
        c2.metric("F1", f"{m.get('f1', 0):.4f}")
        c3.metric("Recall", f"{m.get('recall', 0):.4f}")
        c4.metric("Precision", f"{m.get('precision', 0):.4f}")

        # Filename
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        default_name = f"churn_{selected.lower().replace(' ', '_')}_{ts}.joblib"
        filename = st.text_input(
            "Filename",
            value=default_name,
            key=f"{key_prefix}_save_filename",
        )
        if not filename.endswith(".joblib"):
            filename += ".joblib"

        # Build bundle & download
        bundle = create_model_bundle(
            trained_models[selected],
            selected,
            feature_names,
            m,
            config,
            df_hash,
        )
        bundle_bytes = create_model_download_bytes(bundle)

        st.download_button(
            label=f"⬇️ Download  {selected}  ({len(bundle_bytes)/1024:.1f} KB)",
            data=bundle_bytes,
            file_name=filename,
            mime="application/octet-stream",
            key=f"{key_prefix}_download_btn",
        )

        # JSON config viewer
        with st.expander("View saved metadata (JSON)", expanded=False):
            meta = {k: v for k, v in bundle.items() if k != "model"}
            st.json(meta)


# ============================================================
# DATA DRIFT DETECTION  (KS + PSI)
# ============================================================

def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index between two 1-D arrays.
    PSI < 0.1  → stable  |  0.1–0.2 → minor drift  |  > 0.2 → major drift
    """
    eps = 1e-8
    # Build shared bin edges from combined range
    combined = np.concatenate([expected, actual])
    breakpoints = np.linspace(combined.min(), combined.max(), bins + 1)
    breakpoints[-1] += eps  # ensure max falls in last bin

    exp_counts, _ = np.histogram(expected, bins=breakpoints)
    act_counts, _ = np.histogram(actual, bins=breakpoints)

    exp_pct = (exp_counts + eps) / (len(expected) + eps * bins)
    act_pct = (act_counts + eps) / (len(actual) + eps * bins)

    psi = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))
    return round(abs(psi), 6)


def compute_drift_report(
    df_reference: pd.DataFrame,
    df_new: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare distributions of all numeric columns that appear in both DataFrames.

    Returns a DataFrame with columns:
        Feature, KS_Stat, KS_PValue, PSI, Drift_Level
    where Drift_Level ∈ {Stable, Minor, Major}.
    """
    numeric_cols = [
        c for c in df_reference.select_dtypes(include=[np.number]).columns
        if c in df_new.columns
    ]

    rows = []
    for col in numeric_cols:
        ref_vals = df_reference[col].dropna().values.astype(float)
        new_vals = df_new[col].dropna().values.astype(float)

        if len(ref_vals) < 5 or len(new_vals) < 5:
            continue

        ks_stat, ks_pval = _scipy_stats.ks_2samp(ref_vals, new_vals)
        psi = compute_psi(ref_vals, new_vals)

        if psi > 0.2 or ks_stat > 0.3:
            level = "🔴 Major"
        elif psi > 0.1 or ks_stat > 0.15:
            level = "🟡 Minor"
        else:
            level = "🟢 Stable"

        rows.append({
            "Feature": col,
            "KS Stat": round(ks_stat, 4),
            "KS p-value": round(ks_pval, 4),
            "PSI": psi,
            "Drift Level": level,
        })

    if not rows:
        return pd.DataFrame(columns=["Feature", "KS Stat", "KS p-value", "PSI", "Drift Level"])
    return pd.DataFrame(rows).sort_values("PSI", ascending=False).reset_index(drop=True)


def render_drift_detection_section(
    df_reference: pd.DataFrame | None = None,
    key_prefix: str = "drift",
) -> None:
    """
    Streamlit section for data-drift detection.

    Users upload a *new* dataset; the reference is the currently loaded training
    dataset.  Shows a colour-coded drift table + PSI bar chart.
    """
    with st.expander("📡 Data Drift Detection", expanded=False):
        st.markdown(
            "Upload a new batch of data to compare its feature distributions "
            "against the reference (training) dataset using the **KS test** "
            "and **Population Stability Index (PSI)**."
        )

        col_guide1, col_guide2, col_guide3 = st.columns(3)
        col_guide1.info("**🟢 Stable** – PSI < 0.10 & KS < 0.15")
        col_guide2.warning("**🟡 Minor** – PSI 0.10–0.20 or KS 0.15–0.30")
        col_guide3.error("**🔴 Major** – PSI > 0.20 or KS > 0.30")

        new_file = st.file_uploader(
            "Upload new dataset (CSV) for drift comparison",
            type=["csv"],
            key=f"{key_prefix}_upload",
        )

        if new_file is None:
            st.info("Upload a CSV to run drift analysis.")
            return

        try:
            df_new = pd.read_csv(new_file)
        except Exception as exc:
            st.error(f"Could not parse file: {exc}")
            return

        if df_reference is None:
            st.warning("No reference dataset available. Please train models first.")
            return

        # Compute drift
        with st.spinner("Computing drift statistics…"):
            drift_df = compute_drift_report(df_reference, df_new)

        if drift_df.empty:
            st.warning("No shared numeric columns found between the two datasets.")
            return

        st.markdown(f"**{len(drift_df)} features compared** (numeric columns present in both files)")

        # Summary badges
        n_major = (drift_df["Drift Level"] == "🔴 Major").sum()
        n_minor = (drift_df["Drift Level"] == "🟡 Minor").sum()
        n_stable = (drift_df["Drift Level"] == "🟢 Stable").sum()

        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("🟢 Stable", n_stable)
        bc2.metric("🟡 Minor Drift", n_minor)
        bc3.metric("🔴 Major Drift", n_major)

        if n_major > 0:
            flagged = drift_df[drift_df["Drift Level"] == "🔴 Major"]["Feature"].tolist()
            st.error(
                f"⚠️ **Major drift detected** in: {', '.join(flagged)}\n\n"
                "Consider retraining with the new data or investigating data pipeline issues."
            )
        elif n_minor > 0:
            st.warning("⚠️ Minor drift detected in some features. Monitor model performance closely.")
        else:
            st.success("✅ All features appear stable — no significant drift detected.")

        # Drift table
        st.markdown("#### Drift Report")
        st.dataframe(drift_df, use_container_width=True, hide_index=True)

        # PSI bar chart
        st.markdown("#### PSI by Feature")
        psi_fig = go.Figure()
        colors = [
            "#e74c3c" if l == "🔴 Major" else
            "#f39c12" if l == "🟡 Minor" else
            "#27ae60"
            for l in drift_df["Drift Level"]
        ]
        psi_fig.add_trace(go.Bar(
            x=drift_df["Feature"],
            y=drift_df["PSI"],
            marker_color=colors,
            text=[f"{v:.4f}" for v in drift_df["PSI"]],
            textposition="outside",
        ))
        psi_fig.add_hline(y=0.1, line_dash="dash", line_color="#f39c12",
                          annotation_text="Minor threshold (0.10)")
        psi_fig.add_hline(y=0.2, line_dash="dash", line_color="#e74c3c",
                          annotation_text="Major threshold (0.20)")
        psi_fig.update_layout(
            title="Population Stability Index per Feature",
            xaxis_title="Feature",
            yaxis_title="PSI",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_tickangle=-45,
            height=420,
        )
        st.plotly_chart(psi_fig, use_container_width=True)

        # KS p-value chart
        st.markdown("#### KS p-value by Feature")
        ks_fig = go.Figure()
        ks_colors = [
            "#e74c3c" if l == "🔴 Major" else
            "#f39c12" if l == "🟡 Minor" else
            "#27ae60"
            for l in drift_df["Drift Level"]
        ]
        ks_fig.add_trace(go.Bar(
            x=drift_df["Feature"],
            y=drift_df["KS Stat"],
            marker_color=ks_colors,
            text=[f"{v:.4f}" for v in drift_df["KS Stat"]],
            textposition="outside",
        ))
        ks_fig.add_hline(y=0.15, line_dash="dash", line_color="#f39c12",
                         annotation_text="Minor threshold (0.15)")
        ks_fig.add_hline(y=0.30, line_dash="dash", line_color="#e74c3c",
                         annotation_text="Major threshold (0.30)")
        ks_fig.update_layout(
            title="KS Statistic per Feature",
            xaxis_title="Feature",
            yaxis_title="KS Statistic",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_tickangle=-45,
            height=420,
        )
        st.plotly_chart(ks_fig, use_container_width=True)
