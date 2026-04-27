# ==============================
# ./core/chart_generator.py
# ==============================
"""
Chart Generator Module
Creates interactive visualizations using Plotly
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st


def plot_performance_comparison(performance_df, use_cv=True):
    """
    Create bar charts comparing model performance
    
    Args:
        performance_df: DataFrame with model performance metrics
        use_cv: whether to show CV scores
    """
    # Convert string scores back to float for plotting
    perf_df = performance_df.copy()
    perf_df['Test Score Numeric'] = perf_df['Test Score'].astype(float)
    
    # Test Score Comparison
    fig_test = px.bar(
        perf_df,
        x="Model",
        y="Test Score Numeric",
        color="Model",
        title="Test Performance by Model",
        text="Test Score"
    )
    fig_test.update_traces(texttemplate='%{text}', textposition='outside')
    fig_test.update_yaxes(title_text="Test Score")
    st.plotly_chart(fig_test, use_container_width=True)
    
    # CV Score Comparison
    if use_cv and 'CV Score' in perf_df.columns:
        # Filter out N/A values for plotting
        cv_df = perf_df[perf_df['CV Score'] != 'N/A'].copy()
        if not cv_df.empty:
            cv_df['CV Score Numeric'] = cv_df['CV Score'].astype(float)
            
            fig_cv = px.bar(
                cv_df,
                x="Model",
                y="CV Score Numeric",
                color="Model",
                title="Cross-Validation Performance by Model",
                text="CV Score"
            )
            fig_cv.update_traces(texttemplate='%{text}', textposition='outside')
            fig_cv.update_yaxes(title_text="CV Score")
            st.plotly_chart(fig_cv, use_container_width=True)


def plot_model_score_comparison(performance, title="Model Leaderboard (Test vs CV)"):
    """
    Plot test and CV scores as a dot leaderboard to reduce bar-chart repetition.

    Args:
        performance: dict mapping model name -> (test_score, cv_score) or scalar score
        title: chart title
    """
    rows = []
    for model_name, value in performance.items():
        if isinstance(value, tuple):
            test_score = float(value[0])
            cv_score = float(value[1]) if not pd.isna(value[1]) else np.nan
        else:
            test_score = float(value)
            cv_score = np.nan

        rows.append(
            {
                "Model": str(model_name),
                "Test Score": test_score,
                "CV Score": cv_score,
            }
        )

    if not rows:
        st.info("No performance values available for comparison.")
        return

    plot_df = pd.DataFrame(rows).sort_values("Test Score", ascending=True)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["Test Score"],
            y=plot_df["Model"],
            mode="markers+text",
            marker=dict(size=13, color="#1f6f78", line=dict(width=1, color="#10363a")),
            text=[f"{v:.4f}" for v in plot_df["Test Score"]],
            textposition="middle right",
            name="Test Score",
        )
    )

    if plot_df["CV Score"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=plot_df["CV Score"],
                y=plot_df["Model"],
                mode="markers",
                marker=dict(size=9, color="#e76f51", symbol="diamond"),
                name="CV Score",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Score",
        yaxis_title="Model",
        height=max(320, 66 + 48 * len(plot_df)),
        legend=dict(orientation="h", y=-0.20, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_confusion_matrix_generic(y_true, y_pred, class_labels=None, title="Confusion Matrix"):
    """
    Plot a generic confusion matrix heatmap for binary or multi-class tasks.

    Args:
        y_true: true labels
        y_pred: predicted labels
        class_labels: optional explicit class label ordering
        title: chart title
    """
    from sklearn.metrics import confusion_matrix

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if y_true_arr.size == 0 or y_pred_arr.size == 0:
        st.info("Confusion matrix unavailable: empty predictions.")
        return

    if class_labels is None:
        labels = sorted(pd.unique(np.concatenate([y_true_arr, y_pred_arr])))
    else:
        labels = list(class_labels)

    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    cm = np.asarray(cm)

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_pct = (cm / row_sums) * 100.0

    text = [
        [f"{int(cm[r, c]):,}<br>({cm_pct[r, c]:.1f}%)" for c in range(cm.shape[1])]
        for r in range(cm.shape[0])
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=[str(l) for l in labels],
            y=[str(l) for l in labels],
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Count"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=max(340, 120 + 55 * len(labels)),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_roc_curve_binary(y_true, y_proba, model_name="Model"):
    """
    Plot ROC curve for binary classification.

    Args:
        y_true: binary true labels
        y_proba: positive class probabilities
        model_name: model label
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba)

    if len(np.unique(y_true_arr)) != 2:
        st.info("ROC curve requires binary targets.")
        return

    fpr, tpr, _ = roc_curve(y_true_arr, y_proba_arr)
    auc_score = float(roc_auc_score(y_true_arr, y_proba_arr))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            line=dict(color="#1f6f78", width=3),
            name=f"{model_name} (AUC={auc_score:.3f})",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color="#6b7280", width=1, dash="dash"),
            name="Random baseline",
        )
    )
    fig.update_layout(
        title=f"ROC Curve - {model_name}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02]),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_precision_recall_curve_binary(y_true, y_proba, model_name="Model"):
    """
    Plot precision-recall curve for binary classification.

    Args:
        y_true: binary true labels
        y_proba: positive class probabilities
        model_name: model label
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve

    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba)

    if len(np.unique(y_true_arr)) != 2:
        st.info("Precision-Recall curve requires binary targets.")
        return

    precision, recall, _ = precision_recall_curve(y_true_arr, y_proba_arr)
    ap = float(average_precision_score(y_true_arr, y_proba_arr))
    baseline = float(np.mean(y_true_arr))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            line=dict(color="#e76f51", width=3),
            name=f"{model_name} (AP={ap:.3f})",
        )
    )
    fig.add_hline(
        y=baseline,
        line_dash="dash",
        line_color="#6b7280",
        annotation_text=f"Baseline {baseline:.3f}",
    )
    fig.update_layout(
        title=f"Precision-Recall Curve - {model_name}",
        xaxis_title="Recall",
        yaxis_title="Precision",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02]),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_threshold_metric_curves(y_true, y_proba, model_name="Model"):
    """
    Plot threshold vs precision/recall/F1/accuracy curves for binary classification.

    Args:
        y_true: binary true labels
        y_proba: positive class probabilities
        model_name: model label
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    y_true_arr = np.asarray(y_true)
    y_proba_arr = np.asarray(y_proba)

    if len(np.unique(y_true_arr)) != 2:
        st.info("Threshold curves require binary targets.")
        return

    thresholds = np.linspace(0.01, 0.99, 99)
    precision_vals = []
    recall_vals = []
    f1_vals = []
    accuracy_vals = []

    for threshold in thresholds:
        pred = (y_proba_arr >= threshold).astype(int)
        precision_vals.append(precision_score(y_true_arr, pred, zero_division=0))
        recall_vals.append(recall_score(y_true_arr, pred, zero_division=0))
        f1_vals.append(f1_score(y_true_arr, pred, zero_division=0))
        accuracy_vals.append(accuracy_score(y_true_arr, pred))

    best_idx = int(np.argmax(f1_vals))
    best_thr = float(thresholds[best_idx])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=thresholds, y=precision_vals, name="Precision", line=dict(width=2, color="#3b82f6")))
    fig.add_trace(go.Scatter(x=thresholds, y=recall_vals, name="Recall", line=dict(width=2, color="#ef4444")))
    fig.add_trace(go.Scatter(x=thresholds, y=f1_vals, name="F1", line=dict(width=2, color="#8b5cf6")))
    fig.add_trace(go.Scatter(x=thresholds, y=accuracy_vals, name="Accuracy", line=dict(width=2, color="#22c55e")))
    fig.add_vline(
        x=best_thr,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text=f"Best F1 threshold: {best_thr:.2f}",
        annotation_font_color="#f59e0b",
    )
    fig.update_layout(
        title=f"Threshold vs Metrics - {model_name}",
        xaxis_title="Threshold",
        yaxis_title="Score",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1.02]),
        legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
        height=430,
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_probability_distribution(y_proba, title="Predicted Probability Distribution"):
    """
    Plot distribution of predicted probabilities.

    Args:
        y_proba: probability scores
        title: chart title
    """
    y_proba_arr = np.asarray(y_proba, dtype=float)
    if y_proba_arr.size == 0:
        st.info("No probability scores available.")
        return

    fig = px.histogram(
        x=y_proba_arr,
        nbins=35,
        title=title,
        labels={"x": "Predicted positive probability", "y": "Count"},
        color_discrete_sequence=["#1f6f78"],
    )
    fig.update_layout(height=360)
    st.plotly_chart(fig, use_container_width=True)


def plot_feature_correlation_lollipop(correlation_df, title="Top Correlation Drivers"):
    """
    Plot feature-target correlations in a lollipop-style chart.

    Args:
        correlation_df: DataFrame with feature/correlation/abs_correlation columns
        title: chart title
    """
    if correlation_df is None or correlation_df.empty:
        st.info("No correlation data available.")
        return

    plot_df = correlation_df.copy().sort_values("correlation", ascending=True)
    plot_df = plot_df.reset_index(drop=True)
    plot_df["position"] = np.arange(len(plot_df))

    fig = go.Figure()
    for _, row in plot_df.iterrows():
        fig.add_shape(
            type="line",
            x0=0,
            y0=row["position"],
            x1=float(row["correlation"]),
            y1=row["position"],
            line=dict(color="#9ca3af", width=1.5),
        )

    fig.add_trace(
        go.Scatter(
            x=plot_df["correlation"],
            y=plot_df["position"],
            mode="markers+text",
            marker=dict(
                size=12,
                color=plot_df["correlation"],
                colorscale="RdBu",
                cmin=-1,
                cmax=1,
                line=dict(width=1, color="#1f2937"),
                colorbar=dict(title="Correlation"),
            ),
            text=[f"{v:+.3f}" for v in plot_df["correlation"]],
            textposition="middle right",
            hovertemplate="<b>%{customdata}</b><br>Correlation: %{x:+.4f}<extra></extra>",
            customdata=plot_df["feature"],
            name="Correlation",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Correlation with target",
        yaxis=dict(
            tickmode="array",
            tickvals=plot_df["position"],
            ticktext=plot_df["feature"],
            title="Feature",
        ),
        height=max(320, 90 + 38 * len(plot_df)),
        showlegend=False,
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#6b7280")
    st.plotly_chart(fig, use_container_width=True)


def plot_feature_distribution(df, feature_name, nbins=30):
    """
    Create histogram for feature distribution
    
    Args:
        df: pandas DataFrame
        feature_name: name of feature to plot
        nbins: number of bins for histogram
    """
    fig = px.histogram(
        df,
        x=feature_name,
        nbins=nbins,
        title=f"Distribution of {feature_name}",
        color_discrete_sequence=['#636EFA']
    )
    fig.update_layout(
        xaxis_title=feature_name,
        yaxis_title="Count",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_correlation_heatmap(df, numeric_cols):
    """
    Create correlation heatmap for numeric features
    
    Args:
        df: pandas DataFrame
        numeric_cols: list of numeric column names
    """
    if len(numeric_cols) < 2:
        st.info("Need at least 2 numeric columns for correlation heatmap")
        return
    
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Feature Correlation Heatmap"
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_missing_values(df):
    """
    Create bar chart showing missing values per column
    
    Args:
        df: pandas DataFrame
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) == 0:
        st.success("No missing values in the dataset!")
        return
    
    fig = px.bar(
        x=missing.values,
        y=missing.index,
        orientation='h',
        title="Missing Values by Column",
        labels={'x': 'Count', 'y': 'Column'},
        color=missing.values,
        color_continuous_scale='Reds'
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_target_distribution(y, task_type):
    """
    Create distribution plot for target variable
    
    Args:
        y: target variable
        task_type: 'classification' or 'regression'
    """
    if task_type == "classification":
        value_counts = pd.Series(y).value_counts().sort_index()
        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            labels={'x': 'Class', 'y': 'Count'},
            title="Target Class Distribution",
            color=value_counts.values,
            color_continuous_scale='Viridis'
        )
    else:
        fig = px.histogram(
            x=y,
            nbins=30,
            labels={'x': 'Target Value', 'y': 'Count'},
            title="Target Value Distribution",
            color_discrete_sequence=['#00CC96']
        )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_data_overview(df):
    """
    Create overview visualizations for the dataset
    
    Args:
        df: pandas DataFrame
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Missing Values", f"{df.isnull().sum().sum():,}")
    with col4:
        st.metric("Duplicates", f"{df.duplicated().sum():,}")
    
    # Data types distribution
    dtype_counts = df.dtypes.value_counts()
    fig = px.pie(
        values=dtype_counts.values,
        names=dtype_counts.index.astype(str),
        title="Data Types Distribution"
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_train_vs_test_comparison(all_metrics_dict, task_type):
    """
    Create side-by-side comparison of train vs test scores
    
    Args:
        all_metrics_dict: dict containing all metrics for each model
        task_type: 'classification' or 'regression'
    """
    models = list(all_metrics_dict.keys())
    
    if task_type == "classification":
        train_scores = [all_metrics_dict[m].get('train_accuracy', 0) for m in models]
        test_scores = [all_metrics_dict[m].get('test_accuracy', 0) for m in models]
        metric_name = "Accuracy"
    else:
        train_scores = [all_metrics_dict[m].get('train_r2', 0) for m in models]
        test_scores = [all_metrics_dict[m].get('test_r2', 0) for m in models]
        metric_name = "R² Score"
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Train Score',
        x=models,
        y=train_scores,
        marker_color='lightblue',
        text=[f"{s:.4f}" for s in train_scores],
        textposition='outside'
    ))
    
    fig.add_trace(go.Bar(
        name='Test Score',
        x=models,
        y=test_scores,
        marker_color='coral',
        text=[f"{s:.4f}" for s in test_scores],
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"Train vs Test {metric_name} - Overfitting Detection",
        xaxis_title="Model",
        yaxis_title=metric_name,
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_overfitting_analysis(all_metrics_dict, task_type):
    """
    Create overfitting analysis visualization
    
    Args:
        all_metrics_dict: dict containing all metrics for each model
        task_type: 'classification' or 'regression'
    """
    models = list(all_metrics_dict.keys())
    overfit_diffs = [all_metrics_dict[m].get('overfit_difference', 0) for m in models]
    
    # Color code based on severity
    colors = []
    for diff in overfit_diffs:
        if diff > 0.1:
            colors.append('red')
        elif diff > 0.05:
            colors.append('orange')
        else:
            colors.append('green')
    
    fig = go.Figure(go.Bar(
        x=models,
        y=overfit_diffs,
        marker_color=colors,
        text=[f"{d:.4f}" for d in overfit_diffs],
        textposition='outside'
    ))
    
    # Add threshold lines
    fig.add_hline(y=0.05, line_dash="dash", line_color="orange", 
                  annotation_text="Moderate (5%)", annotation_position="right")
    fig.add_hline(y=0.1, line_dash="dash", line_color="red",
                  annotation_text="High (10%)", annotation_position="right")
    
    fig.update_layout(
        title="Overfitting Analysis (Train - Test Score)",
        xaxis_title="Model",
        yaxis_title="Score Difference",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation
    st.markdown("""
    **Interpretation:**
    - 🟢 **Green (< 5%)**: Good generalization
    - 🟠 **Orange (5-10%)**: Moderate overfitting
    - 🔴 **Red (> 10%)**: High overfitting risk
    """)


def plot_all_metrics_radar(all_metrics_dict, task_type, model_name):
    """
    Create radar chart for all metrics of a specific model
    
    Args:
        all_metrics_dict: dict containing all metrics for each model
        task_type: 'classification' or 'regression'
        model_name: name of the model to visualize
    """
    if model_name not in all_metrics_dict:
        st.warning(f"Metrics not available for {model_name}")
        return
    
    metrics = all_metrics_dict[model_name]
    
    if task_type == "classification":
        categories = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
        test_values = [
            metrics.get('test_accuracy', 0),
            metrics.get('test_f1', 0),
            metrics.get('test_precision', 0),
            metrics.get('test_recall', 0)
        ]
        train_values = [
            metrics.get('train_accuracy', 0),
            metrics.get('train_f1', 0),
            metrics.get('train_precision', 0),
            metrics.get('train_recall', 0)
        ]
        
        if 'test_roc_auc' in metrics:
            categories.append('ROC-AUC')
            test_values.append(metrics['test_roc_auc'])
            train_values.append(metrics.get('train_roc_auc', 0))
    else:
        # For regression, normalize metrics to 0-1 scale for visualization
        # R² and Adjusted R² are already in 0-1 range (higher is better)
        r2_test = max(0, min(1, metrics.get('test_r2', 0)))
        r2_train = max(0, min(1, metrics.get('train_r2', 0)))
        adj_r2_test = max(0, min(1, metrics.get('test_adjusted_r2', 0)))
        adj_r2_train = max(0, min(1, metrics.get('train_adjusted_r2', 0))) if 'train_adjusted_r2' in metrics else r2_train
        
        # Explained Variance is also in 0-1 range (higher is better)
        explained_var_test = max(0, min(1, metrics.get('test_explained_variance', 0)))
        explained_var_train = max(0, min(1, metrics.get('train_explained_variance', 0))) if 'train_explained_variance' in metrics else 0
        
        # For error metrics (MAE, RMSE, MSE, MAPE), we need to invert them
        # We'll use 1 / (1 + error) to normalize (higher is better on radar)
        mae_test = metrics.get('test_mae', 0)
        mae_train = metrics.get('train_mae', 0)
        mae_test_norm = 1 / (1 + mae_test) if mae_test >= 0 else 0
        mae_train_norm = 1 / (1 + mae_train) if mae_train >= 0 else 0
        
        rmse_test = metrics.get('test_rmse', 0)
        rmse_train = metrics.get('train_rmse', 0)
        rmse_test_norm = 1 / (1 + rmse_test) if rmse_test >= 0 else 0
        rmse_train_norm = 1 / (1 + rmse_train) if rmse_train >= 0 else 0
        
        mape_test = metrics.get('test_mape', 0)
        mape_train = metrics.get('train_mape', 0)
        # MAPE is percentage, so scale differently
        mape_test_norm = 1 / (1 + mape_test/100) if not np.isnan(mape_test) and mape_test >= 0 else 0
        mape_train_norm = 1 / (1 + mape_train/100) if not np.isnan(mape_train) and mape_train >= 0 else 0
        
        categories = ['R²', 'Adjusted R²', 'Explained Var', 'MAE\n(inverted)', 'RMSE\n(inverted)', 'MAPE\n(inverted)']
        test_values = [r2_test, adj_r2_test, explained_var_test, mae_test_norm, rmse_test_norm, mape_test_norm]
        train_values = [r2_train, adj_r2_train, explained_var_train, mae_train_norm, rmse_train_norm, mape_train_norm]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=test_values,
        theta=categories,
        fill='toself',
        name='Test',
        line_color='coral'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=train_values,
        theta=categories,
        fill='toself',
        name='Train',
        line_color='lightblue',
        opacity=0.6
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title=f"Metric Comparison: {model_name}",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation for regression metrics
    if task_type == "regression":
        st.info("📊 **Regression Radar Chart Guide:**\n"
                "- **R²**, **Adjusted R²**, **Explained Variance**: Higher is better (0-1 scale)\n"
                "- **MAE**, **RMSE**, **MAPE** (inverted): Error metrics transformed using 1/(1+error) - "
                "higher values on radar = lower actual errors = better performance\n"
                "- Values closer to the edge (1.0) indicate better performance for all metrics shown")


def plot_metrics_heatmap(all_metrics_dict, task_type):
    """
    Create heatmap of all metrics across all models
    
    Args:
        all_metrics_dict: dict containing all metrics for each model
        task_type: 'classification' or 'regression'
    """
    models = list(all_metrics_dict.keys())
    
    if task_type == "classification":
        metric_names = ['Test Accuracy', 'Test F1', 'Test Precision', 'Test Recall']
        metric_keys = ['test_accuracy', 'test_f1', 'test_precision', 'test_recall']
        
        # Add ROC-AUC if available
        if 'test_roc_auc' in all_metrics_dict[models[0]]:
            metric_names.append('Test ROC-AUC')
            metric_keys.append('test_roc_auc')
    else:
        metric_names = ['Test R²', 'Test MAE', 'Test RMSE']
        metric_keys = ['test_r2', 'test_mae', 'test_rmse']
    
    # Create matrix
    matrix = []
    for model in models:
        row = [all_metrics_dict[model].get(key, 0) for key in metric_keys]
        matrix.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=metric_names,
        y=models,
        colorscale='RdYlGn',
        text=[[f"{val:.4f}" for val in row] for row in matrix],
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Score")
    ))
    
    fig.update_layout(
        title="Metrics Heatmap",
        xaxis_title="Metrics",
        yaxis_title="Models",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ==============================
# Clustering Visualization Functions
# ==============================

def plot_regression_predictions(y_test, y_pred, model_name):
    """
    Plot actual vs predicted values for regression models
    
    Args:
        y_test: actual target values
        y_pred: predicted target values  
        model_name: name of the model
    """
    import plotly.graph_objects as go
    import numpy as _np
    from sklearn.metrics import r2_score, mean_squared_error

    # Ensure numpy arrays for numeric operations
    y_true = _np.asarray(y_test)
    y_pred_arr = _np.asarray(y_pred)

    fig = go.Figure()

    # Add scatter plot of actual vs predicted
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred_arr,
        mode='markers',
        name='Predictions',
        marker=dict(
            size=8,
            color='blue',
            opacity=0.6
        ),
        hovertemplate='<b>Actual:</b> %{x:.2f}<br><b>Predicted:</b> %{y:.2f}<extra></extra>'
    ))

    # Add perfect prediction line
    min_val = float(min(_np.nanmin(y_true), _np.nanmin(y_pred_arr)))
    max_val = float(max(_np.nanmax(y_true), _np.nanmax(y_pred_arr)))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash', width=2)
    ))

    # Compute and add best-fit regression line (linear fit of predicted vs actual)
    try:
        # Fit a linear model y_pred = a * y_true + b
        coeffs = _np.polyfit(y_true, y_pred_arr, 1)
        slope, intercept = float(coeffs[0]), float(coeffs[1])
        y_fit = slope * _np.array([min_val, max_val]) + intercept

        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=y_fit,
            mode='lines',
            name='Best Fit',
            line=dict(color='green', width=2)
        ))

        fit_info = f"slope={slope:.3f}, intercept={intercept:.3f}"
    except Exception:
        fit_info = "fit unavailable"

    # Compute R^2 and RMSE for display
    try:
        r2 = r2_score(y_true, y_pred_arr)
        rmse = _np.sqrt(mean_squared_error(y_true, y_pred_arr))
        metrics_text = f"R²={r2:.3f} | RMSE={rmse:.3f}"
    except Exception:
        metrics_text = "R²/ RMSE unavailable"

    fig.update_layout(
        title=f"{model_name} - Predictions vs Actual",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values",
        height=520,
        hovermode='closest',
        annotations=[
            dict(
                x=0.01,
                y=0.99,
                xref='paper',
                yref='paper',
                text=f"{metrics_text} — {fit_info}",
                showarrow=False,
                align='left',
                font=dict(size=11)
            )
        ]
    )

    st.plotly_chart(fig, use_container_width=True)


def plot_residuals_analysis(y_test, y_pred, model_name):
    """
    Plot residuals analysis for regression models
    
    Args:
        y_test: actual target values
        y_pred: predicted target values
        model_name: name of the model
    """
    import plotly.graph_objects as go
    import numpy as np
    
    residuals = y_test - y_pred
    
    fig = go.Figure()
    
    # Add residuals scatter plot
    fig.add_trace(go.Scatter(
        x=y_pred,
        y=residuals,
        mode='markers',
        name='Residuals',
        marker=dict(
            size=8,
            color='purple',
            opacity=0.6
        ),
        hovertemplate='<b>Predicted:</b> %{x:.2f}<br><b>Residual:</b> %{y:.2f}<extra></extra>'
    ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Zero Error")
    
    fig.update_layout(
        title=f"{model_name} - Residuals Plot",
        xaxis_title="Predicted Values",
        yaxis_title="Residuals (Actual - Predicted)",
        height=500,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_cluster_scatter_2d(X_data, labels, model_name, n_clusters=None):
    """
    Create 2D scatter plot of clusters using PCA
    
    Args:
        X_data: feature data (DataFrame or array)
        labels: cluster labels for each sample
        model_name: name of clustering model
        n_clusters: number of clusters (optional, calculated from labels if not provided)
    """
    import plotly.express as px
    from sklearn.decomposition import PCA
    import pandas as pd
    import numpy as np
    
    # Apply PCA to reduce to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_data)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': [f'Cluster {l}' if l != -1 else 'Noise' for l in labels]
    })
    
    # Create scatter plot
    fig = px.scatter(
        plot_df,
        x='PC1',
        y='PC2',
        color='Cluster',
        title=f'{model_name} - 2D Cluster Visualization (PCA)',
        labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)'},
        hover_data={'PC1': ':.2f', 'PC2': ':.2f'}
    )
    
    fig.update_traces(marker=dict(size=8, opacity=0.7))
    fig.update_layout(height=600)
    
    st.plotly_chart(fig, use_container_width=True)


def plot_cluster_distribution(labels, model_name):
    """
    Plot distribution of samples across clusters
    
    Args:
        labels: cluster labels
        model_name: name of clustering model
    """
    import plotly.express as px
    import pandas as pd
    
    # Count samples per cluster
    unique_labels = sorted(set(labels))
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    
    plot_data = []
    for label in unique_labels:
        count = cluster_counts.get(label, 0)
        percentage = (count / len(labels)) * 100
        plot_data.append({
            'Cluster': f'Cluster {label}' if label != -1 else 'Noise',
            'Count': count,
            'Percentage': f'{percentage:.1f}%'
        })
    
    plot_df = pd.DataFrame(plot_data)
    
    fig = px.bar(
        plot_df,
        x='Cluster',
        y='Count',
        text='Percentage',
        title=f'{model_name} - Cluster Distribution',
        color='Cluster'
    )
    
    fig.update_traces(textposition='outside')
    fig.update_layout(height=500, showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)


def plot_silhouette_analysis(X_data, labels, model_name):
    """
    Create silhouette analysis plot for clusters
    
    Args:
        X_data: feature data
        labels: cluster labels
        model_name: name of clustering model
    """
    import plotly.graph_objects as go
    from sklearn.metrics import silhouette_samples, silhouette_score
    import numpy as np
    
    # Calculate silhouette scores
    silhouette_avg = silhouette_score(X_data, labels)
    sample_silhouette_values = silhouette_samples(X_data, labels)
    
    fig = go.Figure()
    
    y_lower = 10
    unique_labels = sorted(set(labels))
    
    for i, label in enumerate(unique_labels):
        # Get silhouette values for this cluster
        cluster_silhouette_values = sample_silhouette_values[labels == label]
        cluster_silhouette_values.sort()
        
        size_cluster = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster
        
        cluster_name = f'Cluster {label}' if label != -1 else 'Noise'
        
        fig.add_trace(go.Scatter(
            x=cluster_silhouette_values,
            y=np.arange(y_lower, y_upper),
            fill='tozerox',
            name=cluster_name,
            mode='lines',
            hovertemplate=f'{cluster_name}<br>Silhouette: %{{x:.3f}}<extra></extra>'
        ))
        
        y_lower = y_upper + 10
    
    # Add average score line
    fig.add_vline(x=silhouette_avg, line_dash="dash", line_color="red",
                  annotation_text=f"Average: {silhouette_avg:.3f}")
    
    fig.update_layout(
        title=f'{model_name} - Silhouette Analysis',
        xaxis_title='Silhouette Coefficient',
        yaxis_title='Cluster',
        height=600,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_elbow_curve(X_data, max_clusters=10):
    """
    Create elbow curve to help determine optimal number of clusters
    
    Args:
        X_data: feature data
        max_clusters: maximum number of clusters to test
    """
    import plotly.graph_objects as go
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    import numpy as np
    
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_clusters + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_data)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_data, kmeans.labels_))
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add inertia trace
    fig.add_trace(go.Scatter(
        x=list(K_range),
        y=inertias,
        name='Inertia',
        mode='lines+markers',
        line=dict(color='blue', width=2),
        yaxis='y'
    ))
    
    # Add silhouette score trace
    fig.add_trace(go.Scatter(
        x=list(K_range),
        y=silhouette_scores,
        name='Silhouette Score',
        mode='lines+markers',
        line=dict(color='green', width=2),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Elbow Curve - Optimal K Selection',
        xaxis=dict(title='Number of Clusters (K)'),
        yaxis=dict(
            title=dict(text='Inertia', font=dict(color='blue')),
            tickfont=dict(color='blue')
        ),
        yaxis2=dict(
            title=dict(text='Silhouette Score', font=dict(color='green')),
            tickfont=dict(color='green'),
            overlaying='y',
            side='right'
        ),
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_cluster_3d(X_data, labels, model_name):
    """
    Create 3D scatter plot of clusters using PCA
    
    Args:
        X_data: feature data
        labels: cluster labels
        model_name: name of clustering model
    """
    import plotly.express as px
    from sklearn.decomposition import PCA
    import pandas as pd
    
    # Apply PCA to reduce to 3D
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_data)
    
    # Create DataFrame for plotting
    plot_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'PC3': X_pca[:, 2],
        'Cluster': [f'Cluster {l}' if l != -1 else 'Noise' for l in labels]
    })
    
    # Create 3D scatter plot
    fig = px.scatter_3d(
        plot_df,
        x='PC1',
        y='PC2',
        z='PC3',
        color='Cluster',
        title=f'{model_name} - 3D Cluster Visualization (PCA)',
        labels={'PC1': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                'PC2': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})',
                'PC3': f'PC3 ({pca.explained_variance_ratio_[2]:.1%})'}
    )
    
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.update_layout(height=700)
    
    st.plotly_chart(fig, use_container_width=True)
