# ==============================
# ./app_modular.py
# ==============================
"""
OptiMLFlow - Modular Version
Main application file that orchestrates all modules
"""

import streamlit as st
import pandas as pd

# Import custom modules
from core.ui_components import (
    render_upload_section,
    render_training_config,
    display_file_info,
    display_dataset_preview,
    select_target_column,
    create_download_buttons,
    display_model_performance,
    display_feature_distribution,
    show_error_message,
    show_warning_message,
    show_success_message
)
from core.file_handler import (
    load_uploaded_file,
    validate_dataframe,
    get_file_metadata
)
from core.preprocessing import (
    preprocess_and_split,
    get_preprocessing_summary
)
from core.model_training import (
    train_multiple_models,
    train_clustering_models,
    create_performance_dataframe,
    create_detailed_metrics_dataframe,
    create_clustering_performance_dataframe,
    detect_task_type
)
from core.chart_generator import (
    plot_performance_comparison,
    plot_feature_distribution,
    plot_data_overview,
    plot_target_distribution,
    plot_train_vs_test_comparison,
    plot_overfitting_analysis,
    plot_all_metrics_radar,
    plot_metrics_heatmap,
    plot_regression_predictions,
    plot_residuals_analysis,
    plot_cluster_scatter_2d,
    plot_cluster_distribution,
    plot_silhouette_analysis,
    plot_elbow_curve,
    plot_cluster_3d
)
from core.churn_analysis import (
    detect_churn_column,
    encode_churn_target,
    plot_churn_rate_overview,
    plot_churn_by_numeric_features,
    plot_churn_by_categorical_features,
    plot_churn_correlations,
    plot_churn_missing_values,
    plot_roc_curves,
    plot_precision_recall_curves,
    plot_confusion_matrix_churn,
    plot_feature_importance_churn,
    plot_threshold_analysis,
    plot_churn_metrics_comparison,
    preprocess_for_churn,
    apply_imbalance_handling,
    train_churn_models,
    render_churn_config,
    build_churn_metrics_dataframe,
    render_model_save_section,
    render_drift_detection_section,
    compute_dataset_hash,
)
from core.domain_workflows import (
    render_solution_hub,
    render_dataset_health_panel,
    render_upsell_probability_tab,
    render_customer_segmentation_tab,
    render_ecommerce_recommendation_tab,
    render_iot_anomaly_tab,
)
from core.guided_journey import render_guided_journey


# Page configuration
st.set_page_config(
    page_title="OptiMLFlow",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Title
st.title(" OptiMLFlow")
st.markdown("**Automated Machine Learning Pipeline**")
st.markdown("---")

# Upload section
uploaded_file, sep = render_upload_section()

# Main application logic
if uploaded_file is not None:
    # Load the file
    df = load_uploaded_file(uploaded_file, sep)
    
    if df is not None:
        # Validate dataframe
        is_valid, message = validate_dataframe(df)
        
        if not is_valid:
            show_error_message(message)
            st.stop()
        
        # Display file info
        display_file_info(df, uploaded_file)
        
        # Display dataset preview
        display_dataset_preview(df)

        experience_mode = st.radio(
            "Experience Mode",
            ["Automated Product Mode", "Advanced Workspace"],
            horizontal=True,
            key="experience_mode_selector",
            help=(
                "Automated mode provides a guided two-phase product flow. "
                "Advanced mode exposes all model and workflow tabs."
            ),
        )

        if experience_mode == "Automated Product Mode":
            render_dataset_health_panel(df)
            render_guided_journey(df)
            st.caption("Switch to Advanced Workspace mode to access full tab-level controls.")
            st.stop()

        # Solution workflow selector and shared data diagnostics
        render_solution_hub()
        render_dataset_health_panel(df)
        
        # Data overview section
        with st.expander(" Data Overview", expanded=False):
            plot_data_overview(df)
        
        # Select target column
        target_col = select_target_column(df)
        
        # Display target distribution
        with st.expander("Target Variable Analysis", expanded=False):
            # Detect task type for visualization
            y = df[target_col]
            auto_task = (y.dtype == object) or (y.nunique() <= 20 and y.dtype != float)
            task_for_viz = "classification" if auto_task else "regression"
            plot_target_distribution(y, task_for_viz)
        
        # Detect task type for configuration
        y = df[target_col]
        auto_detected_task = detect_task_type(y)
        
        # Create tabs for configuration and training
        st.markdown("---")
        st.subheader(" Model Training & Evaluation")
        
        config_tab, train_tab, churn_tab, upsell_tab, segmentation_tab, ecommerce_tab, iot_tab = st.tabs([
            "⚙️ Training Configuration",
            "🚀 Start Training",
            "🔄 Customer Churn",
            "📈 Upsell Probability",
            "🧩 Customer Segmentation",
            "🛍️ E-commerce Reco",
            "🧠 Adaptive Anomaly"
        ])
        
        with config_tab:
            # Render all training configuration in this tab (5-step workflow)
            config = render_training_config(auto_detected_task, df, target_col)
        
        with train_tab:
            st.markdown("### Ready to Train?")
            st.markdown(f"**Task Type:** `{config.get('task_type', 'auto').upper()}`")
            st.markdown(f"**Optimization Metric:** `{config.get('selected_metric', 'Not selected')}`")
            st.markdown("---")
            
            train_now_button = st.button(" Train Models Now", type="primary", use_container_width=True)
            
            # Initialize session state for training results
            if 'training_results' not in st.session_state:
                st.session_state.training_results = None
            
            if train_now_button:
                # Get the actual task type
                actual_task = config.get('current_task_type', auto_detected_task)
                
                # CLUSTERING WORKFLOW (no train/test split)
                if actual_task == 'clustering':
                    st.markdown("---")
                    st.subheader(" Data Preprocessing (Clustering)")
                    
                    try:
                        # For clustering, preprocess all data together (no split)
                        with st.spinner("Preprocessing data for clustering..."):
                            X_all = df.drop(columns=[target_col])
                            # Apply same preprocessing
                            from sklearn.preprocessing import StandardScaler
                            from sklearn.compose import ColumnTransformer
                            from sklearn.preprocessing import OneHotEncoder
                            
                            numeric_features = X_all.select_dtypes(include=['int64', 'float64']).columns.tolist()
                            categorical_features = X_all.select_dtypes(include=['object', 'category']).columns.tolist()
                            
                            transformers = []
                            if numeric_features:
                                transformers.append(('num', StandardScaler(), numeric_features))
                            if categorical_features:
                                transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features))
                            
                            preprocessor = ColumnTransformer(transformers=transformers)
                            X_all_proc = preprocessor.fit_transform(X_all)
                        
                        show_success_message("Data preprocessing completed!")
                        
                    except Exception as e:
                        show_error_message(f"Preprocessing error: {str(e)}")
                        st.stop()
                    
                    # Clustering model training
                    st.markdown("---")
                    st.subheader(" Training Clustering Models")
                    
                    # Progress tracking
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    
                    def update_progress(current, total, model_name):
                        if total > 0:
                            progress = current / total
                            progress_bar.progress(progress)
                            if model_name:
                                progress_text.text(f"Training {model_name}... ({current}/{total})")
                            else:
                                progress_text.text(f"Training complete!")
                    
                    try:
                        # Train clustering models
                        performance, best_model_name, trained_models, best_params_dict, all_metrics_dict, cluster_labels = \
                            train_clustering_models(
                                X_data=X_all_proc,
                                selected_models_clustering=config['selected_models_clustering'],
                                selected_metric_clustering=config['selected_metric'],
                                hyperparameter_tuning=config['hyperparameter_tuning'],
                                n_clusters=config.get('kmeans_n_clusters', 3),
                                progress_callback=update_progress,
                                kmeans_init=config.get('kmeans_init', 'k-means++'),
                                dbscan_eps=config.get('dbscan_eps', 0.5),
                                dbscan_min_samples=config.get('dbscan_min_samples', 5),
                                dbscan_metric=config.get('dbscan_metric', 'euclidean')
                            )
                        
                        # Store results in session state
                        st.session_state.training_results = {
                            'performance': performance,
                            'best_model_name': best_model_name,
                            'current_task': 'clustering',
                            'trained_models': trained_models,
                            'best_params_dict': best_params_dict,
                            'all_metrics_dict': all_metrics_dict,
                            'cluster_labels': cluster_labels,
                            'X_all': X_all_proc,
                            'config': config,
                            'df': df,
                            'target_col': target_col
                        }
                        
                        progress_text.empty()
                        progress_bar.empty()
                        show_success_message("Clustering model training completed!")
                        
                    except Exception as e:
                        show_error_message(f"Training error: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
                
                # SUPERVISED LEARNING WORKFLOW (classification/regression)
                else:
                    # Preprocessing section
                    st.markdown("---")
                    st.subheader(" Data Preprocessing")
                    
                    try:
                        # Preprocess and split data
                        with st.spinner("Preprocessing data..."):
                            X_train_proc, X_test_proc, y_train, y_test, preprocessor, column_info = \
                                preprocess_and_split(
                                    df, target_col, 
                                    test_size=config['test_size']
                                )
                        
                        show_success_message("Data preprocessing completed!")
                        
                        # Display preprocessing summary
                        with st.expander(" Preprocessing Details", expanded=False):
                            st.markdown(get_preprocessing_summary(column_info))
                        
                        # Download buttons for processed data
                        st.subheader(" Download Processed Data")
                        create_download_buttons(X_train_proc, X_test_proc, y_train, y_test)
                        
                    except Exception as e:
                        show_error_message(f"Preprocessing error: {str(e)}")
                        st.stop()
                    
                    # Model training section
                    st.markdown("---")
                    st.subheader(" Training Models")
                    
                    # Progress tracking
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    
                    def update_progress(current, total, model_name):
                        if total > 0:
                            progress = current / total
                            progress_bar.progress(progress)
                            if model_name:
                                progress_text.text(f"Training {model_name}... ({current}/{total})")
                            else:
                                progress_text.text(f"Training complete!")
                    
                    try:
                        # Train models with selected metric
                        performance, best_model_name, use_cv, cv_folds, current_task, trained_models, best_params_dict, all_metrics_dict = \
                            train_multiple_models(
                                X_train_proc, X_test_proc, y_train, y_test,
                                task_type=config['task_type'],
                                selected_models_cls=config['selected_models_cls'],
                                selected_models_reg=config['selected_models_reg'],
                                selected_metric_cls=config['selected_metric'] if actual_task == 'classification' else 'accuracy',
                                selected_metric_reg=config['selected_metric'] if actual_task == 'regression' else 'r2',
                                hyperparameter_tuning=config['hyperparameter_tuning'],
                                cv_fold_option=config['cv_fold_option'],
                                progress_callback=update_progress,
                                tree_max_depth=config.get('tree_max_depth'),
                                n_estimators=config.get('n_estimators', 100),
                                SVM_C=config.get('SVM_C', 1.0),
                                KNN_neighbors=config.get('KNN_neighbors', 5)
                            )
                        
                        # Store results in session state
                        st.session_state.training_results = {
                            'performance': performance,
                            'best_model_name': best_model_name,
                            'use_cv': use_cv,
                            'cv_folds': cv_folds,
                            'current_task': current_task,
                            'trained_models': trained_models,
                            'best_params_dict': best_params_dict,
                            'all_metrics_dict': all_metrics_dict,
                            'X_test': X_test_proc,
                            'y_test': y_test,
                            'config': config,
                            'df': df,
                            'target_col': target_col
                        }
                        
                        progress_text.empty()
                        progress_bar.empty()
                        show_success_message("Model training completed!")
                        
                    except Exception as e:
                        show_error_message(f"Training error: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
            
            # Display results if available (either just trained or from previous run)
            if st.session_state.training_results is not None:
                results = st.session_state.training_results
                
                # Check if task type changed - if so, clear old results
                actual_task_now = config.get('current_task_type', auto_detected_task)
                
                # Normalize to lowercase for comparison
                if results['current_task'] != actual_task_now.lower():
                    st.info("ℹ Task type changed. Previous results cleared. Please train again.")
                    st.session_state.training_results = None
                else:
                    performance = results['performance']
                    best_model_name = results['best_model_name']
                    current_task = results['current_task']
                    best_params_dict = results['best_params_dict']
                    all_metrics_dict = results['all_metrics_dict']
                    selected_metric = results['config']['selected_metric']
                    hyperparameter_tuning = results['config']['hyperparameter_tuning']
                    df = results['df']
                    target_col = results['target_col']
                    
                    st.markdown("---")
                    
                    # Display hyperparameter tuning results if enabled
                    if hyperparameter_tuning and any(best_params_dict.values()):
                        st.info(" **Hyperparameter Tuning Enabled** - Best parameters found and applied!")
                    
                    # Display task type
                    st.info(f" **Detected Task Type:** {current_task.upper()}")
                    st.info(f" **Optimization Metric:** {selected_metric}")
                    
                    # CLUSTERING RESULTS DISPLAY
                    if current_task == 'clustering' and 'cluster_labels' in results and 'X_all' in results:
                        cluster_labels = results['cluster_labels']
                        X_all = results['X_all']
                        
                        # Display performance
                        display_model_performance(performance, best_model_name, selected_metric, use_cv=False)
                        
                        st.markdown("---")
                        st.subheader(" Clustering Analysis & Visualizations")
                        
                        # Create tabs for clustering visualizations
                        tab1, tab2, tab3, tab4, tab5 = st.tabs([
                            " Performance Comparison",
                            " 2D Cluster Visualization", 
                            " Cluster Distribution",
                            " Silhouette Analysis",
                            " Elbow Curve & 3D Plot"
                        ])
                        
                        with tab1:
                            st.markdown("### Clustering Model Performance")
                            clustering_perf_df = create_clustering_performance_dataframe(
                                performance, all_metrics_dict, best_params_dict
                            )
                            st.dataframe(clustering_perf_df, use_container_width=True)
                        
                        with tab2:
                            st.markdown("### 2D Cluster Scatter Plot (PCA)")
                            selected_model_2d = st.selectbox(
                                "Select clustering model",
                                list(cluster_labels.keys()),
                                key="cluster_2d_selector"
                            )
                            n_clusters_display = len(set(cluster_labels[selected_model_2d])) - (1 if -1 in cluster_labels[selected_model_2d] else 0)
                            plot_cluster_scatter_2d(X_all, cluster_labels[selected_model_2d], selected_model_2d, n_clusters_display)
                        
                        with tab3:
                            st.markdown("### Cluster Distribution")
                            selected_model_dist = st.selectbox(
                                "Select clustering model",
                                list(cluster_labels.keys()),
                                key="cluster_dist_selector"
                            )
                            plot_cluster_distribution(cluster_labels[selected_model_dist], selected_model_dist)
                        
                        with tab4:
                            st.markdown("### Silhouette Analysis")
                            selected_model_sil = st.selectbox(
                                "Select clustering model",
                                list(cluster_labels.keys()),
                                key="cluster_sil_selector"
                            )
                            plot_silhouette_analysis(X_all, cluster_labels[selected_model_sil], selected_model_sil)
                        
                        with tab5:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("### Elbow Curve (Optimal K)")
                                max_k = st.slider("Max clusters to test", 2, 15, 10, key="elbow_max_k")
                                plot_elbow_curve(X_all, max_clusters=max_k)
                            
                            with col2:
                                st.markdown("### 3D Cluster Plot (PCA)")
                                selected_model_3d = st.selectbox(
                                    "Select clustering model",
                                    list(cluster_labels.keys()),
                                    key="cluster_3d_selector"
                                )
                                plot_cluster_3d(X_all, cluster_labels[selected_model_3d], selected_model_3d)
                    
                    # SUPERVISED LEARNING RESULTS DISPLAY (Classification/Regression)
                    elif current_task in ['classification', 'regression'] and 'use_cv' in results:
                        use_cv = results['use_cv']
                        
                        # Check for overfitting warnings
                        overfit_warnings = [
                            model for model, metrics in all_metrics_dict.items()
                            if metrics.get('overfit_warning', '') in [' HIGH', ' MODERATE']
                        ]
                        if overfit_warnings:
                            st.warning(f" **Overfitting detected** in: {', '.join(overfit_warnings)}")
                        
                        if not use_cv:
                            show_warning_message(
                                "Not enough samples per class for cross-validation. "
                                "Skipping CV and using single train/test split."
                            )
                        
                        # Display performance
                        display_model_performance(performance, best_model_name, selected_metric, use_cv)
                        
                        # Create and plot performance comparison using TABS
                        st.markdown("---")
                        st.subheader(" Model Analysis & Visualizations")
                        
                        # Create tabs for different visualizations
                        tab1, tab2, tab3, tab4, tab5 = st.tabs([
                            " Performance Comparison", 
                            " Detailed Metrics",
                            " Overfitting Analysis",
                            " Per-Model Analysis",
                            " Feature Distribution"
                        ])
                        
                        with tab1:
                            st.markdown("### Model Performance Comparison")
                            performance_df = create_performance_dataframe(performance, best_params_dict, all_metrics_dict)
                            st.dataframe(performance_df, use_container_width=True)
                            plot_performance_comparison(performance_df, use_cv)
                            
                        with tab2:
                            st.markdown("### Detailed Metrics Table")
                            detailed_df = create_detailed_metrics_dataframe(all_metrics_dict, current_task)
                            st.dataframe(detailed_df, use_container_width=True)
                            
                            st.markdown("### Metrics Heatmap")
                            plot_metrics_heatmap(all_metrics_dict, current_task)
                            
                        with tab3:
                            st.markdown("### Train vs Test Comparison")
                            plot_train_vs_test_comparison(all_metrics_dict, current_task)
                            
                            st.markdown("### Overfitting Gap Analysis")
                            plot_overfitting_analysis(all_metrics_dict, current_task)
                            
                        with tab4:
                            st.markdown("### Per-Model Metric Analysis")
                            selected_model = st.selectbox(
                                "Select model to analyze",
                                list(all_metrics_dict.keys()),
                                key="per_model_analysis_selector"
                            )
                            
                            # Display metrics for selected model
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"#### {selected_model} Metrics")
                                metrics = all_metrics_dict[selected_model]
                                
                                if current_task == "classification":
                                    st.metric("Test Accuracy", f"{metrics.get('test_accuracy', 0):.4f}")
                                    st.metric("Test F1 Score", f"{metrics.get('test_f1', 0):.4f}")
                                    st.metric("Test Precision", f"{metrics.get('test_precision', 0):.4f}")
                                    st.metric("Test Recall", f"{metrics.get('test_recall', 0):.4f}")
                                    if 'test_roc_auc' in metrics:
                                        st.metric("Test ROC-AUC", f"{metrics.get('test_roc_auc', 0):.4f}")
                                else:
                                    st.metric("Test R²", f"{metrics.get('test_r2', 0):.4f}")
                                    st.metric("Test MAE", f"{metrics.get('test_mae', 0):.4f}")
                                    st.metric("Test RMSE", f"{metrics.get('test_rmse', 0):.4f}")
                                
                                st.metric("Overfit Difference", 
                                         f"{metrics.get('overfit_difference', 0):.4f}",
                                         delta=metrics.get('overfit_warning', 'N/A'))
                            
                            with col2:
                                plot_all_metrics_radar(all_metrics_dict, current_task, selected_model)
                            
                            # Add regression prediction visualizations
                            if current_task == "regression":
                                st.markdown("---")
                                st.markdown(f"#### {selected_model} - Prediction Analysis")
                                
                                # Get trained model and make predictions
                                trained_models = results['trained_models']
                                y_test = results['y_test']
                                
                                if selected_model in trained_models:
                                    model = trained_models[selected_model]
                                    # Use stored processed X_test and y_test to generate prediction plots
                                    X_test_stored = results.get('X_test')
                                    y_test_stored = results.get('y_test')

                                    if X_test_stored is None or y_test_stored is None:
                                        st.info(" Prediction plots require test data. Re-train to see predictions.")
                                    else:
                                        try:
                                            y_pred = model.predict(X_test_stored)
                                            col_pred1, col_pred2 = st.columns(2)
                                            with col_pred1:
                                                plot_regression_predictions(y_test_stored, y_pred, selected_model)
                                            with col_pred2:
                                                plot_residuals_analysis(y_test_stored, y_pred, selected_model)
                                        except Exception as e:
                                            st.error(f"Could not generate prediction plots: {str(e)}")
                                            import traceback
                                            st.text(traceback.format_exc())
                        
                        with tab5:
                            st.markdown("### Feature Distribution Analysis")
                            X = df.drop(columns=[target_col])
                            selected_feature = st.selectbox(
                                "Select feature to plot", 
                                X.columns,
                            key="feature_distribution_selector"
                            )
                            plot_feature_distribution(df, selected_feature)
    
        # ============================================================
        # CUSTOMER CHURN SPECIALIZATION TAB
        # ============================================================
        with churn_tab:
            st.markdown("### 🔄 Customer Churn Analysis")
            st.markdown(
                "End-to-end customer churn detection pipeline with automatic class-imbalance "
                "handling, churn-optimised models, and business-focused visualisations."
            )
            st.markdown("---")

            # ── Churn column selection ────────────────────────────────────
            st.markdown("#### 1️⃣  Identify the Churn Column")
            auto_churn_col = detect_churn_column(df)
            col_options = list(df.columns)

            if auto_churn_col:
                default_idx = col_options.index(auto_churn_col)
                st.success(f"✅ Auto-detected churn column: `{auto_churn_col}`")
            else:
                default_idx = 0
                st.warning("⚠️ Could not auto-detect churn column. Please select it manually.")

            churn_col = st.selectbox(
                "Churn / Target column",
                col_options,
                index=default_idx,
                key="churn_col_selector",
                help="Column that indicates whether a customer churned (binary: 0/1, Yes/No, active/cancelled, etc.)"
            )

            # ── Data Preview ─────────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 2️⃣  Data Preview & Exploratory Analysis")

            prev_tab1, prev_tab2, prev_tab3, prev_tab4, prev_tab5 = st.tabs([
                "📊 Churn Overview",
                "🔢 Numeric Features",
                "🏷️ Categorical Features",
                "📈 Correlations",
                "❓ Missing Values"
            ])

            with prev_tab1:
                try:
                    plot_churn_rate_overview(df, churn_col)
                except Exception as e:
                    st.error(f"Could not render churn overview: {e}")

            with prev_tab2:
                try:
                    plot_churn_by_numeric_features(df, churn_col)
                except Exception as e:
                    st.error(f"Could not render numeric distributions: {e}")

            with prev_tab3:
                try:
                    plot_churn_by_categorical_features(df, churn_col)
                except Exception as e:
                    st.error(f"Could not render categorical breakdown: {e}")

            with prev_tab4:
                try:
                    plot_churn_correlations(df, churn_col)
                except Exception as e:
                    st.error(f"Could not render correlations: {e}")

            with prev_tab5:
                try:
                    plot_churn_missing_values(df)
                except Exception as e:
                    st.error(f"Could not render missing values: {e}")

            # ── Configuration ─────────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 3️⃣  Model Configuration")
            churn_config = render_churn_config()

            # ── Train button ──────────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 4️⃣  Train Churn Models")

            if 'churn_results' not in st.session_state:
                st.session_state.churn_results = None

            train_churn_btn = st.button(
                "🚀 Train Churn Models",
                type="primary",
                use_container_width=True,
                key="train_churn_button"
            )

            if train_churn_btn:
                # Validate model selection
                if not churn_config.get("selected_models"):
                    st.error("Please select at least one model.")
                    st.stop()

                with st.spinner("⚙️ Preprocessing data..."):
                    try:
                        X_train_c, X_test_c, y_train_c, y_test_c, feat_names = \
                            preprocess_for_churn(df, churn_col, churn_config["test_size"])
                    except Exception as e:
                        st.error(f"Preprocessing failed: {e}")
                        import traceback; st.text(traceback.format_exc())
                        st.stop()

                # Imbalance handling
                imb_method = churn_config.get("imbalance_method", "smote")
                class_weight = None
                if imb_method != "none":
                    with st.spinner(f"⚖️ Applying {imb_method.upper()} imbalance correction..."):
                        X_train_c, y_train_c, class_weight = apply_imbalance_handling(
                            X_train_c, y_train_c, imb_method
                        )
                        if imb_method not in ("none", "class_weight"):
                            st.success(
                                f"✅ {imb_method.upper()}: resampled training set → "
                                f"{int(y_train_c.sum()):,} churned / {int((y_train_c==0).sum()):,} retained"
                            )

                # Training
                churn_progress_text = st.empty()
                churn_progress_bar = st.progress(0)

                def churn_progress(current, total, name):
                    if total > 0:
                        churn_progress_bar.progress(current / total)
                        if name:
                            churn_progress_text.text(f"Training {name}... ({current}/{total})")
                        else:
                            churn_progress_text.text("Training complete!")

                try:
                    metrics_dict, proba_dict, pred_dict, trained_models_c, best_churn_model = \
                        train_churn_models(
                            X_train_c, X_test_c, y_train_c, y_test_c,
                            selected_model_names=churn_config["selected_models"],
                            class_weight=class_weight,
                            n_estimators=churn_config.get("n_estimators", 200),
                            max_depth=churn_config.get("max_depth"),
                            svm_c=churn_config.get("svm_c", 1.0),
                            cv_folds=churn_config.get("cv_folds", 5),
                            optimize_metric=churn_config.get("optimize_metric", "roc_auc"),
                            progress_callback=churn_progress
                        )

                    churn_progress_text.empty()
                    churn_progress_bar.empty()

                    st.session_state.churn_results = {
                        "metrics_dict": metrics_dict,
                        "proba_dict": proba_dict,
                        "pred_dict": pred_dict,
                        "trained_models": trained_models_c,
                        "best_model": best_churn_model,
                        "y_test": y_test_c,
                        "feat_names": feat_names,
                        "optimize_metric": churn_config.get("optimize_metric", "roc_auc")
                    }

                    show_success_message(f"✅ Training complete! Best model: **{best_churn_model}**")

                except Exception as e:
                    churn_progress_text.empty()
                    churn_progress_bar.empty()
                    st.error(f"Training failed: {e}")
                    import traceback; st.text(traceback.format_exc())

            # ── Results ──────────────────────────────────────────────────
            if st.session_state.get("churn_results") is not None:
                cr = st.session_state.churn_results
                metrics_dict = cr["metrics_dict"]
                proba_dict = cr["proba_dict"]
                pred_dict = cr["pred_dict"]
                trained_models_c = cr["trained_models"]
                best_churn_model = cr["best_model"]
                y_test_c = cr["y_test"]
                feat_names = cr["feat_names"]
                opt_metric = cr["optimize_metric"]

                st.markdown("---")
                st.markdown("#### 5️⃣  Results")

                # Best model banner
                best_roc = metrics_dict[best_churn_model].get("roc_auc", 0)
                best_recall = metrics_dict[best_churn_model].get("recall", 0)
                st.success(
                    f"🏆 **Best Model:** {best_churn_model}  |  "
                    f"ROC-AUC: {best_roc:.4f}  |  Recall: {best_recall:.4f}  "
                    f"(optimised on {opt_metric.upper()})"
                )

                # Metrics table
                churn_df_table = build_churn_metrics_dataframe(metrics_dict)
                st.dataframe(churn_df_table, use_container_width=True, hide_index=True)

                # Result tabs
                (
                    res_tab1, res_tab2, res_tab3,
                    res_tab4, res_tab5, res_tab6
                ) = st.tabs([
                    "📊 Metric Comparison",
                    "📉 ROC Curves",
                    "🎯 Precision–Recall",
                    "🧮 Confusion Matrix",
                    "🌳 Feature Importance",
                    "⚡ Threshold Analyser"
                ])

                with res_tab1:
                    plot_churn_metrics_comparison(metrics_dict)

                with res_tab2:
                    plot_roc_curves(y_test_c, proba_dict)

                with res_tab3:
                    plot_precision_recall_curves(y_test_c, proba_dict)

                with res_tab4:
                    cm_model = st.selectbox(
                        "Select model",
                        list(pred_dict.keys()),
                        key="churn_cm_model"
                    )
                    plot_confusion_matrix_churn(
                        y_test_c, pred_dict[cm_model], cm_model
                    )

                with res_tab5:
                    fi_model = st.selectbox(
                        "Select model",
                        list(trained_models_c.keys()),
                        key="churn_fi_model"
                    )
                    top_n = st.slider("Top N features", 5, min(30, len(feat_names)), 15,
                                      key="churn_fi_top_n")
                    plot_feature_importance_churn(
                        trained_models_c[fi_model], feat_names, fi_model, top_n
                    )

                with res_tab6:
                    thr_model = st.selectbox(
                        "Select model",
                        [m for m, p in proba_dict.items() if p is not None],
                        key="churn_thr_model"
                    )
                    if thr_model and proba_dict.get(thr_model) is not None:
                        plot_threshold_analysis(
                            y_test_c, proba_dict[thr_model], thr_model
                        )
                    else:
                        st.info("Threshold analysis requires probability scores. "
                                "Select a probabilistic model.")

                # ── Model Save & Drift Detection ─────────────────────────
                st.markdown("---")
                render_model_save_section(
                    trained_models_c,
                    metrics_dict,
                    feat_names,
                    best_churn_model,
                    churn_config,
                    compute_dataset_hash(df),
                    key_prefix="churn",
                )

                st.markdown("---")
                render_drift_detection_section(
                    df_reference=df,
                    key_prefix="churn_drift",
                )

        # ============================================================
        # CUSTOMER ANALYTICS - UPSELL PROBABILITY TAB
        # ============================================================
        with upsell_tab:
            render_upsell_probability_tab(df)

        # ============================================================
        # CUSTOMER ANALYTICS - SEGMENTATION TAB
        # ============================================================
        with segmentation_tab:
            render_customer_segmentation_tab(df)

        # ============================================================
        # E-COMMERCE RECOMMENDATION SYSTEM TAB
        # ============================================================
        with ecommerce_tab:
            render_ecommerce_recommendation_tab(df)

        # ============================================================
        # GENERIC ADAPTIVE ANOMALY TAB
        # ============================================================
        with iot_tab:
            render_iot_anomaly_tab(df)

else:
    st.info(" Please upload a dataset above to get started!")
    
    st.markdown("""
    ### Welcome to OptiMLFlow

    OptiMLFlow is now structured as a two-phase AI product:

    **Phase 1: General Auto-Modelling**
    - Auto-detect target and task type
    - Run preprocessing, feature engineering, training, and evaluation
    - Explain results through insight cards and narrative steps

    **Phase 2: Industry-Specific Analysis**
    - Customer Analytics (Churn, Upsell, Segmentation)
    - E-commerce Recommendation Systems
    - IoT / Manufacturing Anomaly Detection
    - Focus on business decisions, not only metrics

    **How to start:**
    1. Upload a CSV/TXT/DATA dataset
    2. Keep `Automated Product Mode` for the guided flow
    3. Run Phase 1 to build a baseline model
    4. Enter Phase 2 to generate domain insights
    5. Switch to `Advanced Workspace` when you need full manual controls
    
    ---
    *Built with Streamlit, scikit-learn, and Plotly*
    """)
