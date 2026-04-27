# ==============================
# ./core/ui_components.py
# ==============================
"""
UI Components Module
Handles all Streamlit UI elements, sidebar controls, and user inputs
"""

import streamlit as st


def render_upload_section():
    """
    Render the upload section in main page
    Returns uploaded file and separator config
    """
    st.header("📁 Upload Data")
    
    uploaded_file = st.file_uploader(
        "Upload your dataset", 
        type=["csv", "data", "txt"]
    )
    
    if uploaded_file:
        # Add file type detection and handling
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        if file_ext == 'data':
            # Try to detect separator for .data files
            sample = uploaded_file.read(2048).decode('utf-8')
            uploaded_file.seek(0)
            
            # Detect most common separator in first few lines
            first_lines = sample.split('\n')[:5]
            possible_seps = [',', ';', '\t', ' ']
            sep_counts = {
                sep: sum(line.count(sep) for line in first_lines) 
                for sep in possible_seps
            }
            detected_sep = max(sep_counts, key=sep_counts.get)
            
            sep = st.text_input(
                "Detected separator. Change if needed:", 
                detected_sep
            )
        else:
            sep = st.text_input("CSV separator (default ',')", ",")
    else:
        sep = ","
    
    return uploaded_file, sep


def render_training_config(auto_detected_task, df, target_col):
    """
    Render all training configuration options in main page with 5-step workflow
    Returns a dictionary with all user selections
    
    Args:
        auto_detected_task: automatically detected task type from target variable
        df: dataframe with data
        target_col: name of target column
    """
    config = {}
    
    st.markdown("---")
    
    # ==========================================
    # STEP 1: Task Type Selection (at the top)
    # ==========================================
    st.markdown("### 🎯 STEP 1: Task Type Selection")
    
    task_type_options = ["Auto", "Classification", "Regression", "Clustering"]
    default_index = 0  # Auto
    
    config['task_type'] = st.selectbox(
        "Select Task Type",
        task_type_options,
        index=default_index,
        help="Auto-detect from target variable, or manually specify task type. Choose Clustering for unsupervised learning."
    )
    
    # Show auto-detection info if Auto is selected
    if config['task_type'] == "Auto":
        from core.model_training import detect_task_type
        auto_detected_detailed = detect_task_type(df[target_col] if target_col and target_col in df.columns else None)
        st.info(f"**Auto-Detection:** {auto_detected_detailed.upper()}")
        current_task_type = auto_detected_task
    elif config['task_type'] == "Clustering":
        current_task_type = "clustering"
    elif config['task_type'] == "Classification":
        current_task_type = "classification"
    else:  # Regression
        current_task_type = "regression"
    
    config['current_task_type'] = current_task_type
    
    st.markdown("---")
    
    # ==========================================
    # STEP 2: Model-Specific Parameters
    # ==========================================
    st.markdown("### ⚙️ STEP 2: Model-Specific Parameters")
    
    if current_task_type == "clustering":
        st.markdown("**Clustering Model Parameters**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**KMeans Parameters**")
            config['kmeans_n_clusters'] = st.slider(
                "Number of clusters (K)",
                min_value=2,
                max_value=15,
                value=3,
                help="Number of clusters to form. Only used if hyperparameter tuning is disabled."
            )
            config['kmeans_init'] = st.selectbox(
                "Initialization method",
                ["k-means++", "random"],
                index=0,
                help="Method for initialization: k-means++ (smart) or random"
            )
        
        with col2:
            st.markdown("**DBSCAN Parameters**")
            config['dbscan_eps'] = st.slider(
                "Epsilon (eps)",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1,
                help="Maximum distance between two samples to be considered neighbors"
            )
            config['dbscan_min_samples'] = st.slider(
                "Min samples",
                min_value=2,
                max_value=20,
                value=5,
                help="Minimum number of samples in a neighborhood for a core point"
            )
            config['dbscan_metric'] = st.selectbox(
                "Distance metric",
                ["euclidean", "manhattan", "cosine"],
                index=0,
                help="Distance metric to use for DBSCAN"
            )
    
    elif current_task_type == "classification":
        st.markdown("**Classification Model Parameters**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Tree-based Models**")
            max_depth_option = st.selectbox(
                "Max tree depth",
                ["None (unlimited)"] + list(range(1, 51)),
                index=0,
                help="Maximum depth of decision trees"
            )
            config['tree_max_depth'] = None if max_depth_option == "None (unlimited)" else max_depth_option
            
            config['n_estimators'] = st.slider(
                "Number of estimators",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Number of trees in ensemble models (RandomForest, XGBoost, etc.)"
            )
        
        with col2:
            st.markdown("**Other Models**")
            config['svm_C'] = st.slider(
                "SVM regularization (C)",
                min_value=0.01,
                max_value=100.0,
                value=1.0,
                step=0.01,
                help="Regularization parameter for SVM"
            )
            config['knn_neighbors'] = st.slider(
                "KNN neighbors (K)",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of neighbors for K-Nearest Neighbors"
            )
    
    else:  # regression
        st.markdown("**Regression Model Parameters**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Tree-based Models**")
            max_depth_option = st.selectbox(
                "Max tree depth",
                ["None (unlimited)"] + list(range(1, 51)),
                index=0,
                help="Maximum depth of decision trees"
            )
            config['tree_max_depth'] = None if max_depth_option == "None (unlimited)" else max_depth_option
            
            config['n_estimators'] = st.slider(
                "Number of estimators",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Number of trees in ensemble models"
            )
        
        with col2:
            st.markdown("**Other Models**")
            config['svr_C'] = st.slider(
                "SVR regularization (C)",
                min_value=0.01,
                max_value=100.0,
                value=1.0,
                step=0.01,
                help="Regularization parameter for SVR"
            )
            config['knn_neighbors'] = st.slider(
                "KNN neighbors (K)",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of neighbors for K-Nearest Neighbors"
            )
    
    st.markdown("---")
    
    # ==========================================
    # STEP 3: Training Parameters
    # ==========================================
    st.markdown("### 📊 STEP 3: Training Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if current_task_type != "clustering":
            config['test_size'] = st.slider(
                "Test set size (fraction)",
                0.1, 0.5, 0.2, step=0.05
            )
        else:
            config['test_size'] = 0.2  # Default for clustering (not used)
        
        config['hyperparameter_tuning'] = st.checkbox(
            "Enable Hyperparameter Tuning",
            help="Use GridSearchCV to find optimal hyperparameters"
        )
        if config['hyperparameter_tuning']:
            st.warning(f"⚠️ Hyperparameter tuning will override manual parameters and test multiple combinations")
    
    with col2:
        if current_task_type != "clustering":
            config['cv_fold_option'] = st.selectbox(
                "Cross-validation folds",
                ["Auto"] + [str(i) for i in range(2, 11)],
                index=0,
                help="Number of folds for cross-validation"
            )
        else:
            config['cv_fold_option'] = "Auto"  # Not used for clustering
    
    st.markdown("---")
    
    # ==========================================
    # STEP 4: Model Selection
    # ==========================================
    st.markdown("### 🤖 STEP 4: Model Selection")
    
    from core.model_training import get_available_models
    
    if current_task_type == "classification":
        models_cls, _ = get_available_models()
        classification_models_list = list(models_cls.keys())
        default_cls = [m for m in ["RandomForestClassifier", "LogisticRegression", "XGBClassifier"] if m in classification_models_list]
        
        config['selected_models_cls'] = st.multiselect(
            "Classification Models",
            classification_models_list,
            default=default_cls[:2] if len(default_cls) >= 2 else default_cls,
            help="Select models to train for classification"
        )
        st.caption(f"📊 {len(classification_models_list)} models available")
        config['selected_models_reg'] = []
        config['selected_models_clustering'] = []
        
    elif current_task_type == "regression":
        _, models_reg = get_available_models()
        regression_models_list = list(models_reg.keys())
        default_reg = [m for m in ["RandomForestRegressor", "LinearRegression", "XGBRegressor"] if m in regression_models_list]
        
        config['selected_models_reg'] = st.multiselect(
            "Regression Models",
            regression_models_list,
            default=default_reg[:2] if len(default_reg) >= 2 else default_reg,
            help="Select models to train for regression"
        )
        st.caption(f"� {len(regression_models_list)} models available")
        config['selected_models_cls'] = []
        config['selected_models_clustering'] = []
        
    else:  # clustering
        clustering_models_list = ["KMeans", "DBSCAN", "AgglomerativeClustering", "MeanShift", "SpectralClustering"]
        default_clustering = ["KMeans", "DBSCAN"]
        
        config['selected_models_clustering'] = st.multiselect(
            "Clustering Models",
            clustering_models_list,
            default=default_clustering,
            help="Select clustering algorithms to train"
        )
        st.caption(f"� {len(clustering_models_list)} models available")
        config['selected_models_cls'] = []
        config['selected_models_reg'] = []
    
    st.markdown("---")
    
    # ==========================================
    # STEP 5: Evaluation Metric
    # ==========================================
    st.markdown("### 🏆 STEP 5: Evaluation Metric")
    
    selected_metric = render_metric_selector(current_task_type)
    
    st.markdown("---")
    st.info(f"**Selected Metric for Optimization:** `{selected_metric}`")
    st.markdown("This metric will be used for:")
    st.markdown("- ✅ Hyperparameter tuning (if enabled)")
    st.markdown("- ✅ Model selection and ranking")
    if current_task_type != "clustering":
        st.markdown("- ✅ Cross-validation scoring")
    
    config['selected_metric'] = selected_metric
    
    return config



def display_file_info(df, uploaded_file):
    """Display information about the uploaded file"""
    st.write(f"**File type:** .{uploaded_file.name.split('.')[-1]}")
    st.write(f"**Number of rows:** {df.shape[0]:,}")
    st.write(f"**Number of columns:** {df.shape[1]}")
    
    # Display any parsing warnings
    if df.shape[1] == 1:
        st.warning("Only one column detected. Please check if the separator is correct.")


def display_dataset_preview(df, n_rows=5):
    """Display a preview of the dataset"""
    st.subheader("Dataset Preview")
    st.dataframe(df.head(n_rows))


def select_target_column(df):
    """Allow user to select the target column"""
    target_col = st.selectbox("Select target column (y)", df.columns)
    return target_col


def create_download_buttons(X_train_proc, X_test_proc, y_train, y_test):
    """Create download buttons for processed data"""
    import pandas as pd
    
    col1, col2 = st.columns(2)
    
    with col1:
        train_data = pd.concat(
            [X_train_proc, y_train.reset_index(drop=True)], 
            axis=1
        )
        st.download_button(
            "📥 Download Processed Train Data",
            train_data.to_csv(index=False).encode('utf-8'),
            "train_processed.csv", 
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        test_data = pd.concat(
            [X_test_proc, y_test.reset_index(drop=True)], 
            axis=1
        )
        st.download_button(
            "📥 Download Processed Test Data",
            test_data.to_csv(index=False).encode('utf-8'),
            "test_processed.csv", 
            "text/csv",
            use_container_width=True
        )


def display_model_performance(performance, best_model, selected_metric, use_cv=True):
    """Display model performance results"""
    st.subheader("🏆 Model Performance")
    
    # Create columns for better layout
    for model_name, perf_value in performance.items():
        with st.container():
            # Check if perf_value is a tuple (supervised learning) or single value (clustering)
            if isinstance(perf_value, tuple):
                score, cv_score = perf_value
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**{model_name}**")
                with col2:
                    st.metric("Test Score", f"{score:.4f}")
                with col3:
                    if use_cv:
                        st.metric("CV Score", f"{cv_score:.4f}")
                    else:
                        st.write("CV: Skipped")
            else:
                # Clustering - single score
                score = perf_value
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**{model_name}**")
                with col2:
                    st.metric(f"{selected_metric.replace('_', ' ').title()}", f"{score:.4f}")
    
    # Highlight best model
    st.success(f"✅ **Recommended Model:** {best_model}")


def display_feature_distribution(df, X):
    """Display feature distribution plot"""
    st.subheader("📊 Feature Distribution")
    selected_feature = st.selectbox("Select feature to plot", X.columns)
    
    import plotly.express as px
    fig_feat = px.histogram(
        df, 
        x=selected_feature, 
        nbins=30, 
        title=f"Distribution of {selected_feature}"
    )
    st.plotly_chart(fig_feat, use_container_width=True)
    
    return selected_feature


def show_error_message(error_msg):
    """Display error message"""
    st.error(f"❌ {error_msg}")


def show_warning_message(warning_msg):
    """Display warning message"""
    st.warning(f"⚠️ {warning_msg}")


def show_success_message(success_msg):
    """Display success message"""
    st.success(f"✅ {success_msg}")


def show_info_message(info_msg):
    """Display info message"""
    st.info(f"ℹ️ {info_msg}")


def get_available_metrics():
    """
    Get all available evaluation metrics
    
    Returns:
        tuple: (classification_metrics_dict, regression_metrics_dict)
    """
    classification_metrics = {
        "accuracy": "Accuracy - Overall correctness",
        "balanced_accuracy": "Balanced Accuracy - Adjusted for class imbalance",
        "f1": "F1 Score - Harmonic mean of precision and recall",
        "f1_macro": "F1 Macro - Unweighted mean F1 per class",
        "f1_micro": "F1 Micro - Global F1 across all classes",
        "precision": "Precision - True positives / (True positives + False positives)",
        "recall": "Recall - True positives / (True positives + False negatives)",
        "roc_auc": "ROC-AUC - Area under ROC curve",
        "matthews_corrcoef": "Matthews Correlation - Balanced measure for imbalanced data",
        "cohen_kappa": "Cohen's Kappa - Agreement corrected for chance"
    }
    
    regression_metrics = {
        "r2": "R² Score - Coefficient of determination",
        "adjusted_r2": "Adjusted R² - R² adjusted for number of predictors",
        "mae": "MAE - Mean Absolute Error",
        "mse": "MSE - Mean Squared Error",
        "rmse": "RMSE - Root Mean Squared Error",
        "mape": "MAPE - Mean Absolute Percentage Error",
        "median_absolute_error": "Median Absolute Error - Robust to outliers",
        "max_error": "Max Error - Worst case prediction error",
        "explained_variance": "Explained Variance - Proportion of variance explained"
    }
    
    return classification_metrics, regression_metrics


def render_metric_selector(task_type):
    """
    Render metric selector based on task type
    
    Args:
        task_type: 'classification', 'regression', or 'clustering'
        
    Returns:
        str: selected metric key
    """
    cls_metrics, reg_metrics = get_available_metrics()
    
    if task_type == "classification":
        st.subheader("📊 Select Evaluation Metric")
        st.markdown("Choose the metric to optimize during training and model selection:")
        
        # Create options with descriptions
        metric_options = list(cls_metrics.keys())
        metric_labels = [f"{key}: {desc}" for key, desc in cls_metrics.items()]
        
        selected_idx = st.selectbox(
            "Classification Metrics",
            range(len(metric_options)),
            format_func=lambda i: metric_labels[i],
            index=0  # Default to accuracy
        )
        
        selected_metric = metric_options[selected_idx]
        
    elif task_type == "clustering":
        st.subheader("📊 Select Evaluation Metric")
        st.markdown("Choose the metric to optimize during model selection:")
        
        # Clustering metrics
        clustering_metrics = {
            "silhouette": "Silhouette Score (higher is better, range: -1 to 1)",
            "davies_bouldin": "Davies-Bouldin Index (lower is better)",
            "calinski_harabasz": "Calinski-Harabasz Score (higher is better)"
        }
        
        metric_options = list(clustering_metrics.keys())
        metric_labels = [f"{key}: {desc}" for key, desc in clustering_metrics.items()]
        
        selected_idx = st.selectbox(
            "Clustering Metrics",
            range(len(metric_options)),
            format_func=lambda i: metric_labels[i],
            index=0  # Default to silhouette
        )
        
        selected_metric = metric_options[selected_idx]
        
    else:  # regression
        st.subheader("📊 Select Evaluation Metric")
        st.markdown("Choose the metric to optimize during training and model selection:")
        
        # Create options with descriptions
        metric_options = list(reg_metrics.keys())
        metric_labels = [f"{key}: {desc}" for key, desc in reg_metrics.items()]
        
        selected_idx = st.selectbox(
            "Regression Metrics",
            range(len(metric_options)),
            format_func=lambda i: metric_labels[i],
            index=0  # Default to r2
        )
        
        selected_metric = metric_options[selected_idx]
    
    return selected_metric
