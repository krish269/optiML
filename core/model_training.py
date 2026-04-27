# ==============================
# ./core/model_training.py
# ==============================
"""
Enhanced Model Training Module
Handles model training, hyperparameter tuning, and evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor
)
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error
)

# Try to import XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Try to import LightGBM
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def detect_task_type(y):
    """
    Automatically detect if task is classification or regression
    
    Args:
        y: target variable (pandas Series or array)
        
    Returns:
        str: 'classification' or 'regression'
    """
    auto_task = (y.dtype == object) or (y.nunique() <= 20 and y.dtype != float)
    return 'classification' if auto_task else 'regression'


def get_available_models(tree_max_depth=None, n_estimators=100, SVM_C=1.0, KNN_neighbors=5):
    """
    Get dictionaries of available models with custom parameters
    
    Args:
        tree_max_depth: Max depth for tree-based models (None for unlimited)
        n_estimators: Number of estimators for ensemble models
        SVM_C: Regularization parameter for SVM/SVR
        KNN_neighbors: Number of neighbors for KNN
    
    Returns:
        tuple: (classification_models, regression_models)
    """
    models_classification = {
        "RandomForestClassifier": RandomForestClassifier(
            random_state=42, 
            n_estimators=n_estimators,
            max_depth=tree_max_depth
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            random_state=42,
            n_estimators=n_estimators,
            max_depth=tree_max_depth if tree_max_depth else 3
        ),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "AdaBoostClassifier": AdaBoostClassifier(
            random_state=42,
            n_estimators=n_estimators
        ),
        "GaussianNB": GaussianNB(),
        "SVC": SVC(random_state=42, probability=True, C=SVM_C),
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=KNN_neighbors),
        "DecisionTreeClassifier": DecisionTreeClassifier(
            random_state=42,
            max_depth=tree_max_depth
        )
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models_classification["XGBClassifier"] = XGBClassifier(
            random_state=42, 
            eval_metric='logloss',
            n_estimators=n_estimators,
            max_depth=tree_max_depth if tree_max_depth else 6
        )
    
    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        models_classification["LGBMClassifier"] = LGBMClassifier(
            random_state=42, 
            verbose=-1,
            n_estimators=n_estimators,
            max_depth=tree_max_depth if tree_max_depth else -1
        )
    
    models_regression = {
        "RandomForestRegressor": RandomForestRegressor(
            random_state=42,
            n_estimators=n_estimators,
            max_depth=tree_max_depth
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            random_state=42,
            n_estimators=n_estimators,
            max_depth=tree_max_depth if tree_max_depth else 3
        ),
        "LinearRegression": LinearRegression(),
        "AdaBoostRegressor": AdaBoostRegressor(
            random_state=42,
            n_estimators=n_estimators
        ),
        "SVR": SVR(C=SVM_C),
        "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=KNN_neighbors),
        "DecisionTreeRegressor": DecisionTreeRegressor(
            random_state=42,
            max_depth=tree_max_depth
        )
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models_regression["XGBRegressor"] = XGBRegressor(
            random_state=42,
            n_estimators=n_estimators,
            max_depth=tree_max_depth if tree_max_depth else 6
        )
    
    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        models_regression["LGBMRegressor"] = LGBMRegressor(
            random_state=42, 
            verbose=-1,
            n_estimators=n_estimators,
            max_depth=tree_max_depth if tree_max_depth else -1
        )
    
    return models_classification, models_regression


def get_param_grids():
    """
    Get hyperparameter grids for different models
    
    Returns:
        tuple: (classification_grids, regression_grids)
    """
    param_grids_cls = {
        "RandomForestClassifier": {
            'n_estimators': [100, 200],
            'max_depth': [None, 5, 10]
        },
        "GradientBoostingClassifier": {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1]
        },
        "LogisticRegression": {
            'C': [0.1, 1, 10]
        },
        "AdaBoostClassifier": {
            'n_estimators': [50, 100],
            'learning_rate': [0.5, 1.0]
        },
        "GaussianNB": {},
        "SVC": {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        },
        "KNeighborsClassifier": {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        },
        "DecisionTreeClassifier": {
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }
    }
    
    # Add XGBoost grids if available
    if XGBOOST_AVAILABLE:
        param_grids_cls["XGBClassifier"] = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    
    # Add LightGBM grids if available
    if LIGHTGBM_AVAILABLE:
        param_grids_cls["LGBMClassifier"] = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    
    param_grids_reg = {
        "RandomForestRegressor": {
            'n_estimators': [100, 200],
            'max_depth': [None, 5, 10]
        },
        "GradientBoostingRegressor": {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1]
        },
        "LinearRegression": {},
        "AdaBoostRegressor": {
            'n_estimators': [50, 100],
            'learning_rate': [0.5, 1.0]
        },
        "SVR": {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        },
        "KNeighborsRegressor": {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        },
        "DecisionTreeRegressor": {
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }
    }
    
    # Add XGBoost grids if available
    if XGBOOST_AVAILABLE:
        param_grids_reg["XGBRegressor"] = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    
    # Add LightGBM grids if available
    if LIGHTGBM_AVAILABLE:
        param_grids_reg["LGBMRegressor"] = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2]
        }
    
    return param_grids_cls, param_grids_reg


def determine_cv_folds(y_train, cv_fold_option, task_type):
    """
    Determine the number of CV folds and whether to use CV
    
    Args:
        y_train: training target variable
        cv_fold_option: user-selected CV option
        task_type: 'classification' or 'regression'
        
    Returns:
        tuple: (cv_folds: int, use_cv: bool)
    """
    if cv_fold_option == "Auto":
        if task_type == "classification":
            min_class_count = y_train.value_counts().min()
            cv_folds = min(10, min_class_count)
            use_cv = cv_folds >= 2
        else:
            cv_folds = min(10, len(y_train))
            use_cv = cv_folds >= 2
    else:
        cv_folds = int(cv_fold_option)
        use_cv = cv_folds >= 2
    
    return cv_folds, use_cv


def train_single_model(model, X_train, y_train, X_test, y_test,
                       param_grid, hyperparameter_tuning,
                       cv_folds, use_cv, selected_metric,
                       task_type, metric_func_map, model_name):
    """
    Train a single model with optional hyperparameter tuning
    
    Returns:
        tuple: (score, cv_score, best_estimator, best_params, all_metrics)
    """
    best_params = None
    
    # Hyperparameter tuning
    if hyperparameter_tuning and param_grid:
        # Use at least 3 folds for GridSearchCV, but not more than available
        grid_cv_folds = max(3, min(cv_folds, 5))
        
        # Convert metric for GridSearchCV
        grid_scoring_map = {
            # Classification
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
            "f1": "f1_weighted",
            "f1_macro": "f1_macro",
            "f1_micro": "f1_micro",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
            "roc_auc": "roc_auc",
            "matthews_corrcoef": "matthews_corrcoef",
            "cohen_kappa": "cohen_kappa",
            # Regression
            "r2": "r2",
            "adjusted_r2": "r2",  # Use r2 for tuning, calculate adjusted later
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
            "rmse": "neg_root_mean_squared_error",
            "mape": "neg_mean_absolute_percentage_error",
            "median_absolute_error": "neg_median_absolute_error",
            "max_error": "max_error",
            "explained_variance": "explained_variance"
        }
        
        grid_scoring = grid_scoring_map.get(selected_metric, selected_metric)
        
        try:
            gs = GridSearchCV(
                model, 
                param_grid, 
                cv=grid_cv_folds, 
                scoring=grid_scoring,
                n_jobs=-1,
                verbose=0,
                error_score='raise'
            )
            gs.fit(X_train, y_train)
            best_estimator = gs.best_estimator_
            best_params = gs.best_params_
            
            # Print tuning results
            print(f"✓ {model_name} - Best params: {best_params}")
            print(f"  Best {grid_scoring} score: {gs.best_score_:.4f}")
            
        except Exception as e:
            print(f"⚠ GridSearchCV failed for {model_name}: {str(e)}")
            print(f"  Falling back to default parameters...")
            best_estimator = model.fit(X_train, y_train)
            best_params = None
    else:
        best_estimator = model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = best_estimator.predict(X_train)
    y_pred_test = best_estimator.predict(X_test)
    
    # Calculate comprehensive metrics
    all_metrics = {}
    
    if task_type == "classification":
        from sklearn.metrics import (
            precision_score, recall_score, balanced_accuracy_score,
            matthews_corrcoef, cohen_kappa_score
        )
        
        # Calculate all classification metrics
        all_metrics['train_accuracy'] = accuracy_score(y_train, y_pred_train)
        all_metrics['test_accuracy'] = accuracy_score(y_test, y_pred_test)
        all_metrics['train_balanced_accuracy'] = balanced_accuracy_score(y_train, y_pred_train)
        all_metrics['test_balanced_accuracy'] = balanced_accuracy_score(y_test, y_pred_test)
        all_metrics['train_f1'] = f1_score(y_train, y_pred_train, average="weighted", zero_division=0)
        all_metrics['test_f1'] = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)
        all_metrics['train_f1_macro'] = f1_score(y_train, y_pred_train, average="macro", zero_division=0)
        all_metrics['test_f1_macro'] = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
        all_metrics['train_f1_micro'] = f1_score(y_train, y_pred_train, average="micro", zero_division=0)
        all_metrics['test_f1_micro'] = f1_score(y_test, y_pred_test, average="micro", zero_division=0)
        all_metrics['train_precision'] = precision_score(y_train, y_pred_train, average="weighted", zero_division=0)
        all_metrics['test_precision'] = precision_score(y_test, y_pred_test, average="weighted", zero_division=0)
        all_metrics['train_recall'] = recall_score(y_train, y_pred_train, average="weighted", zero_division=0)
        all_metrics['test_recall'] = recall_score(y_test, y_pred_test, average="weighted", zero_division=0)
        all_metrics['test_matthews_corrcoef'] = matthews_corrcoef(y_test, y_pred_test)
        all_metrics['test_cohen_kappa'] = cohen_kappa_score(y_test, y_pred_test)
        
        # ROC-AUC
        if hasattr(best_estimator, "predict_proba"):
            y_train_proba = best_estimator.predict_proba(X_train)
            y_test_proba = best_estimator.predict_proba(X_test)
            
            n_classes = y_train_proba.shape[1]
            
            if n_classes == 2:
                all_metrics['train_roc_auc'] = roc_auc_score(y_train, y_train_proba[:, 1])
                all_metrics['test_roc_auc'] = roc_auc_score(y_test, y_test_proba[:, 1])
            else:
                from sklearn.preprocessing import label_binarize
                classes = np.unique(y_train)
                y_train_bin = label_binarize(y_train, classes=classes)
                y_test_bin = label_binarize(y_test, classes=classes)
                all_metrics['train_roc_auc'] = roc_auc_score(y_train_bin, y_train_proba, average='weighted', multi_class='ovr')
                all_metrics['test_roc_auc'] = roc_auc_score(y_test_bin, y_test_proba, average='weighted', multi_class='ovr')
        
        # Select primary score based on selected metric
        metric_key_map = {
            "accuracy": "test_accuracy",
            "balanced_accuracy": "test_balanced_accuracy",
            "f1": "test_f1",
            "f1_macro": "test_f1_macro",
            "f1_micro": "test_f1_micro",
            "precision": "test_precision",
            "recall": "test_recall",
            "roc_auc": "test_roc_auc",
            "matthews_corrcoef": "test_matthews_corrcoef",
            "cohen_kappa": "test_cohen_kappa"
        }
        
        score = all_metrics.get(metric_key_map.get(selected_metric, "test_accuracy"), all_metrics['test_accuracy'])
            
    else:  # Regression
        from sklearn.metrics import (
            mean_squared_error, mean_absolute_percentage_error,
            median_absolute_error, max_error, explained_variance_score
        )
        
        # Calculate all regression metrics
        all_metrics['train_r2'] = r2_score(y_train, y_pred_train)
        all_metrics['test_r2'] = r2_score(y_test, y_pred_test)
        
        # Adjusted R² calculation
        n_samples = len(y_test)
        n_features = X_test.shape[1]
        all_metrics['test_adjusted_r2'] = 1 - (1 - all_metrics['test_r2']) * (n_samples - 1) / (n_samples - n_features - 1)
        
        all_metrics['train_mae'] = mean_absolute_error(y_train, y_pred_train)
        all_metrics['test_mae'] = mean_absolute_error(y_test, y_pred_test)
        all_metrics['train_mse'] = mean_squared_error(y_train, y_pred_train)
        all_metrics['test_mse'] = mean_squared_error(y_test, y_pred_test)
        all_metrics['train_rmse'] = np.sqrt(all_metrics['train_mse'])
        all_metrics['test_rmse'] = np.sqrt(all_metrics['test_mse'])
        all_metrics['test_median_absolute_error'] = median_absolute_error(y_test, y_pred_test)
        all_metrics['test_max_error'] = max_error(y_test, y_pred_test)
        all_metrics['test_explained_variance'] = explained_variance_score(y_test, y_pred_test)
        
        # MAPE (handle zero values)
        try:
            all_metrics['train_mape'] = mean_absolute_percentage_error(y_train, y_pred_train)
            all_metrics['test_mape'] = mean_absolute_percentage_error(y_test, y_pred_test)
        except:
            all_metrics['train_mape'] = np.nan
            all_metrics['test_mape'] = np.nan
        
        # Select primary score based on metric
        metric_key_map = {
            "r2": "test_r2",
            "adjusted_r2": "test_adjusted_r2",
            "mae": "test_mae",
            "mse": "test_mse",
            "rmse": "test_rmse",
            "mape": "test_mape",
            "median_absolute_error": "test_median_absolute_error",
            "max_error": "test_max_error",
            "explained_variance": "test_explained_variance"
        }
        
        score = all_metrics.get(metric_key_map.get(selected_metric, "test_r2"), all_metrics['test_r2'])
    
    # Detect overfitting
    if task_type == "classification":
        train_score = all_metrics['train_accuracy']
        test_score = all_metrics['test_accuracy']
    else:
        train_score = all_metrics['train_r2']
        test_score = all_metrics['test_r2']
    
    overfit_diff = train_score - test_score
    all_metrics['overfit_difference'] = overfit_diff
    
    if overfit_diff > 0.1:  # More than 10% difference
        all_metrics['overfit_warning'] = "⚠️ HIGH"
        print(f"  ⚠️ WARNING: Possible overfitting detected!")
        print(f"     Train score: {train_score:.4f} | Test score: {test_score:.4f} | Diff: {overfit_diff:.4f}")
    elif overfit_diff > 0.05:
        all_metrics['overfit_warning'] = "⚡ MODERATE"
        print(f"  ⚡ Moderate train/test gap: {overfit_diff:.4f}")
    else:
        all_metrics['overfit_warning'] = "✓ GOOD"
    
    # Cross-validation score
    if use_cv:
        try:
            # Convert metric for cross_val_score (same mapping as GridSearchCV)
            cv_scoring_map = {
                "accuracy": "accuracy",
                "balanced_accuracy": "balanced_accuracy",
                "f1": "f1_weighted",
                "f1_macro": "f1_macro",
                "f1_micro": "f1_micro",
                "precision": "precision_weighted",
                "recall": "recall_weighted",
                "roc_auc": "roc_auc",
                "matthews_corrcoef": "matthews_corrcoef",
                "cohen_kappa": "cohen_kappa",
                "r2": "r2",
                "adjusted_r2": "r2",
                "mae": "neg_mean_absolute_error",
                "mse": "neg_mean_squared_error",
                "rmse": "neg_root_mean_squared_error",
                "mape": "neg_mean_absolute_percentage_error",
                "median_absolute_error": "neg_median_absolute_error",
                "max_error": "max_error",
                "explained_variance": "explained_variance"
            }
            
            cv_scoring = cv_scoring_map.get(selected_metric, selected_metric)
                
            cv_scores = cross_val_score(
                best_estimator, X_train, y_train,
                cv=cv_folds, 
                scoring=cv_scoring,
                n_jobs=-1
            )
            cv_score = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Convert back from negative if needed
            if cv_scoring.startswith("neg_"):
                cv_score = -cv_score
            
            all_metrics['cv_score'] = cv_score
            all_metrics['cv_std'] = cv_std
                
        except Exception as e:
            print(f"⚠ Cross-validation failed for {model_name}: {str(e)}")
            cv_score = float('nan')
            all_metrics['cv_score'] = cv_score
            all_metrics['cv_std'] = float('nan')
    else:
        cv_score = float('nan')
        all_metrics['cv_score'] = cv_score
        all_metrics['cv_std'] = float('nan')
    
    return score, cv_score, best_estimator, best_params, all_metrics


def train_multiple_models(X_train, X_test, y_train, y_test,
                         task_type="Auto",
                         selected_models_cls=None,
                         selected_models_reg=None,
                         selected_metric_cls="accuracy",
                         selected_metric_reg="r2",
                         hyperparameter_tuning=False,
                         cv_fold_option="Auto",
                         progress_callback=None,
                         tree_max_depth=None,
                         n_estimators=100,
                         SVM_C=1.0,
                         KNN_neighbors=5):
    """
    Train multiple models and return performance metrics
    
    Args:
        progress_callback: Optional callback function(current, total, model_name) for progress updates
        tree_max_depth: Max depth for tree-based models (None for unlimited)
        n_estimators: Number of estimators for ensemble models
        SVM_C: Regularization parameter for SVM/SVR
        KNN_neighbors: Number of neighbors for KNN
    
    Returns:
        tuple: (performance, best_model_name, use_cv, cv_folds, current_task, 
                trained_models, best_params_dict, all_metrics_dict)
    """
    # Detect task type
    if task_type == "Auto":
        current_task = detect_task_type(y_train)
    else:
        current_task = task_type.lower()
    
    # Label encode target variable for classification if needed
    label_encoder = None
    y_train_encoded = y_train
    y_test_encoded = y_test
    
    if current_task == "classification":
        # Always use LabelEncoder for classification to ensure labels are 0-indexed and consecutive
        # This is required by some models like XGBoost
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        # Show encoding if labels were transformed
        original_classes = list(label_encoder.classes_)
        encoded_classes = list(range(len(label_encoder.classes_)))
        if original_classes != encoded_classes:
            print(f"ℹ️ Label encoding applied: {original_classes} → {encoded_classes}")
    
    # Get models and param grids with custom parameters
    models_cls, models_reg = get_available_models(
        tree_max_depth=tree_max_depth,
        n_estimators=n_estimators,
        SVM_C=SVM_C,
        KNN_neighbors=KNN_neighbors
    )
    param_grids_cls, param_grids_reg = get_param_grids()
    
    # Select models to train
    if current_task == "classification":
        if not selected_models_cls:
            selected_models_cls = ["RandomForestClassifier", "LogisticRegression"]
        models_to_train = {k: models_cls[k] for k in selected_models_cls}
        param_grids = param_grids_cls
        selected_metric = selected_metric_cls
        metric_func_map = {
            "accuracy": accuracy_score,
            "f1": f1_score,
            "roc_auc": roc_auc_score
        }
    else:
        if not selected_models_reg:
            selected_models_reg = ["RandomForestRegressor", "LinearRegression"]
        models_to_train = {k: models_reg[k] for k in selected_models_reg}
        param_grids = param_grids_reg
        selected_metric = selected_metric_reg
        metric_func_map = {
            "r2": r2_score,
            "mae": mean_absolute_error
        }
    
    # Determine CV folds
    cv_folds, use_cv = determine_cv_folds(y_train, cv_fold_option, current_task)
    
    # Adjust cv_folds if hyperparameter tuning is enabled
    if hyperparameter_tuning and cv_folds < 3:
        print(f"⚠ Increasing CV folds from {cv_folds} to 3 for hyperparameter tuning")
        cv_folds = 3
        use_cv = True
    
    # Train each model
    performance = {}
    trained_models = {}
    best_params_dict = {}
    all_metrics_dict = {}
    
    total_models = len(models_to_train)
    
    print(f"\n🚀 Training {total_models} model(s)...")
    print(f"   Task: {current_task}")
    print(f"   Metric: {selected_metric}")
    print(f"   Hyperparameter Tuning: {'Enabled' if hyperparameter_tuning else 'Disabled'}")
    print(f"   CV Folds: {cv_folds if use_cv else 'Disabled'}\n")
    
    for idx, (name, model) in enumerate(models_to_train.items(), 1):
        if progress_callback:
            progress_callback(idx - 1, total_models, name)
        
        print(f"Training {name}...")
        param_grid = param_grids.get(name, {})
        
        score, cv_score, trained_model, best_params, all_metrics = train_single_model(
            model, X_train, y_train_encoded, X_test, y_test_encoded,
            param_grid, hyperparameter_tuning,
            cv_folds, use_cv, selected_metric,
            current_task, metric_func_map, name
        )
        
        performance[name] = (score, cv_score)
        trained_models[name] = trained_model
        best_params_dict[name] = best_params
        all_metrics_dict[name] = all_metrics
        
        print(f"  Test Score: {score:.4f}")
        if use_cv and not np.isnan(cv_score):
            print(f"  CV Score: {cv_score:.4f}")
        print()
    
    # Final progress update
    if progress_callback:
        progress_callback(total_models, total_models, "")
    
    # Find best model
    best_model_name = max(performance, key=lambda k: performance[k][0])
    print(f"🏆 Best Model: {best_model_name} (Test Score: {performance[best_model_name][0]:.4f})\n")
    
    return performance, best_model_name, use_cv, cv_folds, current_task, trained_models, best_params_dict, all_metrics_dict


def create_performance_dataframe(performance, best_params_dict=None, all_metrics_dict=None):
    """
    Create a DataFrame from performance dictionary
    
    Args:
        performance: dict of model performances
        best_params_dict: optional dict of best parameters from hyperparameter tuning
        all_metrics_dict: optional dict of all metrics per model
        
    Returns:
        pd.DataFrame: Performance dataframe
    """
    perf_data = []
    for model_name, (test_score, cv_score) in performance.items():
        row = {
            "Model": model_name,
            "Test Score": f"{test_score:.4f}",
            "CV Score": f"{cv_score:.4f}" if not np.isnan(cv_score) else "N/A"
        }
        
        # Add overfitting warning if available
        if all_metrics_dict and model_name in all_metrics_dict:
            metrics = all_metrics_dict[model_name]
            if 'overfit_warning' in metrics:
                row["Overfitting"] = metrics['overfit_warning']
        
        # Add best parameters if available
        if best_params_dict and model_name in best_params_dict and best_params_dict[model_name]:
            row["Best Parameters"] = str(best_params_dict[model_name])
        
        perf_data.append(row)
    
    return pd.DataFrame(perf_data)


def create_detailed_metrics_dataframe(all_metrics_dict, task_type):
    """
    Create a detailed DataFrame with all metrics for each model
    
    Args:
        all_metrics_dict: dict containing all metrics for each model
        task_type: 'classification' or 'regression'
        
    Returns:
        pd.DataFrame: Detailed metrics dataframe
    """
    detailed_data = []
    
    for model_name, metrics in all_metrics_dict.items():
        row = {"Model": model_name}
        
        if task_type == "classification":
            row["Train Accuracy"] = f"{metrics.get('train_accuracy', 0):.4f}"
            row["Test Accuracy"] = f"{metrics.get('test_accuracy', 0):.4f}"
            row["Train F1"] = f"{metrics.get('train_f1', 0):.4f}"
            row["Test F1"] = f"{metrics.get('test_f1', 0):.4f}"
            row["Test Precision"] = f"{metrics.get('test_precision', 0):.4f}"
            row["Test Recall"] = f"{metrics.get('test_recall', 0):.4f}"
            
            if 'test_roc_auc' in metrics:
                row["Test ROC-AUC"] = f"{metrics.get('test_roc_auc', 0):.4f}"
        else:
            row["Train R²"] = f"{metrics.get('train_r2', 0):.4f}"
            row["Test R²"] = f"{metrics.get('test_r2', 0):.4f}"
            row["Train MAE"] = f"{metrics.get('train_mae', 0):.4f}"
            row["Test MAE"] = f"{metrics.get('test_mae', 0):.4f}"
            row["Test RMSE"] = f"{metrics.get('test_rmse', 0):.4f}"
            
            if not np.isnan(metrics.get('test_mape', np.nan)):
                row["Test MAPE"] = f"{metrics.get('test_mape', 0):.4f}"
        
        row["Overfit Diff"] = f"{metrics.get('overfit_difference', 0):.4f}"
        row["Status"] = metrics.get('overfit_warning', 'N/A')
        
        detailed_data.append(row)
    
    return pd.DataFrame(detailed_data)


# ==============================
# Clustering Training Functions
# ==============================

def train_clustering_models(X_data, 
                            selected_models_clustering=None,
                            selected_metric_clustering="silhouette",
                            hyperparameter_tuning=False,
                            n_clusters=3,
                            progress_callback=None,
                            kmeans_init='k-means++',
                            dbscan_eps=0.5,
                            dbscan_min_samples=5,
                            dbscan_metric='euclidean'):
    """
    Train clustering models
    
    Args:
        X_data: Feature data (already preprocessed, no target variable)
        selected_models_clustering: List of clustering model names to train
        selected_metric_clustering: Metric to use for evaluation
        hyperparameter_tuning: Whether to use GridSearchCV
        n_clusters: Number of clusters for KMeans (used if hyperparameter_tuning=False)
        progress_callback: Callback function for progress updates
        kmeans_init: KMeans initialization method ('k-means++' or 'random')
        dbscan_eps: DBSCAN epsilon parameter
        dbscan_min_samples: DBSCAN min_samples parameter
        dbscan_metric: DBSCAN distance metric
        
    Returns:
        Tuple of (performance_dict, best_model_name, trained_models, best_params_dict, all_metrics_dict, cluster_labels)
    """
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, SpectralClustering
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    from sklearn.model_selection import GridSearchCV
    
    # Default models if none selected
    if selected_models_clustering is None or len(selected_models_clustering) == 0:
        selected_models_clustering = ["KMeans", "DBSCAN"]
    
    # Available clustering models
    models_clustering = {}
    
    # Create models based on hyperparameter tuning setting
    if not hyperparameter_tuning:
        # Use user-specified parameters
        models_clustering["KMeans"] = KMeans(n_clusters=n_clusters, init=kmeans_init, random_state=42, n_init=10)
        models_clustering["DBSCAN"] = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric=dbscan_metric)
        models_clustering["AgglomerativeClustering"] = AgglomerativeClustering(n_clusters=n_clusters)
        models_clustering["MeanShift"] = MeanShift()
        models_clustering["SpectralClustering"] = SpectralClustering(n_clusters=n_clusters, random_state=42)
    else:
        # Use defaults, GridSearch will find optimal
        models_clustering["KMeans"] = KMeans(random_state=42, n_init=10)
        models_clustering["DBSCAN"] = DBSCAN()
        models_clustering["AgglomerativeClustering"] = AgglomerativeClustering()
        models_clustering["MeanShift"] = MeanShift()
        models_clustering["SpectralClustering"] = SpectralClustering(random_state=42)
    
    # Hyperparameter grids
    param_grids_clustering = {
        "KMeans": {
            'n_clusters': [2, 3, 4, 5, 6],
            'init': ['k-means++', 'random']
        },
        "DBSCAN": {
            'eps': [0.3, 0.5, 0.7, 1.0],
            'min_samples': [3, 5, 7, 10]
        },
        "AgglomerativeClustering": {
            'n_clusters': [2, 3, 4, 5, 6],
            'linkage': ['ward', 'complete', 'average']
        }
    }
    
    # Filter selected models
    models_to_train = {k: v for k, v in models_clustering.items() if k in selected_models_clustering}
    
    performance = {}
    trained_models = {}
    best_params_dict = {}
    all_metrics_dict = {}
    cluster_labels = {}
    
    total_models = len(models_to_train)
    
    for idx, (model_name, model) in enumerate(models_to_train.items(), 1):
        if progress_callback:
            progress_callback(idx - 1, total_models, model_name)
        
        try:
            if hyperparameter_tuning and model_name in param_grids_clustering:
                # Custom scorer for clustering
                from sklearn.metrics import make_scorer
                scorer = make_scorer(silhouette_score)
                
                grid_search = GridSearchCV(
                    model,
                    param_grids_clustering[model_name],
                    cv=3,
                    scoring=scorer,
                    n_jobs=-1
                )
                grid_search.fit(X_data)
                model = grid_search.best_estimator_
                best_params_dict[model_name] = grid_search.best_params_
                labels = model.labels_ if hasattr(model, 'labels_') else model.fit_predict(X_data)
            else:
                # Use model with user-specified or default parameters
                labels = model.fit_predict(X_data)
                best_params_dict[model_name] = {}
            
            # Calculate metrics
            n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1) if -1 in labels else 0
            
            metrics = {
                'n_clusters': n_clusters_found,
                'n_noise_points': n_noise
            }
            
            # Only calculate these metrics if we have at least 2 clusters
            if n_clusters_found >= 2 and n_noise < len(labels):
                try:
                    metrics['silhouette'] = silhouette_score(X_data, labels)
                    metrics['davies_bouldin'] = davies_bouldin_score(X_data, labels)
                    metrics['calinski_harabasz'] = calinski_harabasz_score(X_data, labels)
                except:
                    metrics['silhouette'] = 0
                    metrics['davies_bouldin'] = 0
                    metrics['calinski_harabasz'] = 0
            else:
                metrics['silhouette'] = 0
                metrics['davies_bouldin'] = 0
                metrics['calinski_harabasz'] = 0
            
            # Store results
            performance[model_name] = metrics.get(selected_metric_clustering, metrics['silhouette'])
            trained_models[model_name] = model
            all_metrics_dict[model_name] = metrics
            cluster_labels[model_name] = labels
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
            performance[model_name] = -1
            all_metrics_dict[model_name] = {'error': str(e)}
    
    # Find best model
    if performance:
        best_model_name = max(performance, key=performance.get)
    else:
        best_model_name = None
    
    return performance, best_model_name, trained_models, best_params_dict, all_metrics_dict, cluster_labels


def create_clustering_performance_dataframe(performance, all_metrics_dict, best_params_dict):
    """
    Create a DataFrame with clustering performance metrics
    
    Args:
        performance: dict with model names and scores
        all_metrics_dict: dict with detailed metrics for each model
        best_params_dict: dict with best parameters for each model
        
    Returns:
        pd.DataFrame: Performance dataframe
    """
    perf_data = []
    
    for model_name, score in performance.items():
        metrics = all_metrics_dict.get(model_name, {})
        best_params = best_params_dict.get(model_name, {})
        
        row = {
            "Model": model_name,
            "Silhouette": f"{metrics.get('silhouette', 0):.4f}",
            "Davies-Bouldin": f"{metrics.get('davies_bouldin', 0):.4f}",
            "Calinski-Harabasz": f"{metrics.get('calinski_harabasz', 0):.2f}",
            "N_Clusters": metrics.get('n_clusters', 0),
            "Noise_Points": metrics.get('n_noise_points', 0),
            "Best_Params": str(best_params) if best_params else "Default"
        }
        
        perf_data.append(row)
    
    df = pd.DataFrame(perf_data)
    
    # Sort by silhouette score (higher is better)
    df = df.sort_values("Silhouette", ascending=False)
    
    return df
