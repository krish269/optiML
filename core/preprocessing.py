# ==============================
# ./core/preprocessing.py
# ==============================
"""
Advanced Preprocessing Module
Enhanced preprocessing with automatic feature detection and transformation
"""

import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, 
    StandardScaler, MinMaxScaler
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def identify_column_types(X):
    """
    Identify different types of columns in the dataset
    
    Args:
        X: pandas DataFrame (features only)
        
    Returns:
        dict: Dictionary containing different column types
    """
    # Identify ID columns:
    #  1. All values are unique (classic integer/string IDs)
    #  2. Nearly all values are unique (>95%) for numeric columns
    #  3. Column name contains common ID-like keywords
    _n = len(X)
    _id_name_pattern = re.compile(
        r'(^|[_\s\-])id([_\s\-]|$)|_id$|^id_|^(cust|customer|user|row|record|account|employee|emp|member|order|trans|transaction|invoice|ticket|case|item|prod|product|sku)[\s_\-]?id',
        re.IGNORECASE
    )
    id_cols = [
        col for col in X.columns
        if (
            X[col].nunique() == _n                               # all unique
            or (X[col].dtype != object and X[col].nunique() / _n > 0.95 and _n > 50)  # near-all unique numeric
            or bool(_id_name_pattern.search(col))               # name looks like an ID
        )
    ]
    
    # Identify numeric and categorical columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=[object, 'category', 'bool']).columns.tolist()
    
    # Separate high cardinality and low cardinality categorical columns
    high_card_cols = [col for col in cat_cols if X[col].nunique() > 50]
    low_card_cols = [col for col in cat_cols if col not in high_card_cols]
    
    return {
        'id_cols': id_cols,
        'num_cols': num_cols,
        'cat_cols': cat_cols,
        'high_card_cols': high_card_cols,
        'low_card_cols': low_card_cols
    }


def select_scaler(X, num_cols):
    """
    Automatically select the best scaler based on data variance
    
    Args:
        X: pandas DataFrame
        num_cols: list of numeric column names
        
    Returns:
        str: Name of selected scaler
    """
    if not num_cols:
        return 'StandardScaler'
    
    # Use StandardScaler if variance is high, MinMaxScaler otherwise
    max_variance = X[num_cols].var().max()
    scaler_choice = 'StandardScaler' if max_variance > 1 else 'MinMaxScaler'
    
    return scaler_choice


def create_preprocessor(X):
    """
    Create a preprocessing pipeline based on column types
    
    Args:
        X: pandas DataFrame (features only)
        
    Returns:
        tuple: (preprocessor, column_info)
    """
    # Identify column types
    col_types = identify_column_types(X)
    
    # Remove ID columns
    if col_types['id_cols']:
        X = X.drop(columns=col_types['id_cols'])
        # Update column types after dropping ID columns
        col_types = identify_column_types(X)
    
    # Select appropriate scaler
    scaler_choice = select_scaler(X, col_types['num_cols'])
    
    # Build transformers list
    transformers = []
    
    # Numeric pipeline
    if col_types['num_cols']:
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler() if scaler_choice == 'StandardScaler' else MinMaxScaler())
        ])
        transformers.append(('num', num_pipeline, col_types['num_cols']))
    
    # Low cardinality categorical pipeline (OneHotEncoder)
    if col_types['low_card_cols']:
        cat_low_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        transformers.append(('cat_low', cat_low_pipeline, col_types['low_card_cols']))
    
    # High cardinality categorical pipeline (OrdinalEncoder)
    if col_types['high_card_cols']:
        cat_high_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        transformers.append(('cat_high', cat_high_pipeline, col_types['high_card_cols']))
    
    # Create the column transformer
    preprocessor = ColumnTransformer(transformers, remainder='drop')
    
    # Prepare column info for return
    column_info = {
        'id_cols': col_types['id_cols'],
        'num_cols': col_types['num_cols'],
        'low_card_cols': col_types['low_card_cols'],
        'high_card_cols': col_types['high_card_cols'],
        'scaler': scaler_choice
    }
    
    return preprocessor, column_info


def preprocess_and_split(df, target_col, test_size=0.2, random_state=42):
    """
    Complete preprocessing pipeline with train-test split
    
    Args:
        df: pandas DataFrame
        target_col: name of target column
        test_size: fraction of data for test set
        random_state: random seed
        
    Returns:
        tuple: (X_train_proc, X_test_proc, y_train, y_test, preprocessor, column_info)
    """
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()
    
    # Create preprocessor
    preprocessor, column_info = create_preprocessor(X)
    
    # Remove ID columns from X
    if column_info['id_cols']:
        X = X.drop(columns=column_info['id_cols'])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Remove rows with unknown categorical values from test set
    X_test, y_test, removed_count = remove_unknown_categories(
        X_train, X_test, y_test, column_info
    )
    
    # Fit and transform
    X_train_proc = pd.DataFrame(preprocessor.fit_transform(X_train))
    X_test_proc = pd.DataFrame(preprocessor.transform(X_test))
    
    # Store removal info
    column_info['removed_test_rows'] = removed_count
    
    return X_train_proc, X_test_proc, y_train, y_test, preprocessor, column_info


def remove_unknown_categories(X_train, X_test, y_test, column_info):
    """
    Remove rows from test set that contain unknown categorical values
    
    Args:
        X_train: Training features DataFrame
        X_test: Test features DataFrame
        y_test: Test target Series
        column_info: Dictionary with column information
        
    Returns:
        tuple: (X_test_cleaned, y_test_cleaned, removed_count)
    """
    # Get all categorical columns
    cat_cols = column_info['low_card_cols'] + column_info['high_card_cols']
    
    if not cat_cols:
        return X_test, y_test, 0
    
    # Reset indices to ensure alignment
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    
    # Track which rows to keep
    rows_to_keep = pd.Series([True] * len(X_test), index=X_test.index)
    
    # Check each categorical column
    for col in cat_cols:
        if col in X_test.columns:
            # Get known categories from training set
            train_categories = set(X_train[col].dropna().unique())
            
            # Find rows in test set with unknown categories
            test_values = X_test[col]
            unknown_mask = ~test_values.isin(train_categories) & test_values.notna()
            
            # Mark these rows for removal
            rows_to_keep &= ~unknown_mask
    
    # Count removed rows
    removed_count = (~rows_to_keep).sum()
    
    # Filter test set
    X_test_cleaned = X_test[rows_to_keep].reset_index(drop=True)
    y_test_cleaned = y_test[rows_to_keep].reset_index(drop=True)
    
    return X_test_cleaned, y_test_cleaned, removed_count


def get_preprocessing_summary(column_info):
    """
    Generate a summary of preprocessing steps
    
    Args:
        column_info: dictionary with column information
        
    Returns:
        str: Formatted summary string
    """
    summary = []
    summary.append("### Preprocessing Summary")
    summary.append("")
    
    if column_info['id_cols']:
        summary.append(f"**ID Columns Removed:** {len(column_info['id_cols'])}")
        summary.append(f"  - {', '.join(column_info['id_cols'])}")
        summary.append("")
    
    if column_info['num_cols']:
        summary.append(f"**Numeric Columns:** {len(column_info['num_cols'])}")
        summary.append(f"  - Scaler: {column_info['scaler']}")
        summary.append(f"  - Columns: {', '.join(column_info['num_cols'])}")
        summary.append("")
    
    if column_info['low_card_cols']:
        summary.append(f"**Low Cardinality Categorical:** {len(column_info['low_card_cols'])}")
        summary.append(f"  - Encoding: OneHotEncoder")
        summary.append(f"  - Columns: {', '.join(column_info['low_card_cols'])}")
        summary.append("")
    
    if column_info['high_card_cols']:
        summary.append(f"**High Cardinality Categorical:** {len(column_info['high_card_cols'])}")
        summary.append(f"  - Encoding: OrdinalEncoder")
        summary.append(f"  - Columns: {', '.join(column_info['high_card_cols'])}")
        summary.append("")
    
    if column_info.get('removed_test_rows', 0) > 0:
        summary.append(f"**Test Set Cleanup:**")
        summary.append(f"  - Removed {column_info['removed_test_rows']} rows with unknown categorical values")
        summary.append("")
    
    return "\n".join(summary)
