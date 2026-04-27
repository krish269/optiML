# ==============================
# ./core/__init__.py
# ==============================
"""
OptiMLFlow Core Package
Contains all modular components for the ML pipeline
"""

__version__ = "2.0.0"
__author__ = "OptiMLFlow Team"

# Import key functions for easy access
from .file_handler import load_uploaded_file, validate_dataframe
from .preprocessing import preprocess_and_split, get_preprocessing_summary
from .model_training import (
    train_multiple_models, 
    create_performance_dataframe,
    create_detailed_metrics_dataframe
)
from .chart_generator import (
    plot_performance_comparison,
    plot_feature_distribution,
    plot_data_overview,
    plot_train_vs_test_comparison,
    plot_overfitting_analysis,
    plot_all_metrics_radar,
    plot_metrics_heatmap
)
from .ui_components import render_upload_section, render_training_config

__all__ = [
    'load_uploaded_file',
    'validate_dataframe',
    'preprocess_and_split',
    'get_preprocessing_summary',
    'train_multiple_models',
    'create_performance_dataframe',
    'create_detailed_metrics_dataframe',
    'plot_performance_comparison',
    'plot_feature_distribution',
    'plot_data_overview',
    'plot_train_vs_test_comparison',
    'plot_overfitting_analysis',
    'plot_all_metrics_radar',
    'plot_metrics_heatmap',
    'render_upload_section',
    'render_training_config'
]
