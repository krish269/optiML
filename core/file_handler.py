# ==============================
# ./core/file_handler.py
# ==============================
"""
File Handler Module
Handles file upload, reading, and initial validation
"""

import pandas as pd
import streamlit as st


def load_uploaded_file(uploaded_file, sep=','):
    """
    Load and parse an uploaded file
    
    Args:
        uploaded_file: Streamlit uploaded file object
        sep: Separator character for CSV parsing
        
    Returns:
        pd.DataFrame: Loaded dataframe or None if error
    """
    try:
        # Read the file with specified separator
        df = pd.read_csv(uploaded_file, sep=sep, on_bad_lines='warn')
        
        # Check if data was loaded properly
        if df.empty:
            st.error("The uploaded file is empty or couldn't be read properly.")
            return None
        
        return df
        
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.error("Please check if the file format and separator are correct.")
        return None


def detect_separator(uploaded_file):
    """
    Detect the separator in a data file
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        str: Detected separator character
    """
    try:
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
        
        return detected_sep
    except Exception as e:
        st.warning(f"Could not detect separator: {str(e)}. Using comma as default.")
        return ','


def validate_dataframe(df):
    """
    Validate the loaded dataframe
    
    Args:
        df: pandas DataFrame
        
    Returns:
        tuple: (is_valid: bool, message: str)
    """
    if df is None:
        return False, "DataFrame is None"
    
    if df.empty:
        return False, "DataFrame is empty"
    
    if df.shape[0] < 10:
        return False, f"Dataset too small ({df.shape[0]} rows). Need at least 10 rows."
    
    if df.shape[1] < 2:
        return False, "Dataset must have at least 2 columns (features + target)"
    
    return True, "DataFrame is valid"


def get_file_metadata(uploaded_file, df):
    """
    Extract metadata from uploaded file and dataframe
    
    Args:
        uploaded_file: Streamlit uploaded file object
        df: pandas DataFrame
        
    Returns:
        dict: Metadata dictionary
    """
    metadata = {
        'filename': uploaded_file.name,
        'file_extension': uploaded_file.name.split('.')[-1].lower(),
        'file_size_kb': uploaded_file.size / 1024,
        'num_rows': df.shape[0],
        'num_columns': df.shape[1],
        'column_names': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024)
    }
    
    return metadata
