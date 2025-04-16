import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import importlib.util
import sys
import os

# Check if our external data module exists and import it
external_data_exists = importlib.util.find_spec('external_data')
if external_data_exists:
    from external_data.integration import ExternalFactorIntegration
else:
    print("Warning: external_data module not found. External factors will not be included.")
    ExternalFactorIntegration = None

def preprocess_data(df, date_column, target_column, include_external_factors=True):
    """
    Preprocess the input dataframe for time series forecasting
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe containing the raw data
    date_column : str
        The name of the column containing date information
    target_column : str
        The name of the column containing the target values to forecast
    include_external_factors : bool, optional
        Whether to include external factors in the preprocessing
        
    Returns:
    --------
    pandas.DataFrame
        Preprocessed dataframe with date as index and cleaned target values
    """
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Convert date column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])
    
    # Set date as index
    df_copy.set_index(date_column, inplace=True)
    
    # Sort by date
    df_copy.sort_index(inplace=True)
    
    # Handle missing values in target column
    if df_copy[target_column].isnull().any():
        # For time series, forward fill is often used for missing values
        df_copy[target_column] = df_copy[target_column].fillna(method='ffill')
        # If there are still NAs (e.g., at the beginning), use backward fill
        df_copy[target_column] = df_copy[target_column].fillna(method='bfill')
    
    # Handle negative values (if applicable for demand forecasting)
    df_copy[target_column] = df_copy[target_column].clip(lower=0)
    
    # Resample to regular frequency if needed
    # (This assumes daily data - adjust frequency if needed)
    if not df_copy.index.is_monotonic_increasing or df_copy.index.has_duplicates:
        df_copy = df_copy.resample('D').mean()
        df_copy[target_column] = df_copy[target_column].fillna(method='ffill')
    
    # Keep only the target column for simplicity
    df_clean = df_copy[[target_column]].copy()
    
    return df_clean

def extract_features(df, include_external_factors=True, external_config=None):
    """
    Extract time series features for forecasting models
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed dataframe with date as index and target column
    include_external_factors : bool, optional
        Whether to include external factors in feature extraction
    external_config : dict, optional
        Configuration for external factors
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with additional time series features
    """
    # Make a copy of the input dataframe
    df_features = df.copy()
    target_column = df.columns[0]  # Assuming the target is the first column
    
    # Add date-based features
    df_features['day_of_week'] = df_features.index.dayofweek
    df_features['day_of_month'] = df_features.index.day
    df_features['month'] = df_features.index.month
    df_features['quarter'] = df_features.index.quarter
    df_features['year'] = df_features.index.year
    df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
    
    # Add lag features (previous values)
    for lag in [1, 7, 14, 30]:
        col_name = f'lag_{lag}'
        df_features[col_name] = df_features[target_column].shift(lag)
    
    # Add rolling window statistics
    for window in [7, 14, 30]:
        # Rolling mean
        df_features[f'rolling_mean_{window}'] = df_features[target_column].rolling(
            window=window, min_periods=1).mean()
        
        # Rolling standard deviation (for volatility)
        df_features[f'rolling_std_{window}'] = df_features[target_column].rolling(
            window=window, min_periods=1).std()
        
        # Rolling min and max
        df_features[f'rolling_min_{window}'] = df_features[target_column].rolling(
            window=window, min_periods=1).min()
        df_features[f'rolling_max_{window}'] = df_features[target_column].rolling(
            window=window, min_periods=1).max()
    
    # Add trend features
    df_features['trend'] = np.arange(len(df_features))
    df_features['trend_squared'] = df_features['trend'] ** 2
    
    # Add seasonality indicators (using sine and cosine transformations)
    # For yearly seasonality
    df_features['yearly_sin'] = np.sin(2 * np.pi * df_features['day_of_month'] / 365.25)
    df_features['yearly_cos'] = np.cos(2 * np.pi * df_features['day_of_month'] / 365.25)
    
    # For monthly seasonality
    days_in_month = df_features.index.daysinmonth
    df_features['monthly_sin'] = np.sin(2 * np.pi * df_features['day_of_month'] / days_in_month)
    df_features['monthly_cos'] = np.cos(2 * np.pi * df_features['day_of_month'] / days_in_month)
    
    # For weekly seasonality
    df_features['weekly_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['weekly_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
    
    # Add external factors if requested
    if include_external_factors and ExternalFactorIntegration is not None:
        try:
            # Get the date range from the dataframe
            start_date = df_features.index.min().strftime('%Y-%m-%d')
            end_date = df_features.index.max().strftime('%Y-%m-%d')
            
            # Initialize the external factor integration
            config_path = None
            if external_config is not None and isinstance(external_config, str):
                config_path = external_config
            
            external_integration = ExternalFactorIntegration(config_path=config_path)
            
            # Get external features
            external_df = external_integration.get_integrated_features(start_date, end_date)
            
            # Merge external features with main features dataframe
            if external_df is not None and not external_df.empty:
                # Ensure indices match
                external_df = external_df.reindex(df_features.index)
                
                # Add external features to the main dataframe
                for col in external_df.columns:
                    df_features[col] = external_df[col]
                
                print(f"Added {len(external_df.columns)} external factor features")
        
        except Exception as e:
            print(f"Error incorporating external factors: {e}")
            import traceback
            traceback.print_exc()
    
    # Fill any missing values from calculations
    df_features = df_features.fillna(method='ffill').fillna(method='bfill')
    
    return df_features

def normalize_features(features_df, target_column):
    """
    Normalize features for machine learning models
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        Dataframe with extracted features
    target_column : str
        Name of the target column
        
    Returns:
    --------
    tuple
        (normalized_df, scaler) where normalized_df is the dataframe with scaled features
        and scaler is the fitted StandardScaler object
    """
    # Create a copy of the dataframe
    df_norm = features_df.copy()
    
    # Get feature columns (exclude the target)
    feature_columns = [col for col in df_norm.columns if col != target_column]
    
    # Initialize the scaler
    scaler = StandardScaler()
    
    # Fit and transform the features
    df_norm[feature_columns] = scaler.fit_transform(df_norm[feature_columns])
    
    return df_norm, scaler

def create_train_test_split(features_df, target_column, test_size=0.2):
    """
    Split the data into training and testing sets
    
    Parameters:
    -----------
    features_df : pandas.DataFrame
        Dataframe with extracted features
    target_column : str
        Name of the target column
    test_size : float
        Proportion of data to use for testing
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test) for model training and evaluation
    """
    # Create a copy of the dataframe
    df = features_df.copy()
    
    # Determine the split point
    split_idx = int(len(df) * (1 - test_size))
    
    # Split the data
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Create X and y for training and testing
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column]
    
    return X_train, X_test, y_train, y_test