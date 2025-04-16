"""
AI-Powered Demand Forecasting Models

This package contains modules for data preprocessing, time series forecasting,
and model evaluation. It supports multiple forecasting models including:
- ARIMA
- Prophet
- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting

The system automatically selects the best performing model based on evaluation metrics.
"""

from .preprocessing import preprocess_data, extract_features, normalize_features, create_train_test_split
from .forecasting import train_forecasting_models, get_best_model, predict, save_model, load_model, get_feature_importance
from .evaluation import calculate_metrics, evaluate_forecast, create_forecast_plot, create_feature_importance_plot, create_seasonal_decomposition_plot

__all__ = [
    'preprocess_data',
    'extract_features',
    'normalize_features',
    'create_train_test_split',
    'train_forecasting_models',
    'get_best_model',
    'predict',
    'save_model',
    'load_model',
    'get_feature_importance',
    'calculate_metrics',
    'evaluate_forecast',
    'create_forecast_plot',
    'create_feature_importance_plot',
    'create_seasonal_decomposition_plot'
]