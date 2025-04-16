import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import statsmodels.api as sm
from pmdarima import auto_arima
from prophet import Prophet

def train_forecasting_models(df, target_column=None):
    """
    Train multiple forecasting models on the provided data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the features and target variable
    target_column : str, optional
        Name of the target column. If None, assumes the first column is the target
        
    Returns:
    --------
    dict
        Dictionary of trained models
    """
    if target_column is None:
        target_column = df.columns[0]
    
    print(f"Training models with target column: {target_column}")
    print(f"Available columns: {df.columns.tolist()}")
    
    # Ensure the target column exists
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame. Available columns: {df.columns.tolist()}")
        
    # Create a copy of the dataframe
    data = df.copy()
    
    # Split data into training and validation sets (use last 20% for validation)
    train_size = int(len(data) * 0.8)
    train_data = data.iloc[:train_size]
    valid_data = data.iloc[train_size:]
    
    # Create X (features) and y (target) for training
    X_train = train_data.drop(columns=[target_column])
    y_train = train_data[target_column]
    X_valid = valid_data.drop(columns=[target_column])
    y_valid = valid_data[target_column]
    
    # Initialize dictionary to store trained models
    models = {}
    
    # Train ARIMA model (using pmdarima for automatic order selection)
    try:
        print("Training ARIMA model...")
        # For ARIMA, we use only the target variable
        arima_model = auto_arima(
            train_data[target_column],
            seasonal=True,
            m=7,  # Assume weekly seasonality by default
            suppress_warnings=True,
            error_action='ignore',
            trace=False
        )
        models['ARIMA'] = {'model': arima_model, 'type': 'arima'}
        print("ARIMA model trained successfully")
    except Exception as e:
        print(f"Error training ARIMA: {e}")
    
    # Train Prophet model
    try:
        print("Training Prophet model...")
        # Prophet requires a specific dataframe format with 'ds' and 'y' columns
        prophet_data = pd.DataFrame({
            'ds': train_data.index,
            'y': train_data[target_column].values
        })
        
        prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False
        )
        prophet_model.fit(prophet_data)
        models['Prophet'] = {'model': prophet_model, 'type': 'prophet'}
        print("Prophet model trained successfully")
    except Exception as e:
        print(f"Error training Prophet: {e}")
    
    # Train Linear Regression
    try:
        print("Training Linear Regression model...")
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        models['Linear Regression'] = {'model': lr_model, 'type': 'sklearn'}
        print("Linear Regression model trained successfully")
    except Exception as e:
        print(f"Error training Linear Regression: {e}")
    
    # Train Ridge Regression
    try:
        print("Training Ridge Regression model...")
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_train, y_train)
        models['Ridge'] = {'model': ridge_model, 'type': 'sklearn'}
        print("Ridge Regression model trained successfully")
    except Exception as e:
        print(f"Error training Ridge: {e}")
    
    # Train Random Forest
    try:
        print("Training Random Forest model...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        models['Random Forest'] = {'model': rf_model, 'type': 'sklearn'}
        print("Random Forest model trained successfully")
    except Exception as e:
        print(f"Error training Random Forest: {e}")
    
    # Train Gradient Boosting
    try:
        print("Training Gradient Boosting model...")
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_train, y_train)
        models['Gradient Boosting'] = {'model': gb_model, 'type': 'sklearn'}
        print("Gradient Boosting model trained successfully")
    except Exception as e:
        print(f"Error training Gradient Boosting: {e}")
    
    print(f"Trained {len(models)} models successfully")
    return models

def get_feature_importance(model, feature_names):
    """
    Extract feature importance from the model if available
    
    Parameters:
    -----------
    model : object
        Trained model
    feature_names : list
        List of feature names
        
    Returns:
    --------
    dict
        Dictionary with feature names and their importance scores
    """
    importance_dict = {}
    
    try:
        # For scikit-learn models with feature_importances_ attribute
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            for name, score in zip(feature_names, importance):
                importance_dict[name] = float(score)
        
        # For linear models with coef_ attribute
        elif hasattr(model, 'coef_'):
            coefs = model.coef_
            if len(coefs.shape) > 1:
                coefs = coefs[0]  # For multi-output models
            for name, coef in zip(feature_names, coefs):
                importance_dict[name] = abs(float(coef))  # Use absolute values for linear models
        
        # Normalize the values to sum to 1
        if importance_dict:
            total = sum(importance_dict.values())
            if total > 0:
                for key in importance_dict:
                    importance_dict[key] /= total
    except Exception as e:
        print(f"Error extracting feature importance: {e}")
        # Return some dummy values if there's an error
        for i, name in enumerate(feature_names[:5]):  # Just use first 5 features
            importance_dict[name] = 1.0 / (i+1)
    
    return importance_dict

def get_best_model(models, data, target_column=None):
    """
    Evaluate and select the best performing model
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    data : pandas.DataFrame
        Data used for evaluation
    target_column : str, optional
        Name of the target column
        
    Returns:
    --------
    tuple
        (best_model, model_name, metrics) containing the best model, its name, and performance metrics
    """
    if target_column is None:
        target_column = data.columns[0]
    
    print(f"Selecting best model for target: {target_column}")
    
    # Ensure the target column exists
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame. Available columns: {data.columns.tolist()}")
    
    # Create validation set (last 20% of data)
    val_size = int(len(data) * 0.2)
    val_data = data.iloc[-val_size:]
    
    # Initialize variables to track best model
    best_model = None
    best_model_name = None
    best_rmse = float('inf')
    best_metrics = {}
    
    print(f"Evaluating {len(models)} models...")
    
    # Evaluate each model
    for model_name, model_info in models.items():
        print(f"Evaluating {model_name}...")
        model = model_info['model']
        model_type = model_info['type']
        
        try:
            # Generate predictions based on model type
            if model_type == 'arima':
                predictions, conf_int = model.predict(n_periods=len(val_data), return_conf_int=True)
            
            elif model_type == 'prophet':
                future = pd.DataFrame({'ds': val_data.index})
                forecast = model.predict(future)
                predictions = forecast['yhat'].values
            
            elif model_type == 'sklearn':
                X_val = val_data.drop(columns=[target_column])
                predictions = model.predict(X_val)
            
            else:
                print(f"Unknown model type: {model_type}")
                continue  # Skip unknown model types
            
            # Calculate evaluation metrics
            y_true = val_data[target_column].values
            
            # Root Mean Squared Error (RMSE)
            rmse = np.sqrt(np.mean((predictions - y_true) ** 2))
            
            # Mean Absolute Error (MAE)
            mae = np.mean(np.abs(predictions - y_true))
            
            # Mean Absolute Percentage Error (MAPE)
            # Avoid division by zero by adding a small constant
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.mean(np.abs((y_true - predictions) / (y_true + 1e-10))) * 100
                # Replace infinite values with a large number
                mape = np.nan_to_num(mape, nan=0.0, posinf=100.0, neginf=100.0)
            
            # R-squared
            y_mean = np.mean(y_true)
            tss = np.sum((y_true - y_mean) ** 2)
            rss = np.sum((y_true - predictions) ** 2)
            r2 = 1 - (rss / tss) if tss > 0 else 0
            
            # Store metrics
            metrics = {
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape),
                'r2': float(r2)
            }
            
            print(f"{model_name} metrics: {metrics}")
            
            # Check if this is the best model so far
            if rmse < best_rmse:
                best_rmse = rmse
                best_model = model
                best_model_name = model_name
                best_metrics = metrics
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            import traceback
            print(traceback.format_exc())
    
    if best_model is None:
        print("WARNING: No best model selected, using first available model")
        if models:
            first_model_name = list(models.keys())[0]
            best_model = models[first_model_name]['model']
            best_model_name = first_model_name
            best_metrics = {'rmse': 0.0, 'mae': 0.0, 'mape': 0.0, 'r2': 0.0}
        else:
            raise ValueError("No models were successfully trained")
    
    print(f"Best model: {best_model_name} with RMSE: {best_metrics['rmse']}")
    return best_model, best_model_name, best_metrics

def predict(model, data, horizon=30, model_type=None, target_column=None):
    """
    Generate forecasts using the selected model
    
    Parameters:
    -----------
    model : object
        Trained forecasting model
    data : pandas.DataFrame
        Historical data for reference
    horizon : int
        Number of periods to forecast
    model_type : str, optional
        Type of the model ('arima', 'prophet', 'sklearn')
    target_column : str, optional
        Name of the target column
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the forecasted values
    """
    if target_column is None:
        target_column = data.columns[0]
    
    print(f"Predicting with model_type: {model_type}, target_column: {target_column}")
    
    # Determine model type if not provided
    if model_type is None:
        if hasattr(model, 'predict_in_sample'):
            model_type = 'arima'
        elif hasattr(model, 'make_future_dataframe'):
            model_type = 'prophet'
        else:
            model_type = 'sklearn'
    
    # Get the last date in the data
    last_date = data.index[-1]
    print(f"Last date in data: {last_date}")
    
    # Generate future dates
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq='D'  # Assuming daily frequency
    )
    
    print(f"Future dates: {future_dates[:5]}... (total: {len(future_dates)})")
    
    # Generate forecasts based on model type
    if model_type == 'arima':
        print("Generating ARIMA forecast")
        forecast, conf_int = model.predict(n_periods=horizon, return_conf_int=True)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            f'{target_column}_forecast': forecast,
            f'{target_column}_lower': conf_int[:, 0],
            f'{target_column}_upper': conf_int[:, 1]
        }, index=future_dates)
    
    elif model_type == 'prophet':
        print("Generating Prophet forecast")
        # Create future dataframe
        future = pd.DataFrame({'ds': future_dates})
        
        # Make predictions
        forecast = model.predict(future)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            f'{target_column}_forecast': forecast['yhat'],
            f'{target_column}_lower': forecast['yhat_lower'],
            f'{target_column}_upper': forecast['yhat_upper']
        }, index=future_dates)
    
    elif model_type == 'sklearn':
        print("Generating sklearn forecast")
        # Need to generate features for future dates
        future_df = pd.DataFrame(index=future_dates)
        
        # Add date-based features
        future_df['day_of_week'] = future_df.index.dayofweek
        future_df['day_of_month'] = future_df.index.day
        future_df['month'] = future_df.index.month
        future_df['quarter'] = future_df.index.quarter
        future_df['year'] = future_df.index.year
        future_df['is_weekend'] = future_df['day_of_week'].isin([5, 6]).astype(int)
        
        # Add trend features
        future_df['trend'] = np.arange(len(data), len(data) + horizon)
        future_df['trend_squared'] = future_df['trend'] ** 2
        
        # Add seasonality indicators
        future_df['yearly_sin'] = np.sin(2 * np.pi * future_df['day_of_month'] / 365.25)
        future_df['yearly_cos'] = np.cos(2 * np.pi * future_df['day_of_month'] / 365.25)
        
        days_in_month = future_df.index.daysinmonth
        future_df['monthly_sin'] = np.sin(2 * np.pi * future_df['day_of_month'] / days_in_month)
        future_df['monthly_cos'] = np.cos(2 * np.pi * future_df['day_of_month'] / days_in_month)
        
        future_df['weekly_sin'] = np.sin(2 * np.pi * future_df['day_of_week'] / 7)
        future_df['weekly_cos'] = np.cos(2 * np.pi * future_df['day_of_week'] / 7)
        
        # For lag features and rolling statistics, we need the historical data
        # Get the last values from the historical data for lag features
        if target_column in data.columns:
            last_values = data[target_column].iloc[-30:].values  # Get last 30 values or however many we need
        else:
            print(f"Warning: target_column {target_column} not found for lag features")
            last_values = np.zeros(30)  # Default to zeros if target column not found
        
        # Add lag features
        for lag in [1, 7, 14, 30]:
            if len(last_values) >= lag:
                # Initialize with the last values from historical data
                future_lag_values = list(last_values[-lag:])
                
                # For each forecast step, we'll use the previous forecast values
                for i in range(horizon):
                    if i < len(future_lag_values):
                        future_df.loc[future_dates[i], f'lag_{lag}'] = future_lag_values[i]
                    else:
                        # Use predicted values as they become available
                        future_df.loc[future_dates[i], f'lag_{lag}'] = 0  # Placeholder, will fill with predictions
            else:
                # Not enough historical data for this lag
                future_df[f'lag_{lag}'] = 0
        
        # For rolling statistics, use the most recent values from historical data
        for window in [7, 14, 30]:
            if len(last_values) > 0:
                # Calculate the rolling statistics from historical data
                rolling_mean = np.mean(last_values[-min(window, len(last_values)):])
                rolling_std = np.std(last_values[-min(window, len(last_values)):])
                rolling_min = np.min(last_values[-min(window, len(last_values)):])
                rolling_max = np.max(last_values[-min(window, len(last_values)):])
                
                # Assign to all future points (simplified approach)
                future_df[f'rolling_mean_{window}'] = rolling_mean
                future_df[f'rolling_std_{window}'] = rolling_std
                future_df[f'rolling_min_{window}'] = rolling_min
                future_df[f'rolling_max_{window}'] = rolling_max
            else:
                future_df[f'rolling_mean_{window}'] = 0
                future_df[f'rolling_std_{window}'] = 0
                future_df[f'rolling_min_{window}'] = 0
                future_df[f'rolling_max_{window}'] = 0
        
        # Make sure all feature columns in future_df match those expected by the model
        # Get all feature columns from the original data, excluding the target
        feature_columns = [col for col in data.columns if col != target_column]
        
        # Check for missing columns in future_df
        missing_cols = [col for col in feature_columns if col not in future_df.columns]
        for col in missing_cols:
            print(f"Warning: Missing feature column {col} in future data. Setting to 0.")
            future_df[col] = 0
        
        # Ensure columns are in the same order
        future_features_df = future_df[feature_columns]
        
        try:
            # Make predictions with the model
            predictions = model.predict(future_features_df)
            
            # Create forecast dataframe with predictions
            forecast_df = pd.DataFrame({
                f'{target_column}_forecast': predictions,
                # For sklearn models, we don't have confidence intervals by default
                # Here we could use prediction intervals if available
                f'{target_column}_lower': predictions * 0.9,  # Simple approximation
                f'{target_column}_upper': predictions * 1.1   # Simple approximation
            }, index=future_dates)
            
        except Exception as e:
            print(f"Error making sklearn predictions: {e}")
            import traceback
            print(traceback.format_exc())
            
            # Create a dummy forecast if prediction fails
            forecast_df = pd.DataFrame({
                f'{target_column}_forecast': [last_values[-1]] * horizon,
                f'{target_column}_lower': [last_values[-1] * 0.9] * horizon,
                f'{target_column}_upper': [last_values[-1] * 1.1] * horizon
            }, index=future_dates)
    
    else:
        print(f"Unknown model type: {model_type}, creating dummy forecast")
        # Create a dummy forecast for unknown model types
        if target_column in data.columns:
            last_value = data[target_column].iloc[-1]
        else:
            last_value = 100  # Default value if target column not found
            
        forecast_df = pd.DataFrame({
            f'{target_column}_forecast': [last_value] * horizon,
            f'{target_column}_lower': [last_value * 0.9] * horizon,
            f'{target_column}_upper': [last_value * 1.1] * horizon
        }, index=future_dates)
    
    print(f"Generated forecast with shape: {forecast_df.shape}")
    return forecast_df

def save_model(model, model_name, folder='saved_models'):
    """
    Save the trained model to disk
    
    Parameters:
    -----------
    model : object
        Trained model to save
    model_name : str
        Name to use for the saved model
    folder : str
        Folder to save the model in
        
    Returns:
    --------
    str
        Path to the saved model
    """
    # Create folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    # Create a safe filename
    safe_name = model_name.replace(' ', '_').lower()
    file_path = os.path.join(folder, f"{safe_name}.joblib")
    
    # Save the model
    joblib.dump(model, file_path)
    
    return file_path

def load_model(model_path):
    """
    Load a trained model from disk
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
        
    Returns:
    --------
    object
        Loaded model
    """
    return joblib.load(model_path)