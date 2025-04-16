import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true, y_pred):
    """
    Calculate common evaluation metrics for time series forecasting
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary containing the evaluation metrics
    """
    # Ensure we don't have NaN values
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = np.array(y_true)[mask]
    y_pred = np.array(y_pred)[mask]
    
    # If we have no valid data, return None
    if len(y_true) == 0:
        return None
    
    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Mean Absolute Percentage Error (MAPE)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        # Replace infinite values with a large number
        mape = np.nan_to_num(mape, nan=0.0, posinf=100.0, neginf=100.0)
    
    # R-squared score
    r2 = r2_score(y_true, y_pred)
    
    # Mean Percentage Error (MPE) - to measure bias
    with np.errstate(divide='ignore', invalid='ignore'):
        mpe = np.mean((y_true - y_pred) / (y_true + 1e-10)) * 100
        # Replace infinite values
        mpe = np.nan_to_num(mpe, nan=0.0, posinf=100.0, neginf=100.0)
    
    # Median Absolute Error (MedAE)
    med_ae = np.median(np.abs(y_true - y_pred))
    
    # Return as dictionary
    metrics = {
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'r2': float(r2),
        'mpe': float(mpe),
        'median_ae': float(med_ae)
    }
    
    return metrics



def evaluate_forecast(forecast_df, actual_df=None, target_column=None):
    """
    Evaluate the forecast performance
    
    Parameters:
    -----------
    forecast_df : pandas.DataFrame
        DataFrame containing the forecasted values
    actual_df : pandas.DataFrame, optional
        DataFrame containing the actual values (for backtesting)
    target_column : str, optional
        Name of the target column
        
    Returns:
    --------
    dict
        Dictionary containing the evaluation metrics and statistics
    """
    # Ensure we have valid data
    if forecast_df is None or forecast_df.empty:
        print("Warning: Empty forecast dataframe")
        return {
            'forecast_stats': {
                'mean': 0.0, 'median': 0.0, 'min': 0.0,
                'max': 0.0, 'std': 0.0
            },
            'metrics': None
        }
        
    print(f"Evaluating forecast with shape {forecast_df.shape} and columns {forecast_df.columns.tolist()}")
    
    # Determine column names
    if target_column:
        forecast_col = f"{target_column}_forecast"
    else:
        # Try to find the forecast column
        possible_cols = [col for col in forecast_df.columns if 'forecast' in col.lower()]
        if possible_cols:
            forecast_col = possible_cols[0]
        else:
            forecast_col = forecast_df.columns[0]  # Default to first column
    
    print(f"Using forecast column: {forecast_col}")
    
    # DIRECT ACCESS APPROACH - Extract forecast values directly
    forecast_values = []
    try:
        for i in range(len(forecast_df)):
            try:
                # Try column-based access first
                if forecast_col in forecast_df.columns:
                    value = float(forecast_df[forecast_col].iloc[i])
                else:
                    # Fallback to positional access
                    value = float(forecast_df.iloc[i, 0])
                
                if not pd.isna(value):
                    forecast_values.append(value)
            except:
                # Try an alternative access method if the first fails
                try:
                    value = float(forecast_df.iloc[i, 0])
                    if not pd.isna(value):
                        forecast_values.append(value)
                except:
                    # Skip problematic values
                    pass
    except Exception as e:
        print(f"Error extracting forecast values: {e}")
    
    # Calculate forecast statistics from extracted values
    if forecast_values:
        import numpy as np
        forecast_stats = {
            'mean': float(np.mean(forecast_values)),
            'median': float(np.median(forecast_values)),
            'min': float(np.min(forecast_values)),
            'max': float(np.max(forecast_values)),
            'std': float(np.std(forecast_values)) if len(forecast_values) > 1 else 0.0
        }
    else:
        # Default values if no valid forecast values
        forecast_stats = {
            'mean': 0.0, 'median': 0.0, 'min': 0.0,
            'max': 0.0, 'std': 0.0
        }
    
    # If no actual data is provided, return just the forecast statistics
    if actual_df is None or actual_df.empty:
        return {
            'forecast_stats': forecast_stats,
            'metrics': None
        }
    
    # If actual data is provided, we can calculate evaluation metrics
    if target_column is None:
        # Assume first column is the target
        target_column = actual_df.columns[0]
    
    try:
        # Filter actual data to match forecast period
        actual_subset = actual_df.loc[actual_df.index.isin(forecast_df.index), target_column]
        
        # If we have actual data for the forecast period
        if len(actual_subset) > 0:
            # Create a matching forecast series directly from extracted values
            matching_forecasts = []
            
            for date_idx in actual_subset.index:
                # Find the position of this date in the forecast dataframe
                if date_idx in forecast_df.index:
                    pos = forecast_df.index.get_loc(date_idx)
                    if pos < len(forecast_values):
                        matching_forecasts.append(forecast_values[pos])
                    else:
                        matching_forecasts.append(np.nan)
                else:
                    matching_forecasts.append(np.nan)
            
            # Check for valid pairs
            valid_pairs = [(actual, forecast) for actual, forecast 
                           in zip(actual_subset.values, matching_forecasts) 
                           if not pd.isna(actual) and not pd.isna(forecast)]
            
            if valid_pairs:
                # Unzip the valid pairs
                valid_actuals, valid_forecasts = zip(*valid_pairs)
                
                # Calculate metrics
                metrics = calculate_metrics(
                    np.array(valid_actuals),
                    np.array(valid_forecasts)
                )
            else:
                metrics = None
        else:
            metrics = None
        
    except Exception as e:
        print(f"Error in evaluate_forecast metrics calculation: {e}")
        import traceback
        print(traceback.format_exc())
        metrics = None
        
    return {
        'forecast_stats': forecast_stats,
        'metrics': metrics
    }


def create_forecast_plot(historical_df, forecast_df, target_column=None):
    """
    Create a Plotly figure for visualizing the forecast
    
    Parameters:
    -----------
    historical_df : pandas.DataFrame
        DataFrame containing the historical data
    forecast_df : pandas.DataFrame
        DataFrame containing the forecasted values
    target_column : str, optional
        Name of the target column
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object for the forecast visualization
    """
    if target_column is None:
        # Assume first column is the target
        target_column = historical_df.columns[0]
    
    # Determine forecast column name
    forecast_col = f"{target_column}_forecast" if f"{target_column}_forecast" in forecast_df.columns else forecast_df.columns[0]
    
    # Create a new Plotly figure
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical_df.index,
        y=historical_df[target_column],
        mode='lines',
        name='Historical Data',
        line=dict(color='blue')
    ))
    
    # Add forecast data
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df[forecast_col],
        mode='lines',
        name='Forecast',
        line=dict(color='red')
    ))
    
    # Add confidence intervals if available
    lower_col = f"{target_column}_lower" if f"{target_column}_lower" in forecast_df.columns else None
    upper_col = f"{target_column}_upper" if f"{target_column}_upper" in forecast_df.columns else None
    
    if lower_col and upper_col:
        # Lower bound
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df[lower_col],
            mode='lines',
            name='Lower 95% CI',
            line=dict(color='rgba(255, 0, 0, 0.2)'),
            fill=None
        ))
        
        # Upper bound
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df[upper_col],
            mode='lines',
            name='Upper 95% CI',
            line=dict(color='rgba(255, 0, 0, 0.2)'),
            fill='tonexty'  # Fill between upper and lower bounds
        ))
    
    # Update layout
    fig.update_layout(
        title=f'Demand Forecast for {target_column}',
        xaxis_title='Date',
        yaxis_title='Value',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    return fig

def create_feature_importance_plot(feature_importance):
    """
    Create a Plotly figure for visualizing feature importance
    
    Parameters:
    -----------
    feature_importance : dict
        Dictionary mapping feature names to importance scores
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object for the feature importance visualization
    """
    # Sort features by importance
    if isinstance(feature_importance, dict) and 'features' in feature_importance and 'importance' in feature_importance:
        features = feature_importance['features']
        importance = feature_importance['importance']
    else:
        # Convert to format we need
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features = [f[0] for f in sorted_features]
        importance = [f[1] for f in sorted_features]
    
    # Take top N features for clarity
    N = min(15, len(features))
    features = features[:N]
    importance = importance[:N]
    
    # Create a new Plotly figure
    fig = go.Figure()
    
    # Add bar chart for feature importance
    fig.add_trace(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker=dict(
            color='royalblue',
            line=dict(color='rgba(0, 0, 0, 0.2)', width=1)
        )
    ))
    
    # Update layout
    fig.update_layout(
        title='Feature Importance',
        xaxis_title='Importance Score',
        yaxis_title='Feature',
        height=500,
        margin=dict(l=200, r=20, t=30, b=50)
    )
    
    return fig

def create_seasonal_decomposition_plot(data, target_column=None):
    """
    Create a Plotly figure for visualizing seasonal decomposition
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame containing time series data
    target_column : str, optional
        Name of the target column
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object for the seasonal decomposition visualization
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    if target_column is None:
        # Assume first column is the target
        target_column = data.columns[0]
    
    # Ensure the data is regular (no missing dates)
    data_copy = data.copy()
    
    # Get the frequency of the data
    if data_copy.index.inferred_freq is None:
        # If frequency cannot be inferred, assume it's daily
        freq = pd.infer_freq(data_copy.index)
        if freq is None:
            freq = 'D'
    else:
        freq = data_copy.index.inferred_freq
    
    # Perform the seasonal decomposition
    try:
        # Determine the period based on frequency
        if freq in ['D']:
            period = 7  # Weekly seasonality for daily data
        elif freq in ['M', 'MS']:
            period = 12  # Yearly seasonality for monthly data
        elif freq in ['Q', 'QS']:
            period = 4  # Yearly seasonality for quarterly data
        else:
            period = 7  # Default to 7 for other frequencies
        
        # Decompose the series
        result = seasonal_decompose(data_copy[target_column], model='additive', period=period)
        
        # Create a new Plotly figure with subplots
        fig = go.Figure()
        
        # Add a trace for the original data
        fig.add_trace(go.Scatter(
            x=data_copy.index,
            y=data_copy[target_column],
            mode='lines',
            name='Original',
            line=dict(color='blue')
        ))
        
        # Add a trace for the trend component
        fig.add_trace(go.Scatter(
            x=data_copy.index,
            y=result.trend,
            mode='lines',
            name='Trend',
            line=dict(color='red')
        ))
        
        # Add a trace for the seasonal component
        fig.add_trace(go.Scatter(
            x=data_copy.index,
            y=result.seasonal,
            mode='lines',
            name='Seasonal',
            line=dict(color='green')
        ))
        
        # Add a trace for the residual component
        fig.add_trace(go.Scatter(
            x=data_copy.index,
            y=result.resid,
            mode='lines',
            name='Residual',
            line=dict(color='gray')
        ))
        
        # Update layout
        fig.update_layout(
            title=f'Seasonal Decomposition of {target_column}',
            xaxis_title='Date',
            yaxis_title='Value',
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            )
        )
        
        # Add buttons to show/hide components
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    active=0,
                    x=0.5,
                    y=1.15,
                    buttons=list([
                        dict(
                            label="All Components",
                            method="update",
                            args=[{"visible": [True, True, True, True]}]
                        ),
                        dict(
                            label="Original",
                            method="update",
                            args=[{"visible": [True, False, False, False]}]
                        ),
                        dict(
                            label="Trend",
                            method="update",
                            args=[{"visible": [False, True, False, False]}]
                        ),
                        dict(
                            label="Seasonal",
                            method="update",
                            args=[{"visible": [False, False, True, False]}]
                        ),
                        dict(
                            label="Residual",
                            method="update",
                            args=[{"visible": [False, False, False, True]}]
                        )
                    ]),
                )
            ]
        )
        
        return fig
    
    except Exception as e:
        print(f"Error in seasonal decomposition: {e}")
        return None