import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Note: FRED API requires an API key, but we're providing some methods
# that use pre-downloaded/cached data for common indicators to avoid API key requirements

# Dictionary of common economic indicators with their values for recent years (simplified data)
# This is a fallback for when API access is not available
CACHED_MONTHLY_INDICATORS = {
    'unemployment_rate': {
        # Monthly data for 2020-2024 (simplified)
        '2020-01': 3.6, '2020-02': 3.5, '2020-03': 4.4, '2020-04': 14.8, '2020-05': 13.3,
        '2020-06': 11.1, '2020-07': 10.2, '2020-08': 8.4, '2020-09': 7.8, '2020-10': 6.9,
        '2020-11': 6.7, '2020-12': 6.7,
        '2021-01': 6.3, '2021-02': 6.2, '2021-03': 6.0, '2021-04': 6.1, '2021-05': 5.8,
        '2021-06': 5.9, '2021-07': 5.4, '2021-08': 5.2, '2021-09': 4.8, '2021-10': 4.5,
        '2021-11': 4.2, '2021-12': 3.9,
        '2022-01': 4.0, '2022-02': 3.8, '2022-03': 3.6, '2022-04': 3.6, '2022-05': 3.6,
        '2022-06': 3.6, '2022-07': 3.5, '2022-08': 3.7, '2022-09': 3.5, '2022-10': 3.7,
        '2022-11': 3.6, '2022-12': 3.5,
        '2023-01': 3.4, '2023-02': 3.6, '2023-03': 3.5, '2023-04': 3.4, '2023-05': 3.7,
        '2023-06': 3.6, '2023-07': 3.5, '2023-08': 3.8, '2023-09': 3.8, '2023-10': 3.9,
        '2023-11': 3.7, '2023-12': 3.7,
        '2024-01': 3.7, '2024-02': 3.9, '2024-03': 3.8, '2024-04': 3.9
    },
    'inflation_rate': {
        # Monthly data for 2020-2024 (simplified)
        '2020-01': 2.5, '2020-02': 2.3, '2020-03': 1.5, '2020-04': 0.3, '2020-05': 0.1,
        '2020-06': 0.6, '2020-07': 1.0, '2020-08': 1.3, '2020-09': 1.4, '2020-10': 1.2,
        '2020-11': 1.2, '2020-12': 1.4,
        '2021-01': 1.4, '2021-02': 1.7, '2021-03': 2.6, '2021-04': 4.2, '2021-05': 5.0,
        '2021-06': 5.4, '2021-07': 5.4, '2021-08': 5.3, '2021-09': 5.4, '2021-10': 6.2,
        '2021-11': 6.8, '2021-12': 7.0,
        '2022-01': 7.5, '2022-02': 7.9, '2022-03': 8.5, '2022-04': 8.3, '2022-05': 8.6,
        '2022-06': 9.1, '2022-07': 8.5, '2022-08': 8.3, '2022-09': 8.2, '2022-10': 7.7,
        '2022-11': 7.1, '2022-12': 6.5,
        '2023-01': 6.4, '2023-02': 6.0, '2023-03': 5.0, '2023-04': 4.9, '2023-05': 4.0,
        '2023-06': 3.0, '2023-07': 3.2, '2023-08': 3.7, '2023-09': 3.7, '2023-10': 3.2,
        '2023-11': 3.1, '2023-12': 3.4,
        '2024-01': 3.1, '2024-02': 3.2, '2024-03': 3.5, '2024-04': 3.4
    },
    'consumer_confidence': {
        # Monthly data for 2020-2024 (simplified)
        '2020-01': 131.6, '2020-02': 132.6, '2020-03': 118.8, '2020-04': 85.7, '2020-05': 85.9,
        '2020-06': 98.3, '2020-07': 91.7, '2020-08': 86.3, '2020-09': 101.3, '2020-10': 101.4,
        '2020-11': 92.9, '2020-12': 87.1,
        '2021-01': 88.9, '2021-02': 90.4, '2021-03': 109.0, '2021-04': 117.5, '2021-05': 120.0,
        '2021-06': 128.9, '2021-07': 125.1, '2021-08': 115.2, '2021-09': 109.8, '2021-10': 111.6,
        '2021-11': 111.9, '2021-12': 115.2,
        '2022-01': 111.1, '2022-02': 105.7, '2022-03': 107.6, '2022-04': 107.3, '2022-05': 106.4,
        '2022-06': 98.4, '2022-07': 95.3, '2022-08': 103.6, '2022-09': 107.8, '2022-10': 102.2,
        '2022-11': 101.4, '2022-12': 108.3,
        '2023-01': 106.0, '2023-02': 103.4, '2023-03': 104.0, '2023-04': 103.7, '2023-05': 102.5,
        '2023-06': 110.1, '2023-07': 114.0, '2023-08': 108.7, '2023-09': 104.3, '2023-10': 99.1,
        '2023-11': 101.0, '2023-12': 108.4,
        '2024-01': 110.9, '2024-02': 105.6, '2024-03': 103.3, '2024-04': 97.5
    },
    'retail_sales_growth': {
        # Monthly data for 2020-2024 (simplified)
        '2020-01': 4.4, '2020-02': 4.5, '2020-03': -5.7, '2020-04': -19.9, '2020-05': -5.6,
        '2020-06': 1.1, '2020-07': 2.7, '2020-08': 2.4, '2020-09': 5.4, '2020-10': 8.5,
        '2020-11': 4.1, '2020-12': 2.9,
        '2021-01': 9.7, '2021-02': 6.3, '2021-03': 27.7, '2021-04': 51.2, '2021-05': 27.6,
        '2021-06': 18.2, '2021-07': 15.3, '2021-08': 14.9, '2021-09': 12.9, '2021-10': 16.3,
        '2021-11': 18.5, '2021-12': 16.9,
        '2022-01': 12.7, '2022-02': 15.7, '2022-03': 7.0, '2022-04': 7.8, '2022-05': 8.1,
        '2022-06': 8.4, '2022-07': 10.0, '2022-08': 9.4, '2022-09': 8.2, '2022-10': 8.0,
        '2022-11': 6.0, '2022-12': 5.9,
        '2023-01': 5.5, '2023-02': 5.4, '2023-03': 2.4, '2023-04': 1.5, '2023-05': 1.0,
        '2023-06': 1.9, '2023-07': 2.3, '2023-08': 2.5, '2023-09': 3.8, '2023-10': 2.9,
        '2023-11': 4.0, '2023-12': 4.8,
        '2024-01': 4.1, '2024-02': 2.9, '2024-03': 3.8, '2024-04': 3.0
    },
    'gdp_growth': {
        # Quarterly data for 2020-2024 (simplified)
        '2020-01': 2.3, '2020-04': -32.9, '2020-07': 33.4, '2020-10': 4.3,
        '2021-01': 6.3, '2021-04': 6.7, '2021-07': 2.3, '2021-10': 6.9,
        '2022-01': -1.6, '2022-04': -0.6, '2022-07': 3.2, '2022-10': 2.9,
        '2023-01': 2.2, '2023-04': 2.1, '2023-07': 4.9, '2023-10': 3.4,
        '2024-01': 1.6, '2024-04': 2.8
    }
}

def get_cached_economic_indicators(start_date, end_date, indicators=None):
    """
    Get cached economic indicators data without requiring API access
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    indicators : list, optional
        List of indicator names to include, if None, includes all available indicators
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with economic indicators
    """
    # Convert dates to datetime
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Create a DataFrame with all dates in the range
    date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
    result_df = pd.DataFrame({'date': date_range})
    result_df.set_index('date', inplace=True)
    
    # If no indicators specified, use all available
    if indicators is None:
        indicators = list(CACHED_MONTHLY_INDICATORS.keys())
    
    # Process each requested indicator
    for indicator in indicators:
        if indicator in CACHED_MONTHLY_INDICATORS:
            # Get the indicator data
            indicator_data = CACHED_MONTHLY_INDICATORS[indicator]
            
            # Create a temporary dataframe for this indicator
            dates = [pd.to_datetime(d + '-01') for d in indicator_data.keys()]
            values = list(indicator_data.values())
            temp_df = pd.DataFrame({indicator: values}, index=dates)
            
            # Forward fill to get daily values from monthly data
            temp_df = temp_df.resample('D').ffill()
            
            # Filter to the requested date range
            temp_df = temp_df[(temp_df.index >= start_dt) & (temp_df.index <= end_dt)]
            
            # Add to the result dataframe
            if not temp_df.empty:
                result_df[indicator] = temp_df[indicator]
            else:
                # If no data available for the period, fill with NaN
                result_df[indicator] = np.nan
        else:
            print(f"Indicator '{indicator}' not found in cached data")
            result_df[indicator] = np.nan
    
    # Fill any missing values
    result_df = result_df.fillna(method='ffill').fillna(method='bfill')
    
    # Add additional derived features
    if 'unemployment_rate' in result_df.columns:
        # Create indicators for high/low unemployment
        result_df['high_unemployment'] = (result_df['unemployment_rate'] > 5.0).astype(int)
    
    if 'inflation_rate' in result_df.columns:
        # Create indicators for high/low inflation
        result_df['high_inflation'] = (result_df['inflation_rate'] > 4.0).astype(int)
    
    if 'consumer_confidence' in result_df.columns:
        # Create indicators for consumer confidence levels
        result_df['high_consumer_confidence'] = (result_df['consumer_confidence'] > 110).astype(int)
        result_df['low_consumer_confidence'] = (result_df['consumer_confidence'] < 90).astype(int)
    
    # Add rolling means where appropriate
    for col in result_df.columns:
        # Skip derived binary indicators
        if not col.startswith('high_') and not col.startswith('low_'):
            result_df[f'{col}_30d_avg'] = result_df[col].rolling(window=30, min_periods=1).mean()
    
    return result_df

def get_economic_features(start_date, end_date, country='US'):
    """
    Get economic indicator features for a specific date range
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    country : str, optional
        Country code, defaults to 'US'
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with economic features
    """
    # For now, we only support US indicators in our cached data
    if country.upper() != 'US':
        print(f"Warning: Only US economic data is available. Using US data instead of {country}")
    
    # Get default economic indicators
    indicators = [
        'unemployment_rate',
        'inflation_rate', 
        'consumer_confidence',
        'retail_sales_growth',
        'gdp_growth'
    ]
    
    return get_cached_economic_indicators(start_date, end_date, indicators)