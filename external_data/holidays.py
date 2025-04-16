import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_holidays(country_code, year):
    """
    Fetch public holidays for a specific country and year using Nager.Date API
    
    Parameters:
    -----------
    country_code : str
        ISO country code (e.g., 'US', 'GB', 'CA')
    year : int
        Year to fetch holidays for
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with holiday information
    """
    # Nager.Date API base URL
    url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}"
    
    # Make the API request
    response = requests.get(url)
    
    # Check if request was successful
    if response.status_code != 200:
        print(f"Error fetching holidays: {response.text}")
        return None
    
    # Parse the response
    holidays = response.json()
    
    # If no holidays were found, return None
    if not holidays:
        print(f"No holidays found for {country_code} in {year}")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(holidays)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    return df

def create_holiday_features(start_date, end_date, country_code='US'):
    """
    Create a DataFrame with holiday features for time series modeling
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    country_code : str, optional
        ISO country code, defaults to 'US'
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with holiday indicators for each date
    """
    # Convert dates to datetime
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Create a DataFrame with all dates in the range
    date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
    df = pd.DataFrame({'date': date_range})
    df.set_index('date', inplace=True)
    
    # Initialize holiday columns
    df['is_holiday'] = 0
    df['is_major_holiday'] = 0
    df['days_to_nearest_holiday'] = 999  # Large default value
    
    # Years needed for holiday data
    years_needed = list(range(start_dt.year, end_dt.year + 1))
    
    # Major holidays that typically impact retail/business
    major_holidays = [
        'New Year\'s Day',
        'Christmas Day',
        'Thanksgiving Day',
        'Independence Day',
        'Labor Day',
        'Memorial Day'
    ]
    
    # Collect all holidays across the years
    all_holidays = []
    
    for year in years_needed:
        holidays_df = fetch_holidays(country_code, year)
        if holidays_df is not None:
            all_holidays.append(holidays_df)
    
    if not all_holidays:
        print(f"No holiday data available for {country_code}")
        return df
    
    # Combine holiday DataFrames
    holidays_df = pd.concat(all_holidays)
    
    # Mark holidays in the main DataFrame
    for _, holiday in holidays_df.iterrows():
        holiday_date = holiday['date']
        if holiday_date in df.index:
            df.at[holiday_date, 'is_holiday'] = 1
            
            # Check if it's a major holiday
            if any(major in holiday['name'] for major in major_holidays):
                df.at[holiday_date, 'is_major_holiday'] = 1
    
    # Calculate days to nearest holiday for each date
    for date in df.index:
        if df.at[date, 'is_holiday'] == 1:
            df.at[date, 'days_to_nearest_holiday'] = 0
        else:
            # Find closest holiday
            holiday_dates = holidays_df['date'].values
            days_diff = abs(pd.to_datetime(holiday_dates) - date).astype('timedelta64[D]').astype(int)
            if len(days_diff) > 0:
                df.at[date, 'days_to_nearest_holiday'] = min(days_diff)
    
    # Add features for specific holiday periods
    # Pre-Christmas shopping period (Black Friday to Christmas)
    for year in years_needed:
        black_friday = pd.Timestamp(f"{year}-11-24") + pd.Timedelta(days=(4 - pd.Timestamp(f"{year}-11-24").dayofweek) % 7)
        christmas = pd.Timestamp(f"{year}-12-25")
        
        # Mark the holiday shopping season
        shopping_season = pd.date_range(start=black_friday, end=christmas)
        df['is_holiday_shopping_period'] = 0
        for date in shopping_season:
            if date in df.index:
                df.at[date, 'is_holiday_shopping_period'] = 1
    
    return df

def get_holidays_for_period(start_date, end_date, country_code='US'):
    """
    Get holiday features for a specific date range
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    country_code : str, optional
        ISO country code, defaults to 'US'
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with holiday features
    """
    return create_holiday_features(start_date, end_date, country_code)