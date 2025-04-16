import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_historical_weather(latitude, longitude, start_date, end_date=None):
    """
    Fetch historical weather data from Open-Meteo API
    
    Parameters:
    -----------
    latitude : float
        Latitude of the location
    longitude : float
        Longitude of the location
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format. If None, uses current date.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with daily weather data
    """
    # Convert dates to required format
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    
    if end_date is None:
        end_dt = datetime.now()
    else:
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Format dates for API
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")
    
    # Open-Meteo API base URL
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # Parameters for the API request
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': start_str,
        'end_date': end_str,
        'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,snowfall_sum',
        'timezone': 'auto'
    }
    
    # Make the API request
    response = requests.get(url, params=params)
    
    # Check if request was successful
    if response.status_code != 200:
        print(f"Error fetching weather data: {response.text}")
        return None
    
    # Parse the response
    data = response.json()
    
    # Create DataFrame
    if 'daily' in data:
        df = pd.DataFrame({
            'date': pd.to_datetime(data['daily']['time']),
            'temp_max': data['daily']['temperature_2m_max'],
            'temp_min': data['daily']['temperature_2m_min'],
            'precipitation': data['daily']['precipitation_sum'],
            'rain': data['daily']['rain_sum'],
            'snow': data['daily']['snowfall_sum']
        })
        
        # Set date as index
        df.set_index('date', inplace=True)
        return df
    else:
        print("No daily data found in response")
        return None

def get_weather_features(weather_df):
    """
    Extract useful features from weather data
    
    Parameters:
    -----------
    weather_df : pandas.DataFrame
        DataFrame with weather data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with derived weather features
    """
    if weather_df is None or weather_df.empty:
        return None
    
    # Create a copy of the DataFrame
    df = weather_df.copy()
    
    # Calculate average temperature
    df['temp_avg'] = (df['temp_max'] + df['temp_min']) / 2
    
    # Weather condition flags
    df['is_rainy'] = (df['rain'] > 1.0).astype(int)
    df['is_heavy_rain'] = (df['rain'] > 10.0).astype(int)
    df['is_snowy'] = (df['snow'] > 0).astype(int)
    
    # Temperature change from previous day
    df['temp_change'] = df['temp_avg'].diff()
    
    # Rolling weather features (3-day and 7-day)
    df['temp_avg_3d'] = df['temp_avg'].rolling(window=3, min_periods=1).mean()
    df['temp_avg_7d'] = df['temp_avg'].rolling(window=7, min_periods=1).mean()
    df['precipitation_3d'] = df['precipitation'].rolling(window=3, min_periods=1).sum()
    df['precipitation_7d'] = df['precipitation'].rolling(window=7, min_periods=1).sum()
    
    # Fill any NaN values from calculations
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df

def fetch_weather_for_location(location_name, start_date, end_date=None):
    """
    Fetch weather data for a named location
    
    Parameters:
    -----------
    location_name : str
        Name of the location (city, etc.)
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with weather data and derived features
    """
    # Dictionary of common locations and their coordinates
    # This could be expanded or replaced with a geocoding API
    locations = {
        'new york': (40.7128, -74.0060),
        'los angeles': (34.0522, -118.2437),
        'chicago': (41.8781, -87.6298),
        'houston': (29.7604, -95.3698),
        'phoenix': (33.4484, -112.0740),
        'philadelphia': (39.9526, -75.1652),
        'san antonio': (29.4241, -98.4936),
        'san diego': (32.7157, -117.1611),
        'dallas': (32.7767, -96.7970),
        'san francisco': (37.7749, -122.4194),
        'london': (51.5074, -0.1278),
        'paris': (48.8566, 2.3522),
        'berlin': (52.5200, 13.4050),
        'tokyo': (35.6762, 139.6503),
        'sydney': (-33.8688, 151.2093),
        'toronto': (43.6532, -79.3832),
        'mumbai': (19.0760, 72.8777),
        'shanghai': (31.2304, 121.4737),
        'mexico city': (19.4326, -99.1332),
        'cairo': (30.0444, 31.2357)
    }
    
    # Convert location name to lowercase for case-insensitive matching
    location_name_lower = location_name.lower()
    
    # Check if location is in our dictionary
    if location_name_lower in locations:
        lat, lon = locations[location_name_lower]
    else:
        print(f"Location '{location_name}' not found in database. Using a default location (New York).")
        lat, lon = 40.7128, -74.0060  # Default to New York
    
    # Fetch the weather data
    weather_df = fetch_historical_weather(lat, lon, start_date, end_date)
    
    # Extract features
    if weather_df is not None:
        return get_weather_features(weather_df)
    else:
        return None