import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Common retail events and shopping periods
RETAIL_EVENTS = {
    # Format: 'event_name': ('start_month', 'start_day', 'end_month', 'end_day', 'fixed_date')
    # fixed_date=True means the event is on the same date each year
    # fixed_date=False means it needs to be calculated (like Black Friday)
    'new_years': (1, 1, 1, 1, True),
    'valentines_day': (2, 14, 2, 14, True),
    'mothers_day': (5, 1, 5, 15, False),  # Second Sunday in May
    'fathers_day': (6, 15, 6, 22, False),  # Third Sunday in June
    'back_to_school': (8, 1, 9, 15, True),
    'halloween': (10, 31, 10, 31, True),
    'black_friday': (11, 23, 11, 30, False),  # Day after Thanksgiving
    'cyber_monday': (11, 26, 11, 30, False),  # Monday after Thanksgiving
    'christmas_shopping': (12, 1, 12, 24, True),
    'post_christmas': (12, 26, 12, 31, True)
}

def calculate_event_dates(year):
    """
    Calculate the dates for common retail events for a specific year
    
    Parameters:
    -----------
    year : int
        Year for which to calculate event dates
        
    Returns:
    --------
    dict
        Dictionary mapping event names to (start_date, end_date) tuples
    """
    event_dates = {}
    
    for event_name, (start_month, start_day, end_month, end_day, fixed_date) in RETAIL_EVENTS.items():
        if fixed_date:
            # Fixed date events
            start = pd.Timestamp(year=year, month=start_month, day=start_day)
            end = pd.Timestamp(year=year, month=end_month, day=end_day)
        else:
            # Variable date events
            if event_name == 'mothers_day':
                # Second Sunday in May
                first_day = pd.Timestamp(year=year, month=5, day=1)
                first_sunday = first_day + pd.Timedelta(days=(6 - first_day.dayofweek) % 7)
                mothers_day = first_sunday + pd.Timedelta(days=7)
                start = mothers_day - pd.Timedelta(days=7)  # One week before
                end = mothers_day
            
            elif event_name == 'fathers_day':
                # Third Sunday in June
                first_day = pd.Timestamp(year=year, month=6, day=1)
                first_sunday = first_day + pd.Timedelta(days=(6 - first_day.dayofweek) % 7)
                fathers_day = first_sunday + pd.Timedelta(days=14)
                start = fathers_day - pd.Timedelta(days=7)  # One week before
                end = fathers_day
            
            elif event_name == 'black_friday':
                # Day after Thanksgiving (4th Thursday in November)
                first_day = pd.Timestamp(year=year, month=11, day=1)
                first_thursday = first_day + pd.Timedelta(days=(3 - first_day.dayofweek) % 7)
                thanksgiving = first_thursday + pd.Timedelta(days=21)
                black_friday = thanksgiving + pd.Timedelta(days=1)
                start = black_friday
                end = black_friday + pd.Timedelta(days=3)  # Include weekend after Black Friday
            
            elif event_name == 'cyber_monday':
                # Monday after Thanksgiving (4th Thursday in November)
                first_day = pd.Timestamp(year=year, month=11, day=1)
                first_thursday = first_day + pd.Timedelta(days=(3 - first_day.dayofweek) % 7)
                thanksgiving = first_thursday + pd.Timedelta(days=21)
                cyber_monday = thanksgiving + pd.Timedelta(days=4)
                start = cyber_monday
                end = cyber_monday + pd.Timedelta(days=1)
            
            else:
                # Default to the fixed dates if the event is not specifically handled
                start = pd.Timestamp(year=year, month=start_month, day=start_day)
                end = pd.Timestamp(year=year, month=end_month, day=end_day)
        
        event_dates[event_name] = (start, end)
    
    return event_dates

def create_events_features(start_date, end_date):
    """
    Create a DataFrame with retail event features for time series modeling
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with retail event indicators for each date
    """
    # Convert dates to datetime
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    
    # Create a DataFrame with all dates in the range
    date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
    df = pd.DataFrame({'date': date_range})
    df.set_index('date', inplace=True)
    
    # Initialize event columns
    for event_name in RETAIL_EVENTS.keys():
        df[f'is_{event_name}'] = 0
        df[f'days_to_{event_name}'] = 999  # Large default value
        df[f'days_from_{event_name}'] = 999  # Large default value
    
    # Years needed for event data
    years_needed = list(range(start_dt.year, end_dt.year + 1))
    
    # Add event indicators for each year
    for year in years_needed:
        event_dates = calculate_event_dates(year)
        
        for event_name, (start_event, end_event) in event_dates.items():
            # Skip events outside our date range
            if end_event < start_dt or start_event > end_dt:
                continue
            
            # Create date range for this event
            event_days = pd.date_range(start=start_event, end=end_event)
            
            # Mark event days
            for day in event_days:
                if day in df.index:
                    df.at[day, f'is_{event_name}'] = 1
            
            # Calculate days to/from event for non-event days
            event_midpoint = start_event + (end_event - start_event) / 2
            
            for day in df.index:
                if df.at[day, f'is_{event_name}'] == 0:
                    # Days to the event midpoint
                    days_diff = (event_midpoint - day).days
                    
                    if days_diff > 0:
                        # Day is before the event
                        if days_diff < df.at[day, f'days_to_{event_name}']:
                            df.at[day, f'days_to_{event_name}'] = days_diff
                    else:
                        # Day is after the event
                        if abs(days_diff) < df.at[day, f'days_from_{event_name}']:
                            df.at[day, f'days_from_{event_name}'] = abs(days_diff)
    
    # Create composite features
    df['is_any_event'] = 0
    df['is_major_shopping_event'] = 0
    
    # Any event indicator
    event_columns = [col for col in df.columns if col.startswith('is_') and col != 'is_any_event' and col != 'is_major_shopping_event']
    for col in event_columns:
        df['is_any_event'] = np.maximum(df['is_any_event'], df[col])
    
    # Major shopping event indicator (Black Friday, Cyber Monday, Christmas shopping)
    major_shopping_events = ['is_black_friday', 'is_cyber_monday', 'is_christmas_shopping']
    for event in major_shopping_events:
        if event in df.columns:
            df['is_major_shopping_event'] = np.maximum(df['is_major_shopping_event'], df[event])
    
    return df

def get_retail_events_for_period(start_date, end_date):
    """
    Get retail event features for a specific date range
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with retail event features
    """
    return create_events_features(start_date, end_date)

# Custom events tracking for user-defined events
class CustomEventsTracker:
    """
    Class for tracking and managing custom events uploaded by users
    """
    def __init__(self):
        self.events = []
    
    def add_event(self, name, start_date, end_date, importance=1.0):
        """
        Add a custom event to the tracker
        
        Parameters:
        -----------
        name : str
            Name of the event
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        importance : float, optional
            Importance score for the event (0.0 to 1.0)
        """
        self.events.append({
            'name': name,
            'start_date': pd.to_datetime(start_date),
            'end_date': pd.to_datetime(end_date),
            'importance': importance
        })
    
    def get_events_features(self, start_date, end_date):
        """
        Get features for custom events in a specific date range
        
        Parameters:
        -----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with custom event features
        """
        # Convert dates to datetime
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Create a DataFrame with all dates in the range
        date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
        df = pd.DataFrame({'date': date_range})
        df.set_index('date', inplace=True)
        
        # Add a column for event count and importance
        df['custom_event_count'] = 0
        df['custom_event_importance'] = 0.0
        
        # Process each event
        for i, event in enumerate(self.events):
            event_name = event['name'].lower().replace(' ', '_')
            col_name = f'custom_event_{event_name}'
            
            # Add a column for this specific event
            df[col_name] = 0
            
            # Get the date range for this event
            event_days = pd.date_range(start=event['start_date'], end=event['end_date'])
            
            # Mark event days
            for day in event_days:
                if day in df.index:
                    df.at[day, col_name] = 1
                    df.at[day, 'custom_event_count'] += 1
                    df.at[day, 'custom_event_importance'] = max(
                        df.at[day, 'custom_event_importance'],
                        event['importance']
                    )
        
        return df