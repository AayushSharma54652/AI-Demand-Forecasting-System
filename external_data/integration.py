import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

# Import our modules
from .weather import fetch_weather_for_location
from .holidays import get_holidays_for_period
from .economic import get_economic_features
from .events import get_retail_events_for_period, CustomEventsTracker

class ExternalFactorIntegration:
    """
    Class for integrating external factors into demand forecasting
    """
    def __init__(self, config_path=None):
        """
        Initialize the external factor integration
        
        Parameters:
        -----------
        config_path : str, optional
            Path to a JSON configuration file
        """
        self.config = {
            'weather': {
                'enabled': True,
                'location': 'New York',  # Default location
                'importance': 0.8  # Importance weight for feature selection
            },
            'holidays': {
                'enabled': True,
                'country': 'US',  # Default country
                'importance': 0.9
            },
            'economic': {
                'enabled': True,
                'country': 'US',  # Default country
                'importance': 0.7
            },
            'events': {
                'enabled': True,
                'importance': 0.8
            },
            'custom_events': {
                'enabled': False,
                'events': []
            }
        }
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                
                # Update config with user values
                for category in user_config:
                    if category in self.config:
                        self.config[category].update(user_config[category])
            except Exception as e:
                print(f"Error loading configuration: {e}")
        
        # Initialize custom events tracker
        self.custom_events_tracker = CustomEventsTracker()
        
        # Load any custom events from config
        if self.config['custom_events']['enabled'] and 'events' in self.config['custom_events']:
            for event in self.config['custom_events']['events']:
                if 'name' in event and 'start_date' in event and 'end_date' in event:
                    importance = event.get('importance', 1.0)
                    self.custom_events_tracker.add_event(
                        event['name'], 
                        event['start_date'], 
                        event['end_date'], 
                        importance
                    )
    
    def add_custom_event(self, name, start_date, end_date, importance=1.0):
        """
        Add a custom event
        
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
        self.config['custom_events']['enabled'] = True
        self.custom_events_tracker.add_event(name, start_date, end_date, importance)
    
    def get_integrated_features(self, start_date, end_date):
        """
        Get integrated external features for a specific date range
        
        Parameters:
        -----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with all external factors integrated
        """
        # Create a base DataFrame with the date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        date_range = pd.date_range(start=start_dt, end=end_dt, freq='D')
        
        integrated_df = pd.DataFrame({'date': date_range})
        integrated_df.set_index('date', inplace=True)
        
        # Add weather features if enabled
        if self.config['weather']['enabled']:
            print(f"Fetching weather data for {self.config['weather']['location']}...")
            weather_df = fetch_weather_for_location(
                self.config['weather']['location'],
                start_date,
                end_date
            )
            
            if weather_df is not None:
                # Add weather features to integrated DataFrame
                # Prefix columns with 'weather_' to avoid column name conflicts
                for col in weather_df.columns:
                    integrated_df[f'weather_{col}'] = weather_df[col]
                print(f"Added {len(weather_df.columns)} weather features")
            else:
                print("Warning: No weather data available")
        
        # Add holiday features if enabled
        if self.config['holidays']['enabled']:
            print(f"Fetching holiday data for {self.config['holidays']['country']}...")
            holidays_df = get_holidays_for_period(
                start_date,
                end_date,
                self.config['holidays']['country']
            )
            
            if holidays_df is not None:
                # Add holiday features to integrated DataFrame
                # Prefix columns with 'holiday_' to avoid column name conflicts
                for col in holidays_df.columns:
                    integrated_df[f'holiday_{col}'] = holidays_df[col]
                print(f"Added {len(holidays_df.columns)} holiday features")
            else:
                print("Warning: No holiday data available")
        
        # Add economic features if enabled
        if self.config['economic']['enabled']:
            print(f"Fetching economic data for {self.config['economic']['country']}...")
            economic_df = get_economic_features(
                start_date,
                end_date,
                self.config['economic']['country']
            )
            
            if economic_df is not None:
                # Add economic features to integrated DataFrame
                # Prefix columns with 'econ_' to avoid column name conflicts
                for col in economic_df.columns:
                    integrated_df[f'econ_{col}'] = economic_df[col]
                print(f"Added {len(economic_df.columns)} economic features")
            else:
                print("Warning: No economic data available")
        
        # Add retail event features if enabled
        if self.config['events']['enabled']:
            print("Fetching retail event data...")
            events_df = get_retail_events_for_period(start_date, end_date)
            
            if events_df is not None:
                # Add event features to integrated DataFrame
                # Prefix columns with 'event_' to avoid column name conflicts
                for col in events_df.columns:
                    integrated_df[f'event_{col}'] = events_df[col]
                print(f"Added {len(events_df.columns)} retail event features")
            else:
                print("Warning: No retail event data available")
        
        # Add custom event features if enabled
        if self.config['custom_events']['enabled'] and len(self.custom_events_tracker.events) > 0:
            print("Adding custom event data...")
            custom_events_df = self.custom_events_tracker.get_events_features(start_date, end_date)
            
            if custom_events_df is not None:
                # Add custom event features to integrated DataFrame
                for col in custom_events_df.columns:
                    integrated_df[f'custom_{col}'] = custom_events_df[col]
                print(f"Added {len(custom_events_df.columns)} custom event features")
        
        # Replace NaN values with appropriate defaults
        integrated_df = integrated_df.fillna(0)
        
        return integrated_df
    
    def save_config(self, config_path):
        """
        Save the current configuration to a file
        
        Parameters:
        -----------
        config_path : str
            Path to save the configuration file
        """
        # Update the custom events in the config
        self.config['custom_events']['events'] = []
        for event in self.custom_events_tracker.events:
            self.config['custom_events']['events'].append({
                'name': event['name'],
                'start_date': event['start_date'].strftime('%Y-%m-%d'),
                'end_date': event['end_date'].strftime('%Y-%m-%d'),
                'importance': event['importance']
            })
        
        # Save to file
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            print(f"Configuration saved to {config_path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def get_feature_importances(self):
        """
        Get the importance weights for each category of external factors
        
        Returns:
        --------
        dict
            Dictionary with importance weights for each category
        """
        return {
            'weather': self.config['weather']['importance'] if self.config['weather']['enabled'] else 0,
            'holidays': self.config['holidays']['importance'] if self.config['holidays']['enabled'] else 0,
            'economic': self.config['economic']['importance'] if self.config['economic']['enabled'] else 0,
            'events': self.config['events']['importance'] if self.config['events']['enabled'] else 0,
            'custom_events': 1.0 if self.config['custom_events']['enabled'] else 0
        }