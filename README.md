# AI-Powered Demand Forecasting System

A sophisticated demand forecasting platform that leverages machine learning and external data factors to predict future demand patterns with high accuracy.

## Features

- **Multi-model forecasting engine**: Automatically selects the best model from ARIMA, Prophet, Random Forest, Gradient Boosting, Linear Regression, and Ridge Regression
- **Interactive web interface**: Upload data, configure parameters, and visualize results
- **External factor integration**: Incorporates weather data, holidays, economic indicators, and retail events to improve forecast accuracy
- **Comprehensive analytics**: Statistical analysis, feature importance, and seasonal decomposition
- **Export functionality**: Download results in CSV, Excel, and PDF formats

![Forecast Dashboard](static/images/dashboard.png)

## Project Structure

```
├── app.py                  # Main Flask application
├── models/
│   ├── preprocessing.py    # Data preprocessing utilities
│   ├── forecasting.py      # Implementation of forecasting models
│   └── evaluation.py       # Metrics and evaluation functions
├── external_data/          # External data integration
│   ├── weather.py          # Weather data from Open-Meteo API
│   ├── holidays.py         # Holiday calendar from Nager.Date API
│   ├── economic.py         # Economic indicators
│   ├── events.py           # Retail events and custom events
│   └── integration.py      # Main integration module
├── templates/              # HTML templates
├── static/                 # CSS, JavaScript, and images
├── uploads/                # Temporary storage for uploaded files
├── external_data_config/   # Configuration for external data sources
└── requirements.txt        # Project dependencies
```

## Installation

1. Clone the repository
   ```
   git clone https://github.com/yourusername/demand-forecasting.git
   cd demand-forecasting
   ```

2. Create and activate a virtual environment
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

4. Install wkhtmltopdf (required for PDF export)
   - **Windows**: Download from [wkhtmltopdf downloads](https://wkhtmltopdf.org/downloads.html)
   - **MacOS**: `brew install wkhtmltopdf`
   - **Linux**: `sudo apt-get install wkhtmltopdf`

## Usage

1. Start the application
   ```
   python app.py
   ```

2. Open your browser and navigate to `http://localhost:5000`

3. Upload your historical demand data (CSV or Excel format)

4. Configure the forecasting parameters:
   - Select date and target columns
   - Set forecast horizon
   - Enable external factors if desired
   - Configure additional options

5. Generate the forecast and explore the results

6. Export the results in your preferred format

## Data Format Requirements

The system requires a dataset with at least:
- A date column with timestamps
- A target column with numerical demand values

Example CSV format:
```
date,demand,region,price
2023-01-01,120,North,10.5
2023-01-02,115,North,10.5
...
```

## External Factors Integration

The system enhances forecast accuracy by incorporating:

1. **Weather Data**: Temperature, precipitation, and extreme weather events
2. **Public Holidays**: Country-specific holidays that impact demand
3. **Economic Indicators**: Unemployment, inflation, consumer confidence, etc.
4. **Retail Events**: Black Friday, holiday shopping periods, etc.
5. **Custom Events**: User-defined events such as promotions or campaigns

## Model Selection

The system evaluates multiple forecasting models and automatically selects the best performer based on accuracy metrics (RMSE, MAE, MAPE, R²).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Flask](https://flask.palletsprojects.com/) web framework
- [Pandas](https://pandas.pydata.org/) for data manipulation
- [Scikit-learn](https://scikit-learn.org/) for machine learning models
- [Prophet](https://facebook.github.io/prophet/) for time series forecasting
- [Plotly](https://plotly.com/) for interactive visualizations
- [Open-Meteo API](https://open-meteo.com/) for weather data
- [Nager.Date API](https://date.nager.at/) for holiday information