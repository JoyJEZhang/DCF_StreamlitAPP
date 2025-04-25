"""Machine learning models for financial forecasting."""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Check for optional ML dependencies
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

def forecast_revenue(historical_data, method='auto', periods=20):
    """
    Forecast revenue using the best available method.
    
    Parameters:
    - historical_data: DataFrame with 'ds' (date) and 'y' (revenue) columns
    - method: 'prophet', 'linear', or 'auto' (uses best available method)
    - periods: Number of periods to forecast
    
    Returns:
    - DataFrame with forecasted values
    """
    if method == 'prophet' and PROPHET_AVAILABLE:
        return _forecast_with_prophet(historical_data, periods)
    elif method == 'linear' or (method == 'auto' and SKLEARN_AVAILABLE):
        return _forecast_with_linear_regression(historical_data, periods)
    else:
        return _forecast_with_simple_growth(historical_data, periods)

def _forecast_with_prophet(historical_data, periods):
    """Use Prophet for forecasting"""
    model = Prophet(yearly_seasonality=True)
    model.fit(historical_data)
    future = model.make_future_dataframe(periods=periods, freq='Q')
    forecast = model.predict(future)
    return forecast

def _forecast_with_linear_regression(historical_data, periods):
    """Use scikit-learn for linear regression forecasting"""
    # Implementation details...
    pass

def _forecast_with_simple_growth(historical_data, periods):
    """Fallback method using simple growth rate"""
    # Implementation details...
    pass

def predict_revenue_growth(historical_data, methods=['linear', 'random_forest', 'arima']):
    """
    Predict revenue growth rate using multiple ML models and compare results.
    
    Parameters:
    - historical_data: DataFrame with 'date' and 'revenue' columns
    - methods: List of methods to use (from: 'linear', 'random_forest', 'arima')
    
    Returns:
    - Dictionary with model predictions and evaluation metrics
    """
    results = {}
    
    # Create feature set (could include seasonality, trend indicators, etc.)
    features_df = _prepare_features(historical_data)
    
    # Store original data
    results['historical_data'] = historical_data
    
    # Run different models based on availability
    if 'linear' in methods and SKLEARN_AVAILABLE:
        linear_result = _predict_with_linear_regression(features_df)
        results['linear'] = linear_result
    
    if 'random_forest' in methods and SKLEARN_AVAILABLE:
        rf_result = _predict_with_random_forest(features_df)
        results['random_forest'] = rf_result
    
    if 'arima' in methods and STATSMODELS_AVAILABLE:
        arima_result = _predict_with_arima(historical_data)
        results['arima'] = arima_result
    
    # Calculate consensus prediction (average of available models)
    available_predictions = [results[m]['predicted_growth'] for m in methods if m in results]
    if available_predictions:
        results['consensus'] = {
            'predicted_growth': sum(available_predictions) / len(available_predictions),
            'model_weights': {m: 1/len(available_predictions) for m in methods if m in results}
        }
    else:
        # Fallback to simple historical average if no models are available
        growth_rates = _calculate_historical_growth_rates(historical_data)
        results['consensus'] = {
            'predicted_growth': np.mean(growth_rates),
            'model_weights': {'historical_average': 1.0}
        }
    
    return results

def _prepare_features(data):
    """Prepare features for ML models"""
    df = data.copy()
    
    # Convert date to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # Create time-based features
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    
    # Create lagged features (previous periods)
    for i in range(1, 5):
        df[f'revenue_lag_{i}'] = df['revenue'].shift(i)
    
    # Calculate rolling average and standard deviation
    df['revenue_rolling_mean_4'] = df['revenue'].rolling(window=4).mean()
    df['revenue_rolling_std_4'] = df['revenue'].rolling(window=4).std()
    
    # Create growth rate features
    df['quarterly_growth'] = df['revenue'].pct_change()
    df['yearly_growth'] = df['revenue'].pct_change(4)  # Assuming quarterly data
    
    # Drop rows with NaN (created by lag/rolling operations)
    df = df.dropna()
    
    return df

def _predict_with_linear_regression(features_df):
    """Predict growth rate using Linear Regression"""
    # Prepare features and target
    X = features_df.drop(['date', 'revenue', 'quarterly_growth', 'yearly_growth'], axis=1)
    y = features_df['yearly_growth']  # Predict yearly growth rate
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Make prediction on most recent data
    latest_data = X.iloc[-1:].copy()
    latest_data_scaled = scaler.transform(latest_data)
    predicted_growth = model.predict(latest_data_scaled)[0]
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'predicted_growth': float(predicted_growth),
        'model': model,
        'metrics': {
            'mse': mse,
            'r2': r2
        },
        'feature_importance': dict(zip(X.columns, model.coef_))
    }

def _predict_with_random_forest(features_df):
    """Predict growth rate using Random Forest"""
    # Prepare features and target
    X = features_df.drop(['date', 'revenue', 'quarterly_growth', 'yearly_growth'], axis=1)
    y = features_df['yearly_growth']  # Predict yearly growth rate
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make prediction on most recent data
    latest_data = X.iloc[-1:].copy()
    predicted_growth = model.predict(latest_data)[0]
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'predicted_growth': float(predicted_growth),
        'model': model,
        'metrics': {
            'mse': mse,
            'r2': r2
        },
        'feature_importance': dict(zip(X.columns, model.feature_importances_))
    }

def _predict_with_arima(data):
    """Predict growth rate using ARIMA time series model"""
    # Calculate growth rates from revenue
    growth_rates = _calculate_historical_growth_rates(data)
    
    # Create a pandas Series with datetime index for ARIMA
    growth_series = pd.Series(growth_rates, index=data['date'].iloc[1:].values)
    
    # Fit ARIMA model - p, d, q parameters should be determined through analysis
    # Here we use a simple (1,1,1) model as an example
    model = ARIMA(growth_series, order=(1,1,1))
    model_fit = model.fit()
    
    # Forecast next period growth rate
    forecast = model_fit.forecast(steps=1)
    predicted_growth = forecast[0]
    
    # Calculate AIC and BIC as metrics
    aic = model_fit.aic
    bic = model_fit.bic
    
    return {
        'predicted_growth': float(predicted_growth),
        'model': model_fit,
        'metrics': {
            'aic': aic,
            'bic': bic
        }
    }

def _calculate_historical_growth_rates(data):
    """Calculate year-over-year growth rates from historical revenue data"""
    # Assuming data is sorted by date
    df = data.copy()
    df['growth_rate'] = df['revenue'].pct_change(4)  # Assuming quarterly data, for yearly growth
    return df['growth_rate'].dropna().values

def get_recommended_growth_rate(historical_data):
    """
    Get a recommended growth rate based on ML model predictions.
    Returns a growth rate as percentage and model details.
    """
    # Get predictions from all available models
    results = predict_revenue_growth(historical_data)
    
    # Extract consensus prediction
    if 'consensus' in results:
        growth_rate = results['consensus']['predicted_growth']
        # Convert to percentage for display
        growth_percentage = growth_rate * 100
        
        # Get model details for display
        models_used = list(results['consensus']['model_weights'].keys())
        
        return {
            'growth_percentage': growth_percentage,
            'models_used': models_used,
            'all_results': results
        }
    else:
        # Fallback to average historical growth
        historical_avg = np.mean(_calculate_historical_growth_rates(historical_data)) * 100
        return {
            'growth_percentage': historical_avg,
            'models_used': ['historical_average'],
            'all_results': {'historical_average': {'predicted_growth': historical_avg/100}}
        }
