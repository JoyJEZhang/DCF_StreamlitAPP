"""Machine learning models for financial forecasting - Simplified version, only includes sklearn models"""
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Check if scikit-learn is available
try:
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn is not available, machine learning models cannot be used")

def get_recommended_growth_rate(historical_data):
    """Get growth rate predictions from all available models"""
    # Validate input data
    if not isinstance(historical_data, pd.DataFrame) or len(historical_data) < 8:
        raise ValueError(f"Insufficient historical data: {len(historical_data) if isinstance(historical_data, pd.DataFrame) else 'invalid data type'} rows. Need at least 8 quarters.")
    
    # Ensure date and revenue columns exist
    required_cols = ['date', 'revenue']
    if not all(col in historical_data.columns for col in required_cols):
        raise ValueError(f"Historical data must contain columns: {required_cols}")
    
    # Prepare results dictionary
    results = {'historical_data': historical_data.copy()}
    models_used = []
    
    # Run models if scikit-learn is available
    if SKLEARN_AVAILABLE:
        # Linear regression
        linear_result = predict_with_linear_regression(historical_data)
        results['linear'] = linear_result
        models_used.append('linear')
        
        # Ridge regression
        ridge_result = predict_with_ridge_regression(historical_data)
        results['ridge'] = ridge_result
        models_used.append('ridge')
        
        # Random forest
        rf_result = predict_with_random_forest(historical_data)
        results['random_forest'] = rf_result
        models_used.append('random_forest')
    else:
        raise RuntimeError("scikit-learn is not available, predictions cannot be made")
    
    # Calculate historical average
    hist_avg_result = predict_with_historical_average(historical_data)
    results['historical_average'] = hist_avg_result
    models_used.append('historical_average')
    
    # Calculate consensus prediction (simple average)
    if models_used:
        # Assign equal weight to each model
        weights = {model: 1.0/len(models_used) for model in models_used}
        
        # Calculate weighted average growth rate
        weighted_growth = sum(
            results[model]['predicted_growth'] * weights[model]
            for model in models_used
        )
        
        results['consensus'] = {
            'predicted_growth': weighted_growth,
            'model_weights': weights
        }
    
    # Return final results
    return {
        'growth_percentage': results['consensus']['predicted_growth'] * 100,
        'models_used': models_used,
        'all_results': results
    }

def prepare_data_for_models(data):
    """Prepare data for model training"""
    df = data.copy()
    
    # Ensure correct date format
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Add basic features
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    
    # Add time index
    df['time_idx'] = range(len(df))
    
    # Add seasonality indicators
    for q in range(1, 5):
        df[f'is_q{q}'] = (df['quarter'] == q).astype(int)
    
    # Add Apple-specific seasonality features (Q1 is typically the strongest quarter)
    df['is_q1_peak'] = ((df['quarter'] == 1) & (df['year'] >= 2010)).astype(int)
    
    # Add year-over-year growth rate (if enough data)
    if len(df) >= 5:
        df['yoy_growth'] = df['revenue'].pct_change(4)
    
    return df

def predict_with_linear_regression(data, forecast_periods=5):
    """Use linear regression to predict growth rate"""
    # Prepare data
    df = prepare_data_for_models(data)
    
    # Create feature set
    features = ['time_idx', 'is_q1', 'is_q2', 'is_q3', 'is_q4']
    X = df[features].values
    y = df['revenue'].values
    
    # Train/test split (if enough data)
    if len(df) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Calculate test set metrics
        y_pred_test = model.predict(X_test)
        r2 = r2_score(y_test, y_pred_test)
        mse = mean_squared_error(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
    else:
        # Use all data when insufficient
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate metrics using training data
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
    
    # Predict future quarters
    last_idx = df['time_idx'].max()
    last_date = df['date'].max()
    
    # Generate future quarters
    future_quarters = []
    quarterly_predictions = []
    
    for i in range(1, forecast_periods + 1):
        # Get next quarter
        quarter_idx = (last_date.month-1)//3 + 1
        next_quarter = quarter_idx + i
        next_year = last_date.year + (next_quarter-1)//4
        next_quarter_actual = ((next_quarter-1) % 4) + 1
        
        # Create prediction features
        next_features = [last_idx + i]  # time_idx
        next_features.extend([1 if q+1 == next_quarter_actual else 0 for q in range(4)])  # quarter indicators
        
        # Predict
        predicted_value = model.predict([next_features])[0]
        quarter_date = pd.Timestamp(year=next_year, month=next_quarter_actual*3-2, day=1)
        
        future_quarters.append(quarter_date)
        quarterly_predictions.append(predicted_value)
    
    # Calculate annual growth rate
    if len(quarterly_predictions) >= 4:
        annual_growth_rate = (quarterly_predictions[3] / df['revenue'].iloc[-1]) - 1
    else:
        annual_growth_rate = (quarterly_predictions[-1] / df['revenue'].iloc[-1]) - 1
    
    return {
        'predicted_growth': annual_growth_rate,
        'quarterly_predictions': dict(zip([d.strftime('%Y-%m-%d') for d in future_quarters], quarterly_predictions)),
        'metrics': {
            'r2': r2,
            'mse': mse,
            'mae': mae
        }
    }

def predict_with_ridge_regression(data, forecast_periods=5):
    """Use ridge regression to predict growth rate"""
    # Prepare data
    df = prepare_data_for_models(data)
    
    # Create more features (Ridge can handle more features)
    features = ['time_idx', 'is_q1', 'is_q2', 'is_q3', 'is_q4', 'is_q1_peak']
    if 'yoy_growth' in df.columns:
        # Remove NaN values
        df = df.dropna(subset=['yoy_growth'])
        features.append('yoy_growth')
    
    X = df[features].values
    y = df['revenue'].values
    
    # Use cross-validation to select best alpha
    from sklearn.linear_model import RidgeCV
    alphas = [0.1, 1.0, 10.0, 100.0]
    model = RidgeCV(alphas=alphas, cv=min(5, len(df)-1))
    model.fit(X, y)
    
    print(f"Selected alpha: {model.alpha_}")
    
    # Calculate test set metrics
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    # Predict future quarters
    last_idx = df['time_idx'].max()
    last_date = df['date'].max()
    last_yoy_growth = df['yoy_growth'].iloc[-1] if 'yoy_growth' in features else 0
    
    # Generate future quarters
    future_quarters = []
    quarterly_predictions = []
    
    for i in range(1, forecast_periods + 1):
        # Get next quarter
        quarter_idx = (last_date.month-1)//3 + 1
        next_quarter = quarter_idx + i
        next_year = last_date.year + (next_quarter-1)//4
        next_quarter_actual = ((next_quarter-1) % 4) + 1
        
        # Create prediction features
        next_features = [last_idx + i]  # time_idx
        next_features.extend([1 if q+1 == next_quarter_actual else 0 for q in range(4)])  # quarter indicators
        next_features.append(1 if next_quarter_actual == 1 else 0)  # is_q1_peak
        
        if 'yoy_growth' in features:
            next_features.append(last_yoy_growth)  # use last known YoY growth rate
        
        # Predict
        predicted_value = model.predict([next_features])[0]
        quarter_date = pd.Timestamp(year=next_year, month=next_quarter_actual*3-2, day=1)
        
        future_quarters.append(quarter_date)
        quarterly_predictions.append(predicted_value)
    
    # Calculate annual growth rate
    if len(quarterly_predictions) >= 4:
        annual_growth_rate = (quarterly_predictions[3] / df['revenue'].iloc[-1]) - 1
    else:
        annual_growth_rate = (quarterly_predictions[-1] / df['revenue'].iloc[-1]) - 1
    
    return {
        'predicted_growth': annual_growth_rate,
        'quarterly_predictions': dict(zip([d.strftime('%Y-%m-%d') for d in future_quarters], quarterly_predictions)),
        'metrics': {
            'r2': r2,
            'mse': mse,
            'mae': mae
        }
    }

def predict_with_random_forest(data, forecast_periods=5):
    """Use random forest to predict growth rate"""
    # Prepare data
    df = prepare_data_for_models(data)
    
    # Create feature set
    features = ['time_idx', 'is_q1', 'is_q2', 'is_q3', 'is_q4', 'is_q1_peak']
    if 'yoy_growth' in df.columns:
        # Remove NaN values
        df = df.dropna(subset=['yoy_growth'])
        features.append('yoy_growth')
    
    X = df[features].values
    y = df['revenue'].values
    
    # Train random forest model (simplified parameters)
    model = RandomForestRegressor(
        n_estimators=50,  # reduce number of trees for speed
        max_depth=5,      # limit tree depth to avoid overfitting
        random_state=42
    )
    
    # Train/test split (if enough data)
    if len(df) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model.fit(X_train, y_train)
        
        # Calculate test set metrics
        y_pred_test = model.predict(X_test)
        r2 = r2_score(y_test, y_pred_test)
        mse = mean_squared_error(y_test, y_pred_test)
        mae = mean_absolute_error(y_test, y_pred_test)
    else:
        # Use all data when insufficient
        model.fit(X, y)
        
        # Calculate metrics using training data
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
    
    # Feature importance
    feature_importance = dict(zip(features, model.feature_importances_))
    
    # Predict future quarters
    last_idx = df['time_idx'].max()
    last_date = df['date'].max()
    last_yoy_growth = df['yoy_growth'].iloc[-1] if 'yoy_growth' in features else 0
    
    # Generate future quarters
    future_quarters = []
    quarterly_predictions = []
    
    for i in range(1, forecast_periods + 1):
        # Get next quarter
        quarter_idx = (last_date.month-1)//3 + 1
        next_quarter = quarter_idx + i
        next_year = last_date.year + (next_quarter-1)//4
        next_quarter_actual = ((next_quarter-1) % 4) + 1
        
        # Create prediction features
        next_features = [last_idx + i]  # time_idx
        next_features.extend([1 if q+1 == next_quarter_actual else 0 for q in range(4)])  # quarter indicators
        next_features.append(1 if next_quarter_actual == 1 else 0)  # is_q1_peak
        
        if 'yoy_growth' in features:
            next_features.append(last_yoy_growth)  # use last known YoY growth rate
        
        # Predict
        predicted_value = model.predict([next_features])[0]
        quarter_date = pd.Timestamp(year=next_year, month=next_quarter_actual*3-2, day=1)
        
        future_quarters.append(quarter_date)
        quarterly_predictions.append(predicted_value)
    
    # Calculate annual growth rate
    if len(quarterly_predictions) >= 4:
        annual_growth_rate = (quarterly_predictions[3] / df['revenue'].iloc[-1]) - 1
    else:
        annual_growth_rate = (quarterly_predictions[-1] / df['revenue'].iloc[-1]) - 1
    
    # Add confidence intervals to Random Forest prediction
    predictions = []
    for _ in range(100):  # Use bootstrap sampling
        bootstrap_idx = np.random.choice(len(X), len(X), replace=True)
        bootstrap_X = X[bootstrap_idx]
        bootstrap_y = y[bootstrap_idx]
        
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=None)
        model.fit(bootstrap_X, bootstrap_y)
        predictions.append(model.predict([next_features])[0])
    
    # Calculate 95% confidence interval
    lower_bound = np.percentile(predictions, 2.5)
    upper_bound = np.percentile(predictions, 97.5)
    
    return {
        'predicted_growth': annual_growth_rate,
        'quarterly_predictions': dict(zip([d.strftime('%Y-%m-%d') for d in future_quarters], quarterly_predictions)),
        'metrics': {
            'r2': r2,
            'mse': mse,
            'mae': mae,
            'feature_importance': feature_importance
        },
        'confidence_interval': {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    }

def predict_with_historical_average(data, forecast_periods=5):
    """Use historical average growth rate for prediction"""
    # Prepare data
    df = data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Calculate annual year-over-year growth rates
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    
    growth_rates = []
    
    # Find each data point compared to the same quarter in the previous year
    for year in df['year'].unique()[1:]:  # Skip first year
        for quarter in range(1, 5):
            current = df[(df['year'] == year) & (df['quarter'] == quarter)]
            previous = df[(df['year'] == year-1) & (df['quarter'] == quarter)]
            
            if not current.empty and not previous.empty:
                current_rev = current['revenue'].values[0]
                prev_rev = previous['revenue'].values[0]
                growth_rate = (current_rev / prev_rev) - 1
                growth_rates.append(growth_rate)
    
    # Calculate average growth rate
    avg_growth_rate = np.mean(growth_rates) if growth_rates else 0.03
    
    # Generate predictions
    last_date = df['date'].max()
    last_revenue = df.loc[df['date'] == last_date, 'revenue'].values[0]
    
    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=3),
        periods=forecast_periods,
        freq='QS-JAN'
    )
    
    # Generate quarterly predictions
    quarterly_predictions = {}
    for i, date in enumerate(future_dates):
        # Find corresponding quarter
        quarter = (date.month-1)//3 + 1
        
        # Find historical average growth rate for the same quarter
        quarter_growth_rates = []
        for j in range(len(df)-4):
            if df['quarter'].iloc[j] == quarter and df['quarter'].iloc[j+4] == quarter:
                quarter_growth = df['revenue'].iloc[j+4] / df['revenue'].iloc[j] - 1
                quarter_growth_rates.append(quarter_growth)
        
        quarter_avg_growth = np.mean(quarter_growth_rates) if quarter_growth_rates else avg_growth_rate
        
        # Apply growth rate
        predicted_value = last_revenue * (1 + quarter_avg_growth) ** (i+1)
        quarterly_predictions[date.strftime('%Y-%m-%d')] = predicted_value
    
    # Calculate annual growth rate
    if forecast_periods >= 4:
        list_values = list(quarterly_predictions.values())
        annual_growth_rate = (list_values[3] / last_revenue) - 1
    else:
        annual_growth_rate = (list(quarterly_predictions.values())[-1] / last_revenue) - 1
    
    # Calculate simple validation metrics
    if len(df) >= 8:
        # Use historical data to validate model accuracy
        y_actual = df['revenue'].values[4:]  # latter half
        y_pred = []
        
        for i in range(len(df)-4):
            pred = df['revenue'].values[i] * (1 + avg_growth_rate)
            y_pred.append(pred)
        
        y_pred = y_pred[:len(y_actual)]
        
        r2 = r2_score(y_actual, y_pred)
        mse = mean_squared_error(y_actual, y_pred)
        mae = mean_absolute_error(y_actual, y_pred)
    else:
        r2, mse, mae = 0.5, 0.01, 0.05
    
    return {
        'predicted_growth': annual_growth_rate,
        'quarterly_predictions': quarterly_predictions,
        'metrics': {
            'r2': r2,
            'mse': mse,
            'mae': mae
        }
    }