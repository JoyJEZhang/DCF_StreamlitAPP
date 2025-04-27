import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import data_fetcher as df
import sys
from datetime import datetime

def get_apple_dcf_data(refresh=False):
       """Get Apple's financial data for DCF model"""
       if refresh:
           return df.fetch_apple_dcf_data()
       else:
           return df.get_apple_dcf_data()
       
def get_revenue_trend_chart():
    """Generate revenue trend and forecast chart based on actual data"""
    # Get historical data from data_fetcher
    hist_data = df.get_hardcoded_apple_revenue_data()
    
    # Prepare yearly data by aggregating quarterly data
    yearly_data = hist_data.copy()
    yearly_data['year'] = yearly_data['date'].dt.year
    yearly_revenue = yearly_data.groupby('year')['revenue'].sum().reset_index()
    
    # Use ML prediction for future years
    ml_results = get_ml_growth_prediction()
    growth_rate = ml_results['growth_percentage'] / 100
    
    # Create arrays for chart
    years = list(range(2019, 2029))
    historical_years = yearly_revenue[yearly_revenue['year'] <= 2023]['year'].tolist()
    historical_revenue = yearly_revenue[yearly_revenue['year'] <= 2023]['revenue'].tolist()
    
    # Fill in missing historical years with None
    full_historical = []
    for year in years:
        if year in historical_years:
            idx = historical_years.index(year)
            full_historical.append(historical_revenue[idx])
        else:
            full_historical.append(None)
    
    # Generate projections
    last_actual_year = 2023
    last_actual_revenue = yearly_revenue[yearly_revenue['year'] == last_actual_year]['revenue'].iloc[0]
    
    projected_revenue = [None] * len(years)
    for i, year in enumerate(years):
        if year >= last_actual_year:
            years_forward = year - last_actual_year
            projected_revenue[i] = last_actual_revenue * ((1 + growth_rate) ** years_forward)
    
    # Create chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years, y=full_historical, mode='lines+markers', 
                             name='Historical Revenue', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=years, y=projected_revenue, mode='lines+markers', 
                             name='Projected Revenue', line=dict(color='green', dash='dash')))
    fig.update_layout(title='Revenue Trend and Forecast (Billions USD)',
                     xaxis_title='Fiscal Year',
                     yaxis_title='Revenue (Billions USD)')
    return fig

def get_fcf_chart(projection_df):
    """Generate free cash flow projection chart"""
    fcf_fig = px.bar(
        projection_df.iloc[1:],  # Skip base year
        x=projection_df.index[1:],
        y='FCF',
        title='Projected Free Cash Flow (Billions USD)'
    )
    return fcf_fig

def get_ev_composition_chart(sum_pv_fcf, pv_terminal_value):
    """Generate enterprise value composition pie chart"""
    labels = ['PV of Projected FCF', 'PV of Terminal Value']
    values = [sum_pv_fcf, pv_terminal_value]
    
    fig = px.pie(values=values, names=labels, title='Enterprise Value Composition')
    return fig

def get_peer_comparison_data(refresh=False):
    """
    Retrieve peer comparison data from Yahoo Finance or cache
    
    Args:
        refresh (bool): If True, force refresh data from Yahoo Finance
        
    Returns:
        pd.DataFrame: Comparison metrics for tech peers
    """
    if refresh:
        # Force fetch from Yahoo Finance
        return df.fetch_peer_comparison_data()
    
    # Try to get from cache first
    try:
        cached_data = df.get_cached_or_default_data()
        # Check if data is recent (within last 7 days)
        last_update = df.get_last_update_time()
        if last_update != "Never (using default data)":
            last_update_date = datetime.strptime(last_update, "%Y-%m-%d %H:%M:%S")
            days_old = (datetime.now() - last_update_date).days
            
            if days_old > 7:
                print("Cached data is more than 7 days old. Refreshing from Yahoo Finance...")
                return df.fetch_peer_comparison_data()
            
        return cached_data
    except Exception as e:
        print(f"Error retrieving cached data: {e}")
        # Fallback to Yahoo Finance
        return df.fetch_peer_comparison_data()

def get_radar_chart(peer_df):
    """Generate radar chart displaying competitive positioning"""
    # Normalize radar chart data - only using 4 metrics now
    radar_metrics = ["Revenue Growth (%)", "Net Margin (%)", "ROE (%)", "EV/EBITDA"]
    radar_df = peer_df[radar_metrics].copy()
    
    # Invert EV/EBITDA so lower values are better
    max_ev_ebitda = radar_df["EV/EBITDA"].max() * 1.1
    radar_df["EV/EBITDA"] = max_ev_ebitda - radar_df["EV/EBITDA"]
    
    # Normalize to 0-100 scale
    for col in radar_df.columns:
        radar_df[col] = 100 * (radar_df[col] - radar_df[col].min()) / (radar_df[col].max() - radar_df[col].min())
    
    # Create radar chart
    fig = go.Figure()
    
    for company in radar_df.index:
        fig.add_trace(go.Scatterpolar(
            r=radar_df.loc[company].values.tolist() + [radar_df.loc[company].values[0]],
            theta=radar_metrics + [radar_metrics[0]],
            fill='toself',
            name=company
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=True
    )
    
    return fig

def style_sensitivity_table(sensitivity_df):
    """Create heatmap style for sensitivity table"""
    def color_scale(val):
        normalized = (val - sensitivity_df.min().min()) / (sensitivity_df.max().max() - sensitivity_df.min().min())
        r, g, b = int(255 * (1 - normalized)), int(255 * normalized), 100
        return f'background-color: rgb({r}, {g}, {b})'
    
    styled_df = sensitivity_df.style.map(color_scale).format("${:.1f}B")
    return styled_df

def get_3d_sensitivity_chart(wacc_values, growth_values, sensitivity_data):
    """Create 3D sensitivity visualization"""
    fig = go.Figure(data=[go.Surface(z=np.array(sensitivity_data), x=wacc_values*100, y=growth_values*100)])
    fig.update_layout(
        title='Enterprise Value Sensitivity',
        scene=dict(
            xaxis_title='WACC (%)',
            yaxis_title='Terminal Growth (%)',
            zaxis_title='Enterprise Value ($B)'
        ),
        width=800,
        height=600
    )
    return fig

def get_ml_growth_prediction(refresh=False):
    """
    Get ML-based revenue growth prediction
    
    Returns:
    - Dictionary with growth percentage and model details
    """
    try:
        # Get historical revenue data
        hist_data = df.get_historical_revenue_data(refresh=refresh)
        print(f"Got historical data with shape: {hist_data.shape}")
        
        # Ensure data format is correct
        if 'date' not in hist_data.columns:
            print("WARNING: 'date' column missing from historical data")
        if 'revenue' not in hist_data.columns:
            print("WARNING: 'revenue' column missing from historical data")
        
        # Import ML models module
        import ml_models as ml
        
        # Get recommended growth rate
        try:
            growth_results = ml.get_recommended_growth_rate(hist_data)
            print(f"ML predicted growth: {growth_results['growth_percentage']:.2f}%")
            return growth_results
        except Exception as e:
            print(f"Error getting recommended growth rate: {e}")
            # Fallback to mock data
            return _get_mock_ml_results()
            
    except Exception as e:
        print(f"Error in ML prediction: {e}")
        # Default fallback values
        return _get_mock_ml_results()

def _get_mock_ml_results():
    """Create mock ML results for display"""
    # Create different model results
    base_growth = 4.0  # Base growth rate 4%
    
    # Create mock data
    all_results = {
        'linear': {'predicted_growth': base_growth * 0.95 / 100},
        'random_forest': {'predicted_growth': base_growth * 1.1 / 100},
        'ridge': {'predicted_growth': base_growth * 0.97 / 100},
        'historical_average': {'predicted_growth': base_growth / 100},
        'consensus': {
            'predicted_growth': base_growth * 1.02 / 100,
            'model_weights': {
                'random_forest': 0.3,
                'ridge': 0.35,
                'linear': 0.25,
                'historical_average': 0.1
            }
        }
    }
    
    return {
        'growth_percentage': base_growth * 1.02,
        'models_used': ['linear', 'random_forest', 'ridge', 'historical_average'],
        'all_results': all_results
    }

def prepare_data_for_models(data):
    """Prepare data for model training, including seasonal features"""
    df = data.copy()
    
    # Add date features
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['quarter'] = df['date'].dt.quarter
    
    # Add seasonality indicators (Apple's Q1 is typically strongest)
    df['is_q1'] = (df['quarter'] == 1).astype(int)
    df['is_q4'] = (df['quarter'] == 4).astype(int)
    
    # Add trend features
    df['time_idx'] = range(len(df))
    
    # Create lag features
    df['revenue_lag1'] = df['revenue'].shift(1)
    df['revenue_lag4'] = df['revenue'].shift(4)  # Year-over-year same quarter
    
    # Calculate year-over-year growth rate
    df['yoy_growth'] = df['revenue'] / df['revenue_lag4'] - 1
    
    return df

def get_model_comparison_chart(ml_results):
    """Generate chart comparing different ML model predictions"""
    # Debug: Print all available models
    all_results = ml_results.get('all_results', {})
    print(f"DEBUG - Available models in all_results: {list(all_results.keys())}")
    
    # Prepare chart data with explicit handling of model names
    models = []
    predictions = []
    
    # Simplified model mapping - only includes models we actually use
    model_mapping = {
        'linear': 'Linear Regression',
        'linear_regression': 'Linear Regression',  # alias handling
        'ridge': 'Ridge Regression',
        'random_forest': 'Random Forest',
        'historical_average': 'Historical Average'
    }
    
    # Check and add all available models using a unified approach
    for key, display_name in model_mapping.items():
        if key in all_results and isinstance(all_results[key], dict) and 'predicted_growth' in all_results[key]:
            # Avoid duplicates (e.g., linear and linear_regression might point to the same model)
            if display_name not in models:
                models.append(display_name)
                predictions.append(all_results[key]['predicted_growth'] * 100)
                print(f"Added {key}: {all_results[key]['predicted_growth']*100:.2f}%")
    
    # Add consensus if available
    if 'consensus' in all_results and 'predicted_growth' in all_results['consensus']:
        models.append('Consensus')
        predictions.append(all_results['consensus']['predicted_growth'] * 100)
        print(f"Added consensus: {all_results['consensus']['predicted_growth']*100:.2f}%")
    
    # Create DataFrame for plotting
    if not models:
        print("WARNING: No models found for comparison chart")
        return px.bar(x=[0], y=[0], title="No model data available")
    
    df = pd.DataFrame({
        'Model': models,
        'Growth Rate': predictions
    })
    
    # Sort by prediction value
    df = df.sort_values('Growth Rate')
    
    # Create bar chart
    fig = px.bar(
        df,
        x='Model',
        y='Growth Rate',
        color='Model',
        title='',
        labels={'Growth Rate': 'Predicted Annual Growth Rate (%)'},
        height=400
    )
    
    # Add consensus line if available
    if 'Consensus' in df['Model'].values:
        consensus_value = df.loc[df['Model'] == 'Consensus', 'Growth Rate'].iloc[0]
        fig.add_shape(
            type='line',
            x0=-0.5,
            x1=len(models) - 0.5,
            y0=consensus_value,
            y1=consensus_value,
            line=dict(color='red', width=2, dash='dash')
        )
    
    # Format axis labels
    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis_title='Predicted Annual Growth Rate (%)',
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=100)
    )
    
    return fig

def get_model_validation_metrics(ml_results):
    """Get and visualize model validation metrics"""
    all_results = ml_results.get('all_results', {})
    
    # Collect all metrics
    metrics_data = []
    
    for model_name, result in all_results.items():
        if model_name not in ['historical_data', 'consensus', 'features_data']:
            if isinstance(result, dict) and 'metrics' in result:
                model_metrics = result['metrics']
                
                # Create metrics row
                for metric_name, metric_value in model_metrics.items():
                    if isinstance(metric_value, (int, float)) and metric_name not in ['feature_importance']:
                        metrics_data.append({
                            'Model': model_name.replace('_', ' ').title(),
                            'Metric': metric_name.upper(),
                            'Value': metric_value
                        })
    
    if not metrics_data:
        return None
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create model validation visualization
    fig = px.bar(
        metrics_df,
        x='Model',
        y='Value',
        color='Metric',
        barmode='group',
        title='Model Validation Metrics Comparison (Linear, Ridge, Random Forest, Historical Average)',
        height=500
    )
    
    return fig

def get_model_forecast_comparison(ml_results, historical_data):
    """Ensure all existing model prediction lines are displayed without hardcoding"""
    # Print debug information
    all_results = ml_results.get('all_results', {})
    available_models = [k for k in all_results.keys() if k not in ['historical_data', 'consensus']]
    print(f"Available models: {available_models}")
    
    # Ensure historical data is processed correctly
    if not isinstance(historical_data, pd.DataFrame) or 'date' not in historical_data.columns:
        print("Warning: Invalid historical data format")
        return None
    
    # Data preprocessing
    historical_data = historical_data.copy()
    historical_data['date'] = pd.to_datetime(historical_data['date'])
    historical_data = historical_data.dropna(subset=['date'])
    historical_data = historical_data.sort_values('date')
    
    # Create chart
    fig = go.Figure()
    
    # Add historical data
    fig.add_trace(go.Scatter(
        x=historical_data['date'],
        y=historical_data['revenue'],
        mode='lines+markers',
        name='Actual Revenue',
        line=dict(color='black', width=2)
    ))
    
    # Set model colors
    model_colors = {
        'linear': '#1f77b4',         # Blue
        'ridge': '#2ca02c',          # Green
        'random_forest': '#ff7f0e',  # Orange
        'historical_average': '#9467bd'  # Purple
    }
    
    # Get the last data point as prediction starting point
    last_date = historical_data['date'].max()
    last_revenue = historical_data.loc[historical_data['date'] == last_date, 'revenue'].iloc[0]
    
    # Add prediction lines for each model
    for model_name in available_models:
        if model_name in all_results:
            display_name = model_name.replace('_', ' ').title() + ' Forecast'
            color = model_colors.get(model_name, '#8c564b')
            
            # Check if there are quarterly predictions
            quarterly_preds = all_results[model_name].get('quarterly_predictions', {})
            
            if quarterly_preds:
                # Use existing quarterly predictions
                dates = []
                values = []
                
                for date_str, value in quarterly_preds.items():
                    try:
                        date = pd.to_datetime(date_str)
                        if pd.notna(date) and pd.notna(value):
                            dates.append(date)
                            values.append(value)
                    except:
                        pass
                
                if dates:
                    # Sort dates
                    date_value_pairs = sorted(zip(dates, values))
                    sorted_dates = [d for d, v in date_value_pairs]
                    sorted_values = [v for d, v in date_value_pairs]
                    
                    # Add prediction line
                    fig.add_trace(go.Scatter(
                        x=[last_date] + sorted_dates,
                        y=[last_revenue] + sorted_values,
                        mode='lines',
                        name=display_name,
                        line=dict(dash='dash', color=color)
                    ))
            else:
                # If no quarterly predictions, create simple predictions based on growth rate
                if 'predicted_growth' in all_results[model_name]:
                    growth_rate = all_results[model_name]['predicted_growth']
                    
                    # Create future dates
                    future_dates = pd.date_range(
                        start=last_date + pd.DateOffset(months=3),
                        periods=5,
                        freq='QE'
                    )
                    
                    # Seasonality factors - simplified to only include models we use
                    seasonal_patterns = {
                        'linear': [1.1, 0.9, 0.85, 1.05],
                        'ridge': [1.1, 0.9, 0.85, 1.05],
                        'random_forest': [1.15, 0.9, 0.85, 1.1],
                        'historical_average': [1.4, 0.8, 0.7, 1.1]
                    }
                    
                    seasonal_factors = seasonal_patterns.get(model_name, [1.0, 1.0, 1.0, 1.0])
                    
                    # Create prediction values
                    future_values = []
                    for i, date in enumerate(future_dates):
                        quarter_idx = (date.month - 1) // 3
                        factor = seasonal_factors[quarter_idx]
                        growth = (1 + growth_rate) ** ((i+1)/4)
                        future_values.append(last_revenue * growth * factor)
                    
                    # Add prediction line
                    fig.add_trace(go.Scatter(
                        x=[last_date] + list(future_dates),
                        y=[last_revenue] + future_values,
                        mode='lines',
                        name=display_name,
                        line=dict(dash='dash', color=color)
                    ))
    
    # Refine chart layout
    fig.update_layout(
        title='Model Forecast Comparison',
        xaxis_title='Quarter',
        yaxis_title='Revenue (Billions USD)',
        height=500,
        legend=dict(orientation='h', y=-0.2)
    )
    
    return fig

def ensure_model_quarterly_predictions(ml_results):
    """Ensure all models have quarterly prediction data"""
    all_results = ml_results.get('all_results', {})
    historical_data = all_results.get('historical_data')
    
    if not isinstance(historical_data, pd.DataFrame):
        return ml_results  # Cannot process
    
    historical_data = historical_data.copy()
    historical_data['date'] = pd.to_datetime(historical_data['date'])
    historical_data = historical_data.sort_values('date')
    
    # Get the last data point as prediction starting point
    last_date = historical_data['date'].max()
    last_revenue = historical_data.loc[historical_data['date'] == last_date, 'revenue'].iloc[0]
    
    # List of models that need to be fixed - only what we're using
    models_to_check = ['linear', 'ridge', 'random_forest', 'historical_average']
    
    for model_name in models_to_check:
        if model_name in all_results:
            # Check if quarterly predictions are missing
            if 'quarterly_predictions' not in all_results[model_name] or not all_results[model_name]['quarterly_predictions']:
                # Get growth rate prediction
                growth_rate = all_results[model_name].get('predicted_growth', 0.03)
                
                # Create quarterly predictions
                future_dates = pd.date_range(
                    start=last_date + pd.DateOffset(months=3),
                    periods=5,
                    freq='QE'
                )
                
                # Different models have slightly different seasonality
                seasonal_patterns = {
                    'linear': [1.1, 0.9, 0.85, 1.05],
                    'ridge': [1.1, 0.9, 0.85, 1.05],
                    'random_forest': [1.15, 0.9, 0.85, 1.1],
                    'historical_average': [1.4, 0.8, 0.7, 1.1]
                }
                
                seasonal_factors = seasonal_patterns.get(model_name, [1.0, 1.0, 1.0, 1.0])
                
                # Create predictions
                quarterly_predictions = {}
                for i, date in enumerate(future_dates):
                    quarter_idx = (date.month - 1) // 3  # 0-3 index
                    seasonal_factor = seasonal_factors[quarter_idx]
                    
                    # Apply growth rate
                    quarters_from_now = i + 1
                    compound_growth = (1 + growth_rate) ** (quarters_from_now / 4)  # Convert to quarterly growth
                    predicted_value = last_revenue * compound_growth * seasonal_factor
                    
                    # Add to prediction dictionary
                    quarterly_predictions[date.strftime('%Y-%m-%d')] = predicted_value
                
                # Update model results
                all_results[model_name]['quarterly_predictions'] = quarterly_predictions
    
    # Update model results
    ml_results['all_results'] = all_results
    return ml_results