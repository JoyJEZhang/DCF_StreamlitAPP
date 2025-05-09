import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Import custom modules
import data_processor as dp
import dcf_model as dcf
import data_fetcher as df
import ml_models

# Streamlit is a Python framework that makes it easy to create web apps for data science
# Key features used in this app:
# - st.set_page_config(): Configures page layout and title
# - st.sidebar: Creates a sidebar for navigation
# - st.columns(): Creates multi-column layouts
# - st.metric(): Displays key metrics with optional delta values
# - st.plotly_chart(): Embeds interactive Plotly charts
# - st.dataframe(): Shows pandas DataFrames
# - st.expander(): Creates collapsible sections


# Initialize global financial data
def get_financial_data():
    """Get financial data once to use throughout the app"""
    return dp.get_apple_dcf_data()

# Get data at startup
apple_data = get_financial_data()

# Page configuration
st.set_page_config(page_title="Advanced DCF Valuation Dashboard", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Section", ["Executive Summary", "Valuation Model", "Sensitivity Analysis", "Industry Comparison"])

# Common financial data
company_name = "Apple Inc."
ticker = "AAPL"
current_year = 2023
projection_years = 5

# Header styling
def section_header(title):
    st.markdown(f"<h2 style='color:#1E88E5;'>{title}</h2>", unsafe_allow_html=True)

# A. Executive Summary page

    # - Displays key metrics (stock price, market cap, P/E ratio)
    # - Shows investment thesis
    # - Visualizes revenue trends

if page == "Executive Summary":
    st.title(f"Value Engineering Dashboard - {company_name} ({ticker})")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # Get current and previous day's price
        current_price = apple_data['current_price']
        previous_price = apple_data.get('previous_price', current_price * 0.958)  # Default to -4.2% if not available
        price_change = ((current_price - previous_price) / previous_price) * 100
        st.metric("Current Stock Price", f"${current_price:.2f}", f"{price_change:.1f}%")
    with col2:
        st.metric("Market Cap", "$2.98T", "5.7%")
    with col3:
        st.metric("P/E Ratio", "31.5", "-2.1%")
    with col4:
        st.metric("EV/EBITDA", "24.8", "0.5%")
    
    # Overview
    st.markdown("""
    ### Investment Thesis
    This dashboard provides a comprehensive valuation analysis of Apple Inc. using advanced DCF methodology 
    with scenario testing and sensitivity analysis. The model incorporates multi-year projections, industry 
    benchmarking, and return on investment calculations to deliver strategic insights for investment decisions.
    """)
    
    # Revenue trend visualization
    with st.container():
        section_header("Historical Performance vs Projections")
        fig = dp.get_revenue_trend_chart()
        st.plotly_chart(fig, use_container_width=True)

#B. Valuation Model page
    # - Collects user inputs for DCF parameters
    # - Runs DCF calculations
    # - Shows financial projections
    # - Displays valuation results
elif page == "Valuation Model":
    st.title(f"Multi-Year DCF Valuation Model - {company_name}")
    
    st.markdown("""
    This model projects cash flows for the next 5 years and calculates enterprise value using the 
    Discounted Cash Flow method based on FY 2023 financial data as the baseline.
    """)
    
    # Add refresh button for Apple financial data
    col1, col2 = st.columns([3, 1])
    with col1:
        if 'df' in dir(dp) and hasattr(dp.df, 'get_last_update_time'):
            st.info(f"Last data update: {dp.df.get_last_update_time()}")
    with col2:
        refresh_apple_data = st.button("🔄 Refresh Apple Data")
    
    # Get Apple's financial data (refresh if button clicked)
    apple_data = dp.get_apple_dcf_data(refresh=refresh_apple_data)
    
    if refresh_apple_data:
        st.success("Apple financial data updated from Yahoo Finance!")
    
    # New feature: ML-based growth rate prediction
    with st.expander("🧠 ML-based Growth Rate Prediction", expanded=True):
        ml_col1, ml_col2 = st.columns([3, 1])
        with ml_col2:
            refresh_ml = st.button("🔄 Refresh ML Prediction")
        
        # Get ML prediction results
        ml_results = dp.get_ml_growth_prediction(refresh=refresh_ml)
        
        # Display ML prediction
        st.info(f"🤖 **ML Consensus Prediction:** {ml_results['growth_percentage']:.2f}% annual revenue growth rate")
        st.caption(f"Based on: {', '.join(ml_results['models_used'])}")
        
        # Use tabs to display charts
        ml_tab1, ml_tab2, ml_tab3 = st.tabs(["Model Comparison", "Model Validation", "Historical Data"])
        
        with ml_tab1:
            # Display chart comparing different model predictions
            st.subheader("Revenue Growth Rate Predictions by Model")
            model_fig = dp.get_model_comparison_chart(ml_results)
            st.plotly_chart(model_fig, use_container_width=True)
            
            # Replace entire weight information and visualization section
            if 'all_results' in ml_results and 'consensus' in ml_results['all_results']:
                weights = ml_results['all_results']['consensus'].get('model_weights', {})
                if weights:
                    # Create weight data
                    weight_df = pd.DataFrame({
                        'Model': [k.replace('_', ' ').title() for k in weights.keys()],
                        'Weight': [v*100 for v in weights.values()]
                    })
                    
                    # Plot weight chart
                    weight_fig = px.pie(weight_df, values='Weight', names='Model', 
                                        title='Model Weights in Consensus Calculation',
                                        hole=0.4)
                    st.plotly_chart(weight_fig, use_container_width=True)
                    
                    # Add calculation formula example
                    st.markdown("#### Consensus Calculation")
                    
                    # Create formula section
                    formula_parts = []
                    all_results = ml_results.get('all_results', {})
                    
                    for model_name, weight in weights.items():
                        if model_name in all_results and 'predicted_growth' in all_results[model_name]:
                            pred_value = all_results[model_name]['predicted_growth'] * 100
                            formula_parts.append(f"({model_name.replace('_', ' ').title()}: {pred_value:.1f}% × {weight*100:.1f}%)")
                    
                    formula = "Consensus = " + " + ".join(formula_parts)
                    st.markdown(f"```\n{formula}\n```")
            
            # Add user-adjustable weight section
            st.markdown("#### Customize Model Weights")
            st.caption("Adjust the importance of each model in the consensus calculation")
            
            custom_weights = {}
            cols = st.columns(len(weights))
            
            for i, (model, default_weight) in enumerate(weights.items()):
                with cols[i]:
                    custom_weights[model] = st.slider(
                        f"{model.replace('_', ' ').title()}", 
                        min_value=0, 
                        max_value=100, 
                        value=int(default_weight*100),
                        step=5
                    ) / 100
            
            # Normalize custom weights
            total = sum(custom_weights.values())
            if total > 0:
                custom_weights = {k: v/total for k, v in custom_weights.items()}
            
            # Calculate custom weighted consensus
            if st.button("Recalculate Consensus"):
                custom_consensus = sum(
                    ml_results['all_results'][model]['predicted_growth'] * custom_weights.get(model, 0) 
                    for model in custom_weights.keys() if model in ml_results['all_results']
                )
                st.success(f"Custom weighted consensus: {custom_consensus*100:.2f}% annual growth rate")
        
        with ml_tab2:
            st.subheader("Model Performance Evaluation")
            
            # Use a single column for now to simplify layout
            st.markdown("#### Model Selection Guidance")
            st.markdown("""
            **How to interpret metrics:**
            
            - **R²**: Higher is better (max 1.0). Measures how well the model fits historical data.
            - **MSE/RMSE**: Lower is better. Measures prediction error magnitude.
            - **MAE**: Lower is better. Average absolute prediction error.
            - **AIC/BIC**: Lower is better. Balance between fit and complexity.
            
            **When to use each model:**
            
            - **Linear Regression**: 
              • Best for capturing simple linear growth trends
              • Performs well with limited data
              • Easy to interpret and communicate results
              • Provides conservative predictions, less prone to extreme forecasts

            - **Ridge Regression**: 
              • Better than linear regression when dealing with multiple correlated features
              • Prevents overfitting through regularization
              • More robust when features outnumber observations
              • Shrinks coefficients to produce more stable predictions

            - **Random Forest**: 
              • Captures complex non-linear relationships
              • Automatically handles feature interactions
              • Identifies seasonal patterns and product cycle effects
              • Robust against outliers
              • Most effective when sufficient historical data is available

            - **Historical Average**: 
              • Serves as a benchmark for other models
              • Performs well in stable markets with low volatility
              • Simple and less prone to overfitting
              • Highly effective when historical patterns strongly repeat
            """)
            
            st.markdown("#### Accuracy Metrics")
            # Display model validation metrics
            validation_fig = dp.get_model_validation_metrics(ml_results)
            if validation_fig:
                st.plotly_chart(validation_fig, use_container_width=True)
            else:
                st.info("Validation metrics not available for current models")
            
            # Add forecast comparison
            st.markdown("#### Forecast Comparison")
            forecast_fig = dp.get_model_forecast_comparison(ml_results, 
                                                        ml_results.get('all_results', {}).get('historical_data'))
            if forecast_fig:
                st.plotly_chart(forecast_fig, use_container_width=True)
        
        with ml_tab3:
            # Display historical revenue and forecasts - using forecast_comparison for better seasonality visualization
            st.subheader("Historical & Projected Quarterly Revenue (2021-2025)")
            
            # Get historical data
            historical_data = ml_results.get('all_results', {}).get('historical_data')
            if historical_data is not None:
                # Only keep recent years' data for display
                if isinstance(historical_data, pd.DataFrame) and 'date' in historical_data.columns:
                    recent_years_data = historical_data[
                        pd.to_datetime(historical_data['date']) >= pd.Timestamp('2021-01-01')
                    ]
                else:
                    recent_years_data = historical_data
                
                # Use forecast_comparison instead of revenue_history_chart
                forecast_fig = dp.get_model_forecast_comparison(ml_results, recent_years_data)
                if forecast_fig:
                    st.plotly_chart(forecast_fig, use_container_width=True)
                else:
                    st.warning("Unable to generate forecast comparison chart")
            else:
                st.warning("Historical data not available")
            
            # Add chart description
            st.info("""
            **Chart Description**: This chart shows quarterly revenue data from 2021 to 2025, including historical data (black line) and model predictions (dashed lines).
            Note Apple's pronounced seasonal pattern - the first fiscal quarter (calendar Q4) is typically highest due to the holiday sales season.
            """)
            
            # Add historical data diagnostic button
            if st.button("Analyze Historical Data"):
                st.write("### Historical Data Analysis")
                hist_data = ml_results.get('all_results', {}).get('historical_data')
                
                if hist_data is not None and isinstance(hist_data, pd.DataFrame):
                    st.write(f"Number of data points: {len(hist_data)}")
                    
                    # Check year coverage
                    if 'date' in hist_data.columns:
                        hist_data['date'] = pd.to_datetime(hist_data['date'])
                        hist_data['year'] = hist_data['date'].dt.year
                        years_covered = hist_data['year'].nunique()
                        st.write(f"Years covered: {years_covered} ({hist_data['year'].min()}-{hist_data['year'].max()})")
                        
                        # Check seasonality
                        hist_data['quarter'] = hist_data['date'].dt.quarter
                        quarterly_avg = hist_data.groupby('quarter')['revenue'].mean().reset_index()
                        
                        # Calculate inter-quarter differences
                        max_q = quarterly_avg['revenue'].max()
                        min_q = quarterly_avg['revenue'].min()
                        
                        # Display quarterly distribution
                        st.write("Average quarterly revenue:")
                        st.write(quarterly_avg)
                        
                        # Display quarterly distribution chart
                        quarter_fig = px.bar(quarterly_avg, x='quarter', y='revenue',
                                            title='Average Quarterly Revenue',
                                            labels={'quarter': 'Quarter', 'revenue': 'Average Revenue (Billions USD)'})
                        st.plotly_chart(quarter_fig, use_container_width=True)
                        
                        # Determine if there's enough data to train ML models
                        if years_covered < 2:
                            st.warning("Historical data covers less than 2 years, may cause difficulty in recognizing seasonal patterns")
                        elif years_covered < 3:
                            st.info("Historical data covers less than 3 years, seasonal patterns may not be stable enough")
                        else:
                            st.success(f"Historical data covers {years_covered} years, sufficient to identify seasonal patterns")
            

        # Add session state to save button status
        if 'use_ml_prediction' not in st.session_state:
            st.session_state.use_ml_prediction = False

        # Add button and update status
        if st.button("Use ML Prediction in DCF Model"):
            st.session_state.use_ml_prediction = True
            st.success(f"Using ML prediction of {ml_results['growth_percentage']:.2f}% growth rate in the model")
    
    # Collect user input parameters
    with st.expander("Model Parameters", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Base year inputs (FY 2023)
            section_header("Base Year Financials (Billions USD)")
            revenue = st.number_input("Annual Revenue", value=apple_data['revenue'])
            net_income = st.number_input("Net Income", value=apple_data['net_income'])
            depreciation = st.number_input("Depreciation & Amortization", value=apple_data['depreciation'])
            capex = st.number_input("Capital Expenditure", value=apple_data['capex'])
            change_in_wc = st.number_input("Change in Working Capital", value=apple_data['change_in_wc'])
        
        with col2:
            # Growth and ratio assumptions
            section_header("Projection Assumptions")
            # Use ML prediction if button was clicked
            default_growth = ml_results['growth_percentage'] if st.session_state.use_ml_prediction else 5.0
            revenue_growth = st.slider("Revenue Growth Rate (%)", min_value=0.0, max_value=15.0, value=default_growth, step=0.5)
            profit_margin = st.slider("Net Profit Margin (%)", min_value=20.0, max_value=30.0, value=25.3, step=0.1)
            capex_to_revenue = st.slider("CapEx to Revenue (%)", min_value=2.0, max_value=5.0, value=2.8, step=0.1)
            discount_rate = st.slider("Discount Rate (WACC %)", min_value=5.0, max_value=15.0, value=8.5, step=0.1)
            terminal_growth_rate = st.slider("Terminal Growth Rate (%)", min_value=2.0, max_value=5.0, value=3.5, step=0.1)
    
    # Call DCF model calculation function
    params = {
        'revenue': revenue,
        'net_income': net_income,
        'depreciation': depreciation,
        'capex': capex,
        'change_in_wc': change_in_wc,
        'revenue_growth': revenue_growth / 100,
        'profit_margin': profit_margin / 100,
        'capex_to_revenue': capex_to_revenue / 100,
        'discount_rate': discount_rate / 100,
        'terminal_growth_rate': terminal_growth_rate / 100,
        'current_year': current_year,
        'projection_years': projection_years
    }
    
    # Run DCF model calculations
    projection_df, enterprise_value, sum_pv_fcf, pv_terminal_value = dcf.run_dcf_model(params)
    
    # Display projections
    section_header("Multi-Year Financial Projections (Billions USD)")
    st.dataframe(projection_df[['Revenue', 'Net Income', 'FCF']], use_container_width=True)
    
    # Cash flow chart
    section_header("Free Cash Flow Projection")
    fcf_fig = dp.get_fcf_chart(projection_df)
    st.plotly_chart(fcf_fig, use_container_width=True)
    
    # Valuation results
    section_header("DCF Valuation Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Sum of PV of FCF", f"${sum_pv_fcf:.2f}B")
    with col2:
        st.metric("PV of Terminal Value", f"${pv_terminal_value:.2f}B")
    with col3:
        st.metric("Enterprise Value", f"${enterprise_value:.2f}B")
    
    # Valuation composition pie chart
    fig = dp.get_ev_composition_chart(sum_pv_fcf, pv_terminal_value)
    st.plotly_chart(fig, use_container_width=True)
    
    # Return metrics
    section_header("Investment Return Metrics")
    current_price = apple_data['current_price']
    shares_outstanding = apple_data['shares_outstanding']
    net_debt = apple_data['net_debt']
    
    equity_value = enterprise_value - net_debt
    implied_share_price = equity_value / shares_outstanding
    upside_potential = (implied_share_price / current_price - 1) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Equity Value", f"${equity_value:.2f}B")
    with col2:
        st.metric("Implied Share Price", f"${implied_share_price:.2f}")
    with col3:
        st.metric("Upside Potential", f"{upside_potential:.1f}%", f"{upside_potential:.1f}%")

    # Add this somewhere in the DCF Model page
    with st.expander("Compare Historical vs. Predicted Growth"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Recent Historical Growth (YoY)", 
                f"{5.8}%",  # Replace with actual value from peer comparison
                help="Year-over-year revenue growth from most recent quarterly reports"
            )
        
        with col2:
            # Handle NaN values in the display
            if pd.isna(ml_results.get('growth_percentage')):
                growth_value = 3.0  # Default fallback
                st.metric(
                    "ML-Predicted Future Growth", 
                    f"{growth_value:.2f}%",
                    delta="N/A",
                    help="Machine learning consensus prediction for future annual growth (fallback value due to calculation error)"
                )
            else:
                growth_value = ml_results['growth_percentage']
                st.metric(
                    "ML-Predicted Future Growth", 
                    f"{growth_value:.2f}%",
                    delta=f"{growth_value - 5.8:.2f}%",  # Replace 5.8 with actual historical value
                    help="Machine learning consensus prediction for future annual growth"
                )
        
        st.markdown("""
        **Interpretation:**
        - If predicted growth is significantly lower than historical growth, this suggests a conservative forecast that anticipates slowing growth.
        - If predicted growth is higher than historical growth, this suggests an optimistic forecast that anticipates accelerating growth.
        - Significant differences may warrant further investigation of model assumptions and market conditions.
        """)

    # After processing ML prediction results, ensure all models have quarterly predictions
    if 'ml_results' in locals() and ml_results:
        ml_results = dp.ensure_model_quarterly_predictions(ml_results)

#C. Sensitivity Analysis Page
    # - Allows users to adjust key assumptions
    # - Shows impact on valuation
    # - Displays 3D sensitivity charts
elif page == "Sensitivity Analysis":
    st.title("Sensitivity Analysis")
    
    # Sensitivity analysis parameters
    st.markdown("""
    Explore how changes in key assumptions affect Apple's valuation. This analysis helps identify 
    which variables have the most significant impact on enterprise value, enabling more robust 
    investment decisions.
    """)
    
    # Get Apple's financial data
    apple_data = dp.get_apple_dcf_data()
    
    col1, col2 = st.columns(2)
    with col1:
        wacc_range = st.slider("WACC Range (%)", min_value=7.0, max_value=11.0, value=(8.0, 10.0), step=0.5)
        growth_range = st.slider("Terminal Growth Rate Range (%)", min_value=1.5, max_value=3.5, value=(2.0, 3.0), step=0.25)
    
    with col2:
        revenue_growth_range = st.slider("Revenue Growth Range (%)", min_value=3.0, max_value=7.0, value=(4.0, 6.0), step=0.5)
        margin_range = st.slider("Profit Margin Range (%)", min_value=23.0, max_value=27.0, value=(24.0, 26.0), step=0.5)
    
    # Add tab selection for different sensitivity analyses
    sensitivity_tab1, sensitivity_tab2 = st.tabs(["WACC vs. Growth", "Revenue Growth vs. Margin"])
    
    with sensitivity_tab1:
        # Generate sensitivity table and chart for WACC vs Terminal Growth
        section_header("Enterprise Value Sensitivity (Billions USD): WACC vs. Terminal Growth")
        sensitivity_df, sensitivity_data, wacc_values, growth_values = dcf.run_sensitivity_analysis(
            wacc_range, growth_range, base_fcf=94.795
        )
        
        # Add fixed assumptions note
        st.markdown("#### Fixed assumptions: Revenue Growth = 5.0%, Profit Margin = 25.0%, Base FCF = $94.795B")
        
        # Style the dataframe
        styled_df = dp.style_sensitivity_table(sensitivity_df)
        st.dataframe(styled_df, use_container_width=True)
        
        # 3D chart
        section_header("3D Sensitivity Visualization")
        fig = dp.get_3d_sensitivity_chart(wacc_values, growth_values, sensitivity_data)
        st.plotly_chart(fig, use_container_width=True)
    
    with sensitivity_tab2:
        # Generate sensitivity analysis for Revenue Growth vs Margin
        section_header("Enterprise Value Sensitivity (Billions USD): Revenue Growth vs. Profit Margin")
        
        # Create revenue growth values array
        rev_growth_min, rev_growth_max = revenue_growth_range
        rev_growth_values = np.linspace(rev_growth_min/100, rev_growth_max/100, 5)
        
        # Create margin values array
        margin_min, margin_max = margin_range
        margin_values = np.linspace(margin_min/100, margin_max/100, 5)
        
        # Initialize results matrix
        growth_margin_sensitivity = np.zeros((len(margin_values), len(rev_growth_values)))
        
        # Use fixed WACC and terminal growth
        fixed_wacc = 9.0/100
        fixed_terminal_growth = 2.5/100
        
        # Calculate EV for each combination
        for i, margin in enumerate(margin_values):
            for j, growth in enumerate(rev_growth_values):
                # Setup parameters
                params = {
                    'revenue': apple_data['revenue'],
                    'net_income': apple_data['net_income'],
                    'depreciation': apple_data['depreciation'],
                    'capex': apple_data['capex'],
                    'change_in_wc': apple_data['change_in_wc'],
                    'revenue_growth': growth,
                    'profit_margin': margin,
                    'capex_to_revenue': 2.8/100,
                    'discount_rate': fixed_wacc,
                    'terminal_growth_rate': fixed_terminal_growth,
                    'current_year': current_year,
                    'projection_years': projection_years
                }
                
                # Run DCF model calculations
                _, enterprise_value, _, _ = dcf.run_dcf_model(params)
                growth_margin_sensitivity[i, j] = enterprise_value
        
        # Create pandas DataFrame for display
        growth_labels = [f"{g*100:.1f}%" for g in rev_growth_values]
        margin_labels = [f"{m*100:.1f}%" for m in margin_values]
        growth_margin_df = pd.DataFrame(growth_margin_sensitivity, index=margin_labels, columns=growth_labels)
        
        # Display table
        styled_growth_margin_df = dp.style_sensitivity_table(growth_margin_df)
        st.markdown("#### Fixed assumptions: WACC = 9.0%, Terminal Growth = 2.5%")
        st.dataframe(styled_growth_margin_df, use_container_width=True)
        
        # Create 3D visualization
        fig2 = go.Figure(data=[go.Surface(
            z=growth_margin_sensitivity, 
            x=rev_growth_values*100, 
            y=margin_values*100,
            colorscale='RdBu',  # Changed to Red-Blue color scale
            reversescale=True   # Reversed to make high values blue and low values red
        )])
        
        fig2.update_layout(
            title='Enterprise Value Sensitivity',
            scene=dict(
                xaxis_title='Revenue Growth (%)',
                yaxis_title='Profit Margin (%)',
                zaxis_title='Enterprise Value ($B)'
            ),
            width=800,
            height=600
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # Key insights - dynamic calculations
    current_price = apple_data['current_price']
    shares_outstanding = apple_data['shares_outstanding']
    
    # Calculate WACC impact
    middle_growth_idx = len(growth_values)//2
    low_wacc_idx = 0  # First column
    high_wacc_idx = -1  # Last column
    
    if len(sensitivity_data) > 0 and len(sensitivity_data[0]) > 1:
        low_wacc_ev = sensitivity_data[middle_growth_idx][low_wacc_idx]
        high_wacc_ev = sensitivity_data[middle_growth_idx][high_wacc_idx]
        wacc_impact = (low_wacc_ev - high_wacc_ev) / high_wacc_ev * 100
        wacc_diff = (wacc_values[-1] - wacc_values[0]) * 100
        per_half_percent = round(wacc_impact / (wacc_diff / 0.5), 1)
    else:
        per_half_percent = 12.5  # Default value
    
    # Calculate optimal scenario
    min_ev = sensitivity_df.min().min()
    max_ev = sensitivity_df.max().max()
    optimal_share_price = (max_ev - 110.0) / shares_outstanding
    optimal_upside = round((optimal_share_price / current_price - 1) * 100, 1)
    
    # Calculate worst-case scenario
    worst_case_ev = sensitivity_data[-1][-1] if len(sensitivity_data) > 0 and len(sensitivity_data[-1]) > 0 else min_ev
    worst_case_share_price = (worst_case_ev - 110.0) / shares_outstanding
    worst_case_upside = (worst_case_share_price / current_price - 1) * 100
    fair_value_assessment = "fairly valued" if worst_case_upside > -10 else "undervalued" if worst_case_upside > 0 else "potentially overvalued"
    
    # Get lowest WACC for growth sensitivity note
    lowest_wacc = wacc_range[0] if isinstance(wacc_range, tuple) else 8.0
    
    st.markdown(f"""
    ### Key Insights from Sensitivity Analysis:
    
    1. **WACC Impact**: Each 0.5% decrease in WACC results in approximately {per_half_percent}% increase in enterprise value
    2. **Growth Sensitivity**: Terminal growth rate changes have more pronounced effects at lower WACC levels ({lowest_wacc:.1f}%)
    3. **Optimal Scenario**: The most favorable valuation scenario suggests an upside potential of ~{optimal_upside}%
    4. **Risk Assessment**: Even in the most conservative scenario, the model indicates Apple remains {fair_value_assessment}
    """)

#D. Industry Comparison page
    # - Compares Apple with peers
    # - Shows competitive positioning
    # - Provides investment recommendations
elif page == "Industry Comparison":
    st.title("Industry Benchmarking & Peer Analysis")
    
    # Add refresh button for peer data
    col1, col2 = st.columns([3, 1])
    with col2:
        refresh_peers = st.button("🔄 Refresh Peer Data")
    
    # Get peer comparison data
    peer_df = dp.get_peer_comparison_data()
    
    if refresh_peers:
        st.success("Using latest industry data for comparison")
    
    st.markdown("""
    Compare Apple's valuation metrics and financial performance against industry peers to identify
    relative strengths, weaknesses, and potential investment opportunities.
    """)
    
    # Display peer comparison
    section_header("Tech Giant Comparison")
    st.dataframe(peer_df.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='#ffcccc'), use_container_width=True)
    
    # Radar chart
    section_header("Competitive Positioning")
    radar_fig = dp.get_radar_chart(peer_df)
    st.plotly_chart(radar_fig, use_container_width=True)
    
    # Calculate dynamic insights based on peer comparison
    apple_roe = peer_df.loc["Apple", "ROE (%)"]
    apple_margin = peer_df.loc["Apple", "Net Margin (%)"]
    avg_peer_pe = peer_df.loc[peer_df.index != "Apple", "P/E Ratio"].mean()
    apple_pe = peer_df.loc["Apple", "P/E Ratio"]
    pe_premium = (apple_pe / avg_peer_pe - 1) * 100
    
    # Determine relative margin ranking
    margin_rank = peer_df["Net Margin (%)"].rank(ascending=False)["Apple"]
    margin_text = "industry-leading" if margin_rank == 1 else "above-average" if margin_rank <= 3 else "average"
    
    # Define recommendation and price target
    recommendation = "Hold"  # Default recommendation
    price_target = 209.28  # Current stock price as default
    
    st.markdown(f"""
    ### Comprehensive Investment Analysis
    
    **Competitive Position & Valuation**
    - Apple maintains {margin_text} profit margins ({apple_margin:.1f}%) and exceptional ROE ({apple_roe:.1f}%)
    - Currently trading at a {abs(pe_premium):.1f}% {("discount to" if pe_premium < 0 else "premium to")} peers, {("suggesting potential value opportunity" if pe_premium < 0 else "justified by superior financial metrics")}
    - Strong cash flow stability provides resilience against industry disruptions
    

    **Key Investment Considerations**
    1. Revenue Growth: Base case assumes 4.0%, ML models predict 8.1% consensus growth
    2. Margin Expansion: Current {apple_margin:.1f}% margin provides strong foundation for value creation
    3. Market Position: {("Discount" if pe_premium < 0 else "Premium")} to peers suggests potential for multiple {("expansion" if pe_premium < 0 else "contraction")}
    4. Risk Factors: Below Average risk-adjusted returns indicate need for careful position sizing
    """)
