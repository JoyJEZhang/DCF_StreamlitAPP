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

# Add this near line 20, after imports but before page sections
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
page = st.sidebar.radio("Select Section", ["Executive Dashboard", "DCF Model", "Sensitivity Analysis", "Industry Comparison"])

# Common financial data
company_name = "Apple Inc."
ticker = "AAPL"
current_year = 2023
projection_years = 5

# Header styling
def section_header(title):
    st.markdown(f"<h2 style='color:#1E88E5;'>{title}</h2>", unsafe_allow_html=True)

# Executive Dashboard page
if page == "Executive Dashboard":
    st.title(f"Value Engineering Dashboard - {company_name} ({ticker})")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Stock Price", f"${apple_data['current_price']:.2f}", "4.2%")
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

# DCF Model page
elif page == "DCF Model":
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
            # Base year inputs (FY 2024)
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

# Sensitivity Analysis page
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
    
    # Generate sensitivity table and chart
    section_header("Enterprise Value Sensitivity (Billions USD): WACC vs. Terminal Growth")
    sensitivity_df, sensitivity_data, wacc_values, growth_values = dcf.run_sensitivity_analysis(
        wacc_range, growth_range, base_fcf=94.795
    )
    
    # Style the dataframe
    styled_df = dp.style_sensitivity_table(sensitivity_df)
    st.dataframe(styled_df, use_container_width=True)
    
    # 3D chart
    section_header("3D Sensitivity Visualization")
    fig = dp.get_3d_sensitivity_chart(wacc_values, growth_values, sensitivity_data)
    st.plotly_chart(fig, use_container_width=True)
    
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

# Industry Comparison page
elif page == "Industry Comparison":
    st.title("Industry Benchmarking & Peer Analysis")
    
    st.markdown("""
    Compare Apple's valuation metrics and financial performance against industry peers to identify
    relative strengths, weaknesses, and potential investment opportunities.
    """)
    
    # Get peer comparison data
    peer_df = dp.get_peer_comparison_data()
    
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
    pe_premium = round((apple_pe / avg_peer_pe - 1) * 100, 1)
    
    # Use global apple_data
    current_price = apple_data['current_price']
    
    # Determine relative margin ranking
    margin_rank = peer_df["Net Margin (%)"].rank(ascending=False)["Apple"]
    margin_text = "industry-leading" if margin_rank == 1 else "above-average" if margin_rank <= 3 else "average"
    
    # Calculate dynamic price target
    shares_outstanding = 15.7  # In billions
    
    # Simple price target calculation
    pe_target_price = round((apple_pe * 1.05) * (peer_df.loc["Apple", "Net Margin (%)"] / 24) * current_price / apple_pe)
    price_target = pe_target_price
    
    # Determine recommendation strength
    upside_potential = (price_target / current_price - 1) * 100
    recommendation = "Strong buy" if upside_potential > 20 else "Buy" if upside_potential > 10 else "Hold"
    
    st.markdown(f"""
    ### Value Engineering Insights:
    
    1. **Competitive Advantage**: Apple maintains {margin_text} profit margins ({apple_margin:.1f}%) and exceptional ROE ({apple_roe:.1f}%)
    2. **Valuation Context**: Trades at {pe_premium}% {("premium to" if pe_premium > 0 else "discount to")} peers, {("justified by superior financial metrics" if pe_premium > 0 else "suggesting potential value opportunity")}
    3. **Strategic Position**: Cash flow stability provides resilience against industry disruptions
    4. **Investment Thesis**: {recommendation} recommendation with ${price_target} price target based on DCF analysis
    """)
    
    # ROI Analysis
    section_header("ROI Analysis")
    
    # Define all necessary variables
    base_fcf = 94.795
    
    # Define default values for calculations
    base_case_revenue_growth = 0.04  # 4%
    bull_case_revenue_growth = 0.06  # 6% 
    base_case_margin = 0.24  # 24%
    bull_case_margin = 0.26  # 26%
    base_case_wacc = 0.10  # 10%
    bull_case_wacc = 0.08  # 8%
    base_case_growth = 0.02  # 2%
    bull_case_growth = 0.03  # 3%
    
    # Base case calculations
    base_case_fcf = base_fcf * (1 + base_case_revenue_growth) * (base_case_margin / 0.253)
    base_tv = base_case_fcf * (1 + base_case_growth) / (base_case_wacc - base_case_growth)
    base_pv_factor = 1 / ((1 + base_case_wacc) ** 5)
    base_ev = 400 + base_tv * base_pv_factor
    base_equity = base_ev - 110.0
    base_share_price = base_equity / shares_outstanding
    base_upside = (base_share_price / current_price - 1) * 100
    
    # Bull case calculations
    bull_case_fcf = base_fcf * (1 + bull_case_revenue_growth) * (bull_case_margin / 0.253)
    bull_tv = bull_case_fcf * (1 + bull_case_growth) / (bull_case_wacc - bull_case_growth)
    bull_pv_factor = 1 / ((1 + bull_case_wacc) ** 5)
    bull_ev = 400 + bull_tv * bull_pv_factor
    bull_equity = bull_ev - 110.0
    bull_share_price = bull_equity / shares_outstanding
    bull_upside = (bull_share_price / current_price - 1) * 100
    
    # Expected IRR calculation (simplified)
    expected_irr = ((bull_share_price / current_price) ** (1/5) - 1) * 100
    
    # Risk-adjusted return (simplified Sharpe ratio concept)
    risk_adj_return = (expected_irr - 8) / 15
    risk_adj_text = "Superior" if risk_adj_return > 0.5 else "Average" if risk_adj_return > 0.3 else "Below Average"
    
    st.markdown(f"""
    Based on our comprehensive DCF analysis and sensitivity testing, Apple presents a compelling investment opportunity:
    
    - **Base Case**: {base_upside:.1f}% potential upside with moderate growth assumptions ({base_case_revenue_growth*100:.1f}% revenue growth)
    - **Bull Case**: {bull_upside:.1f}% upside potential if revenue growth exceeds {bull_case_revenue_growth*100:.1f}% and margins expand to {bull_case_margin*100:.1f}%
    - **Expected IRR**: {expected_irr:.1f}% annualized return over 5-year investment horizon
    - **Risk-Adjusted Return**: {risk_adj_text} Sharpe ratio compared to industry peers
    """)
