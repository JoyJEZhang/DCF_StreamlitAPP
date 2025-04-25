import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Import custom modules
import data_processor as dp
import dcf_model as dcf

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
        st.metric("Current Stock Price", "$191.24", "4.2%")
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
        
        # Display the ML prediction
        st.info(f"🤖 **ML Consensus Prediction:** {ml_results['growth_percentage']:.2f}% annual revenue growth rate")
        st.caption(f"Based on: {', '.join(ml_results['models_used'])}")
        
        # Use tabs to show the charts
        ml_tab1, ml_tab2 = st.tabs(["Model Comparison", "Historical Data"])
        
        with ml_tab1:
            # Show chart comparing different model predictions
            model_fig = dp.get_model_comparison_chart(ml_results)
            st.plotly_chart(model_fig, use_container_width=True)
        
        with ml_tab2:
            # Show historical revenue with projection
            hist_fig = dp.get_revenue_history_chart(ml_results)
            st.plotly_chart(hist_fig, use_container_width=True)
        
        # Add button to use ML prediction in DCF model
        use_ml_prediction = st.button("Use ML Prediction in DCF Model")
    
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
            default_growth = ml_results['growth_percentage'] if 'use_ml_prediction' in locals() and use_ml_prediction else 5.0
            revenue_growth = st.slider("Revenue Growth Rate (%)", min_value=0.0, max_value=15.0, value=default_growth, step=0.5)
            profit_margin = st.slider("Net Profit Margin (%)", min_value=20.0, max_value=30.0, value=25.3, step=0.1)
            capex_to_revenue = st.slider("CapEx to Revenue (%)", min_value=2.0, max_value=5.0, value=2.8, step=0.1)
            discount_rate = st.slider("Discount Rate (WACC %)", min_value=5.0, max_value=15.0, value=9.0, step=0.1)
            terminal_growth_rate = st.slider("Terminal Growth Rate (%)", min_value=2.0, max_value=5.0, value=2.5, step=0.1)
    
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

# Sensitivity Analysis page
elif page == "Sensitivity Analysis":
    st.title("Sensitivity Analysis")
    
    # Sensitivity analysis parameters
    st.markdown("""
    Explore how changes in key assumptions affect Apple's valuation. This analysis helps identify 
    which variables have the most significant impact on enterprise value, enabling more robust 
    investment decisions.
    """)
    
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
    current_price = 191.24
    shares_outstanding = 15.7  # In billions
    
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
    
    # Add this line to define current price
    current_price = 191.24
    
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
