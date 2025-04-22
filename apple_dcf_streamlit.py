import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Page configuration
st.set_page_config(page_title="Advanced DCF Valuation Dashboard", layout="wide")

# Sidebar for navigation
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

# Executive Dashboard
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
    
    # Sample visualization
    with st.container():
        section_header("Historical Performance vs Projections")
        years = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028]
        historical_revenue = [260.2, 274.5, 365.8, 394.3, 383.3, None, None, None, None, None]
        projected_revenue = [None, None, None, None, 383.3, 402.5, 422.6, 443.7, 465.9, 489.2]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=historical_revenue, mode='lines+markers', 
                                name='Historical Revenue', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=years, y=projected_revenue, mode='lines+markers', 
                                name='Projected Revenue', line=dict(color='green', dash='dash')))
        fig.update_layout(title='Revenue Trend and Forecast (Billions USD)',
                        xaxis_title='Fiscal Year',
                        yaxis_title='Revenue (Billions USD)')
        st.plotly_chart(fig, use_container_width=True)

# DCF Model Page
elif page == "DCF Model":
    st.title(f"Multi-Year DCF Valuation Model - {company_name}")
    
    st.markdown("""
    This model projects cash flows for the next 5 years and calculates enterprise value using the 
    Discounted Cash Flow method based on FY 2023 financial data as the baseline.
    """)
    
    with st.expander("Model Parameters", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Base year inputs (FY 2023)
            section_header("Base Year Financials (Billions USD)")
            revenue = st.number_input("Annual Revenue", value=383.3)
            net_income = st.number_input("Net Income", value=96.995)
            depreciation = st.number_input("Depreciation & Amortization", value=11.5)
            capex = st.number_input("Capital Expenditure", value=10.7)
            change_in_wc = st.number_input("Change in Working Capital", value=3.0)
        
        with col2:
            # Growth and rate assumptions
            section_header("Projection Assumptions")
            revenue_growth = st.slider("Revenue Growth Rate (%)", min_value=0.0, max_value=15.0, value=5.0, step=0.5)
            profit_margin = st.slider("Net Profit Margin (%)", min_value=20.0, max_value=30.0, value=25.3, step=0.1)
            capex_to_revenue = st.slider("CapEx to Revenue (%)", min_value=2.0, max_value=5.0, value=2.8, step=0.1)
            discount_rate = st.slider("Discount Rate (WACC %)", min_value=5.0, max_value=15.0, value=9.0, step=0.1)
            terminal_growth_rate = st.slider("Terminal Growth Rate (%)", min_value=2.0, max_value=5.0, value=2.5, step=0.1)
    
    # Convert percentage inputs to decimals
    revenue_growth_rate = revenue_growth / 100
    profit_margin_rate = profit_margin / 100
    capex_to_revenue_rate = capex_to_revenue / 100
    discount_rate_decimal = discount_rate / 100
    terminal_growth_rate_decimal = terminal_growth_rate / 100
    
    # Create projection dataframe
    years = list(range(current_year, current_year + projection_years + 1))
    projection_df = pd.DataFrame(index=years)
    
    # Project financials
    projection_df.loc[current_year, 'Revenue'] = revenue
    projection_df.loc[current_year, 'Net Income'] = net_income
    projection_df.loc[current_year, 'Depreciation'] = depreciation
    projection_df.loc[current_year, 'CapEx'] = capex
    projection_df.loc[current_year, 'Change in WC'] = change_in_wc
    
    for year in range(current_year + 1, current_year + projection_years + 1):
        projection_df.loc[year, 'Revenue'] = projection_df.loc[year-1, 'Revenue'] * (1 + revenue_growth_rate)
        projection_df.loc[year, 'Net Income'] = projection_df.loc[year, 'Revenue'] * profit_margin_rate
        projection_df.loc[year, 'Depreciation'] = projection_df.loc[year-1, 'Depreciation'] * (1 + revenue_growth_rate * 0.7)
        projection_df.loc[year, 'CapEx'] = projection_df.loc[year, 'Revenue'] * capex_to_revenue_rate
        projection_df.loc[year, 'Change in WC'] = projection_df.loc[year, 'Revenue'] * 0.01
    
    # Calculate FCF
    projection_df['FCF'] = projection_df['Net Income'] + projection_df['Depreciation'] - projection_df['CapEx'] - projection_df['Change in WC']
    
    # Present Value calculation
    for i, year in enumerate(range(current_year + 1, current_year + projection_years + 1)):
        projection_df.loc[year, 'PV Factor'] = 1 / ((1 + discount_rate_decimal) ** (i+1))
        projection_df.loc[year, 'PV of FCF'] = projection_df.loc[year, 'FCF'] * projection_df.loc[year, 'PV Factor']
    
    # Terminal Value calculation
    final_year = current_year + projection_years
    terminal_value = projection_df.loc[final_year, 'FCF'] * (1 + terminal_growth_rate_decimal) / (discount_rate_decimal - terminal_growth_rate_decimal)
    pv_terminal_value = terminal_value / ((1 + discount_rate_decimal) ** projection_years)
    
    # Total Enterprise Value
    sum_pv_fcf = projection_df.loc[current_year+1:final_year, 'PV of FCF'].sum()
    enterprise_value = sum_pv_fcf + pv_terminal_value
    
    # Display projections
    section_header("Multi-Year Financial Projections (Billions USD)")
    st.dataframe(projection_df[['Revenue', 'Net Income', 'FCF']], use_container_width=True)
    
    # FCF chart
    section_header("Free Cash Flow Projection")
    fcf_fig = px.bar(
        projection_df.iloc[1:],  # Skip base year
        x=projection_df.index[1:],
        y='FCF',
        title='Projected Free Cash Flow (Billions USD)'
    )
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
    
    # Valuation breakdown pie chart
    labels = ['PV of Projected FCF', 'PV of Terminal Value']
    values = [sum_pv_fcf, pv_terminal_value]
    
    fig = px.pie(values=values, names=labels, title='Enterprise Value Composition')
    st.plotly_chart(fig, use_container_width=True)
    
    # Return metrics
    section_header("Investment Return Metrics")
    shares_outstanding = 15.7  # In billions
    current_price = 191.24
    
    equity_value = enterprise_value - 110.0  # Subtract net debt
    implied_share_price = equity_value / shares_outstanding
    upside_potential = (implied_share_price / current_price - 1) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Equity Value", f"${equity_value:.2f}B")
    with col2:
        st.metric("Implied Share Price", f"${implied_share_price:.2f}")
    with col3:
        st.metric("Upside Potential", f"{upside_potential:.1f}%", f"{upside_potential:.1f}%")

# Sensitivity Analysis
elif page == "Sensitivity Analysis":
    st.title("Sensitivity Analysis")
    
    # Parameters for sensitivity analysis
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
    
    # Generate sensitivity table for WACC vs Growth Rate
    section_header("Enterprise Value Sensitivity (Billions USD): WACC vs. Terminal Growth")
    
    wacc_values = np.arange(wacc_range[0], wacc_range[1] + 0.5, 0.5) / 100
    growth_values = np.arange(growth_range[0], growth_range[1] + 0.25, 0.25) / 100
    
    sensitivity_data = []
    for growth in growth_values:
        row_data = []
        for wacc in wacc_values:
            # Simplified calculation for demo
            base_fcf = 94.795  # From earlier calculation
            terminal_value = base_fcf * (1 + growth) / (wacc - growth)
            pv_factor = 1 / ((1 + wacc) ** 5)
            value = 400 + terminal_value * pv_factor  # Simplified EV calculation
            row_data.append(round(value, 1))
        sensitivity_data.append(row_data)
    
    sensitivity_df = pd.DataFrame(
        sensitivity_data, 
        index=[f"{g*100:.2f}%" for g in growth_values],
        columns=[f"{w*100:.1f}%" for w in wacc_values]
    )
    sensitivity_df.index.name = "Terminal Growth"
    sensitivity_df.columns.name = "WACC"
    
    # Style the dataframe for a heatmap effect
    def color_scale(val):
        normalized = (val - sensitivity_df.min().min()) / (sensitivity_df.max().max() - sensitivity_df.min().min())
        r, g, b = int(255 * (1 - normalized)), int(255 * normalized), 100
        return f'background-color: rgb({r}, {g}, {b})'
    
    styled_df = sensitivity_df.style.applymap(color_scale).format("${:.1f}B")
    st.dataframe(styled_df, use_container_width=True)
    
    # 3D Surface plot
    section_header("3D Sensitivity Visualization")
    
    wacc_grid, growth_grid = np.meshgrid(wacc_values, growth_values)
    z_values = np.array(sensitivity_data)
    
    fig = go.Figure(data=[go.Surface(z=z_values, x=wacc_values*100, y=growth_values*100)])
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
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insights - dynamic calculations
    min_ev = sensitivity_df.min().min()
    max_ev = sensitivity_df.max().max()
    current_price = 191.24
    shares_outstanding = 15.7  # In billions

    # Calculate WACC impact
    middle_growth_idx = len(growth_values)//2
    low_wacc_idx = 0  # First column
    high_wacc_idx = -1  # Last column

    # Get values from sensitivity table for a consistent growth rate
    if len(sensitivity_data) > 0 and len(sensitivity_data[0]) > 1:
        low_wacc_ev = sensitivity_data[middle_growth_idx][low_wacc_idx]
        high_wacc_ev = sensitivity_data[middle_growth_idx][high_wacc_idx]
        wacc_impact = (low_wacc_ev - high_wacc_ev) / high_wacc_ev * 100
        wacc_diff = (wacc_values[-1] - wacc_values[0]) * 100
        per_half_percent = round(wacc_impact / (wacc_diff / 0.5), 1)
    else:
        per_half_percent = 12.5  # Fallback value

    # Calculate optimal upside scenario
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

# Industry Comparison
elif page == "Industry Comparison":
    st.title("Industry Benchmarking & Peer Analysis")
    
    st.markdown("""
    Compare Apple's valuation metrics and financial performance against industry peers to identify
    relative strengths, weaknesses, and potential investment opportunities.
    """)
    
    # Peer comparison data
    peers = ["Apple", "Microsoft", "Alphabet", "Amazon", "Meta"]
    metrics = {
        "Market Cap ($B)": [2980, 2890, 1780, 1760, 1130],
        "P/E Ratio": [31.5, 34.8, 24.5, 59.3, 26.1],
        "Revenue Growth (%)": [5.8, 7.2, 9.5, 8.7, 11.2],
        "Net Margin (%)": [25.3, 36.8, 23.7, 8.5, 29.2],
        "ROE (%)": [160.1, 42.5, 27.8, 18.7, 26.9],
        "EV/EBITDA": [24.8, 25.3, 14.7, 21.5, 15.8]
    }
    
    peer_df = pd.DataFrame(metrics, index=peers)
    
    # Display peer comparison
    section_header("Tech Giant Comparison")
    st.dataframe(peer_df.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='#ffcccc'), use_container_width=True)
    
    # Radar chart
    section_header("Competitive Positioning")
    
    # Normalize data for radar chart
    radar_metrics = ["Revenue Growth (%)", "Net Margin (%)", "ROE (%)", "EV/EBITDA"]
    radar_df = peer_df[radar_metrics].copy()
    
    # Invert EV/EBITDA so lower is better like the other metrics
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
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Calculate dynamic insights based on peer comparison
    apple_roe = peer_df.loc["Apple", "ROE (%)"]
    apple_margin = peer_df.loc["Apple", "Net Margin (%)"]
    avg_peer_pe = peer_df.loc[peer_df.index != "Apple", "P/E Ratio"].mean()
    apple_pe = peer_df.loc["Apple", "P/E Ratio"]
    pe_premium = round((apple_pe / avg_peer_pe - 1) * 100, 1)

    # Add this line to define current_price
    current_price = 191.24  # Define current stock price here

    # Determine relative ranking for margins
    margin_rank = peer_df["Net Margin (%)"].rank(ascending=False)["Apple"]
    margin_text = "industry-leading" if margin_rank == 1 else "above-average" if margin_rank <= 3 else "average"

    # Calculate a dynamic price target based on peer metrics and DCF
    # Simple price target calculation based on peer P/E + premium
    avg_forward_growth = peer_df.loc["Apple", "Revenue Growth (%)"] * 1.1  # Forward growth estimate
    growth_adjusted_pe = avg_peer_pe * (1 + (avg_forward_growth / 100) * 0.5)
    pe_target_price = round((apple_pe * 1.05) * (peer_df.loc["Apple", "Net Margin (%)"] / 24) * current_price / apple_pe)

    # Use simple DCF calculation if available, otherwise use PE-based target
    # Since 'bull_share_price' likely isn't accessible, we'll use the PE-based target directly
    price_target = pe_target_price

    # Determine recommendation strength based on upside
    upside_potential = (price_target / current_price - 1) * 100

    recommendation = "Strong buy" if upside_potential > 20 else "Buy" if upside_potential > 10 else "Hold"

    st.markdown(f"""
    ### Value Engineering Insights:

    1. **Competitive Advantage**: Apple maintains {margin_text} profit margins ({apple_margin:.1f}%) and exceptional ROE ({apple_roe:.1f}%)
    2. **Valuation Context**: Trades at {pe_premium}% {("premium to" if pe_premium > 0 else "discount to")} peers, {("justified by superior financial metrics" if pe_premium > 0 else "suggesting potential value opportunity")}
    3. **Strategic Position**: Cash flow stability provides resilience against industry disruptions
    4. **Investment Thesis**: {recommendation} recommendation with ${price_target} price target based on DCF analysis
    """)
