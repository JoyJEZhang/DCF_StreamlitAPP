import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def get_revenue_trend_chart():
    """Generate revenue trend and forecast chart"""
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

def get_peer_comparison_data():
    """Retrieve peer comparison data"""
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
    
    return pd.DataFrame(metrics, index=peers)

def get_radar_chart(peer_df):
    """Generate radar chart displaying competitive positioning"""
    # Normalize radar chart data
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
    
    styled_df = sensitivity_df.style.applymap(color_scale).format("${:.1f}B")
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
