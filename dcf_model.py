import pandas as pd
import numpy as np

def run_dcf_model(params):
    """
    Execute DCF model and return calculation results
    
    params: Dictionary containing all model input parameters
    returns: Projection dataframe, enterprise value, PV of FCF and PV of terminal value
    """
    # Create projection dataframe
    years = list(range(params['current_year'], params['current_year'] + params['projection_years'] + 1))
    projection_df = pd.DataFrame(index=years)
    
    # Set base year data
    projection_df.loc[params['current_year'], 'Revenue'] = params['revenue']
    projection_df.loc[params['current_year'], 'Net Income'] = params['net_income']
    projection_df.loc[params['current_year'], 'Depreciation'] = params['depreciation']
    projection_df.loc[params['current_year'], 'CapEx'] = params['capex']
    projection_df.loc[params['current_year'], 'Change in WC'] = params['change_in_wc']
    
    # Generate financial projections
    for year in range(params['current_year'] + 1, params['current_year'] + params['projection_years'] + 1):
        projection_df.loc[year, 'Revenue'] = projection_df.loc[year-1, 'Revenue'] * (1 + params['revenue_growth'])
        projection_df.loc[year, 'Net Income'] = projection_df.loc[year, 'Revenue'] * params['profit_margin']
        projection_df.loc[year, 'Depreciation'] = projection_df.loc[year-1, 'Depreciation'] * (1 + params['revenue_growth'] * 0.7)
        projection_df.loc[year, 'CapEx'] = projection_df.loc[year, 'Revenue'] * params['capex_to_revenue']
        projection_df.loc[year, 'Change in WC'] = projection_df.loc[year, 'Revenue'] * 0.01
    
    # Calculate free cash flow
    projection_df['FCF'] = projection_df['Net Income'] + projection_df['Depreciation'] - projection_df['CapEx'] - projection_df['Change in WC']
    
    # Calculate present values
    for i, year in enumerate(range(params['current_year'] + 1, params['current_year'] + params['projection_years'] + 1)):
        projection_df.loc[year, 'PV Factor'] = 1 / ((1 + params['discount_rate']) ** (i+1))
        projection_df.loc[year, 'PV of FCF'] = projection_df.loc[year, 'FCF'] * projection_df.loc[year, 'PV Factor']
    
    # Terminal value calculation
    final_year = params['current_year'] + params['projection_years']
    terminal_value = projection_df.loc[final_year, 'FCF'] * (1 + params['terminal_growth_rate']) / (params['discount_rate'] - params['terminal_growth_rate'])
    pv_terminal_value = terminal_value / ((1 + params['discount_rate']) ** params['projection_years'])
    
    # Enterprise value
    sum_pv_fcf = projection_df.loc[params['current_year']+1:final_year, 'PV of FCF'].sum()
    enterprise_value = sum_pv_fcf + pv_terminal_value
    
    return projection_df, enterprise_value, sum_pv_fcf, pv_terminal_value

def run_sensitivity_analysis(wacc_range, growth_range, base_fcf=94.795):
    """
    Execute sensitivity analysis and return results
    
    wacc_range: WACC range (min, max)
    growth_range: Growth rate range (min, max)
    base_fcf: Base free cash flow
    
    returns: Sensitivity dataframe, raw sensitivity data, wacc values, growth values
    """
    # Generate WACC vs Growth Rate sensitivity table
    wacc_values = np.arange(wacc_range[0], wacc_range[1] + 0.5, 0.5) / 100
    growth_values = np.arange(growth_range[0], growth_range[1] + 0.25, 0.25) / 100
    
    sensitivity_data = []
    for growth in growth_values:
        row_data = []
        for wacc in wacc_values:
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
    
    return sensitivity_df, sensitivity_data, wacc_values, growth_values
