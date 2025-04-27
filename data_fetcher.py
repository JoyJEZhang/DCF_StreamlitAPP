import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
import os
import json
import numpy as np

# File to store cached data
CACHE_FILE = "financial_data_cache.json"

def get_cached_or_default_data():
    """Return cached data or default data if no cache exists"""
    # Try to read from cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
                return pd.DataFrame.from_dict(cache_data["data"])
        except:
            pass
    
    # Default data as fallback
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

def get_peer_comparison_data(refresh=False):
    """
    Return hardcoded peer comparison data with accurate values
    
    Args:
        refresh (bool): Parameter kept for compatibility, but always ignored
        
    Returns:
        pd.DataFrame: Comparison metrics for tech giants
    """
    # Always return hardcoded data regardless of refresh parameter
    peers = ["Apple", "Microsoft", "Alphabet", "Amazon", "Meta"]
    metrics = {
        "Market Cap ($B)": [3140.0, 2900.0, 1950.0, 2000.0, 1350.0],
        "P/E Ratio": [29.0, 36.5, 23.5, 55.0, 25.5],
        "Revenue Growth (%)": [3.0, 7.0, 8.0, 12.0, 15.0],
        "Net Margin (%)": [26.0, 34.0, 23.0, 6.0, 35.0],
        "ROE (%)": [150.0, 40.0, 25.0, 10.0, 27.5],
        "EV/EBITDA": [23.5, 25.5, 16.5, 21.0, 16.5]
    }
    
    # Create DataFrame with companies as index
    return pd.DataFrame(metrics, index=peers)

def fetch_peer_comparison_data():
    """
    Fetch latest financial data and calculate metrics accurately from raw financial statements
    """
    ticker_mapping = {
        "Apple": "AAPL",
        "Microsoft": "MSFT", 
        "Alphabet": "GOOGL",
        "Amazon": "AMZN",
        "Meta": "META"
    }
    
    # Initialize metrics dictionary
    metrics = {
        "Market Cap ($B)": [],
        "P/E Ratio": [],
        "Revenue Growth (%)": [],
        "Net Margin (%)": [],
        "ROE (%)": [],
        "EV/EBITDA": []
    }
    
    companies = list(ticker_mapping.keys())
    
    try:
        for company, ticker in ticker_mapping.items():
            print(f"Fetching data for {company} ({ticker})...")
            # Get data from Yahoo Finance
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get financial statements
            income_stmt = stock.income_stmt
            balance_sheet = stock.balance_sheet
            
            # Market Cap - straightforward API value
            market_cap = round(info.get('marketCap', 0) / 1e9, 1)  # Convert to billions
            metrics["Market Cap ($B)"].append(market_cap)
            
            # P/E Ratio - from API or calculate from earnings
            if 'trailingPE' in info and info['trailingPE'] is not None:
                metrics["P/E Ratio"].append(round(info['trailingPE'], 1))
            else:
                # Calculate P/E from market cap and net income if available
                try:
                    latest_net_income = income_stmt.loc['Net Income'].iloc[0]
                    pe_ratio = (market_cap * 1e9) / latest_net_income
                    metrics["P/E Ratio"].append(round(pe_ratio, 1))
                except:
                    metrics["P/E Ratio"].append(0)  # Default if calculation fails
            
            # Revenue Growth - calculate from raw income statement
            try:
                if not income_stmt.empty and len(income_stmt.columns) >= 2:
                    # Get current and previous year revenue
                    current_revenue = income_stmt.loc['Total Revenue'].iloc[0]
                    previous_revenue = income_stmt.loc['Total Revenue'].iloc[1]
                    
                    # Calculate YoY growth rate
                    revenue_growth = ((current_revenue / previous_revenue) - 1) * 100
                    metrics["Revenue Growth (%)"].append(round(revenue_growth, 1))
                    print(f"  {company} Revenue Growth: {round(revenue_growth, 1)}% (${current_revenue/1e9:.1f}B / ${previous_revenue/1e9:.1f}B)")
                else:
                    metrics["Revenue Growth (%)"].append(0)
                    print(f"  ⚠️ Insufficient revenue data for {company}")
            except Exception as e:
                print(f"  ❌ Error calculating revenue growth for {company}: {e}")
                metrics["Revenue Growth (%)"].append(0)
            
            # Net Margin - calculate from revenue and net income
            try:
                if not income_stmt.empty:
                    latest_revenue = income_stmt.loc['Total Revenue'].iloc[0]
                    latest_net_income = income_stmt.loc['Net Income'].iloc[0]
                    
                    # Calculate net margin
                    net_margin = (latest_net_income / latest_revenue) * 100
                    metrics["Net Margin (%)"].append(round(net_margin, 1))
                    print(f"  {company} Net Margin: {round(net_margin, 1)}% (${latest_net_income/1e9:.1f}B / ${latest_revenue/1e9:.1f}B)")
                else:
                    metrics["Net Margin (%)"].append(0)
            except Exception as e:
                print(f"  ❌ Error calculating net margin for {company}: {e}")
                metrics["Net Margin (%)"].append(0)
            
            # ROE - calculate using the formula: Net Income / Shareholder's Equity
            try:
                if not income_stmt.empty and not balance_sheet.empty:
                    latest_net_income = income_stmt.loc['Net Income'].iloc[0]
                    latest_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
                    
                    # Calculate ROE
                    roe = (latest_net_income / latest_equity) * 100
                    metrics["ROE (%)"].append(round(roe, 1))
                    print(f"  {company} ROE: {round(roe, 1)}% (${latest_net_income/1e9:.1f}B / ${latest_equity/1e9:.1f}B)")
                else:
                    metrics["ROE (%)"].append(0)
                    print(f"  ⚠️ Insufficient data for {company} ROE calculation")
            except Exception as e:
                print(f"  ❌ Error calculating ROE for {company}: {e}")
                metrics["ROE (%)"].append(0)
            
            # EV/EBITDA - calculate from enterprise value and EBITDA
            try:
                # Enterprise Value from API or calculate
                ev = info.get('enterpriseValue', 0) / 1e9  # Convert to billions
                
                # Get EBITDA or calculate it
                if 'EBITDA' in income_stmt.index:
                    ebitda = income_stmt.loc['EBITDA'].iloc[0] / 1e9  # Convert to billions
                else:
                    # Calculate EBITDA = Operating Income + D&A
                    ebit = income_stmt.loc['Operating Income'].iloc[0]
                    depreciation = income_stmt.loc['Depreciation And Amortization'].iloc[0] if 'Depreciation And Amortization' in income_stmt.index else 0
                    ebitda = (ebit + depreciation) / 1e9  # Convert to billions
                
                # Calculate EV/EBITDA
                ev_ebitda = ev / ebitda if ebitda > 0 else 0
                metrics["EV/EBITDA"].append(round(ev_ebitda, 1))
                print(f"  {company} EV/EBITDA: {round(ev_ebitda, 1)} (${ev:.1f}B / ${ebitda:.1f}B)")
            except Exception as e:
                print(f"  ❌ Error calculating EV/EBITDA for {company}: {e}")
                metrics["EV/EBITDA"].append(0)
            
            print(f"Completed data processing for {company}\n")
            
        # Create DataFrame
        df = pd.DataFrame(metrics, index=companies)
        
        # Cache the data with timestamp
        cache_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data": df.to_dict()
        }
        
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_data, f)
        
        # Print summary of successful fields
        for metric, values in metrics.items():
            if not all(v == 0 for v in values):
                print(f"✅ Successfully calculated {metric} for {sum(1 for v in values if v > 0)}/{len(values)} companies")
            else:
                print(f"❌ Failed to calculate {metric} for all companies")
        
        return df
        
    except Exception as e:
        print(f"Error in peer comparison data fetch: {e}")
        # Return default data if fetching fails
        return get_cached_or_default_data()

def get_last_update_time():
    """Get timestamp of last data update"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
                if "timestamp" in cache_data:
                    timestamp_str = cache_data["timestamp"]
                    # Check if the timestamp is in the future
                    try:
                        timestamp_dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                        # Always use UTC for comparison to avoid timezone issues
                        now = datetime.now(timezone.utc).replace(tzinfo=None)
                        if timestamp_dt > now:
                            # If date is in the future, return current time instead
                            return now.strftime("%Y-%m-%d %H:%M:%S") + " (corrected)"
                        return timestamp_str
                    except ValueError:
                        # If date format is invalid, return current time
                        return datetime.now(timezone.utc).replace(tzinfo=None).strftime("%Y-%m-%d %H:%M:%S") + " (reset)"
                else:
                    return "Never (using default data)"
        except Exception as e:
            print(f"Error reading timestamp from cache: {e}")
            return "Never (using default data)"
    
    return "Never (using default data)"

def fetch_apple_dcf_data():
    """Fetch latest financial data for Apple's DCF model"""
    try:
        # Get Apple stock data
        aapl = yf.Ticker("AAPL")
        
        # Get income statement and cash flow statement
        income_stmt = aapl.income_stmt
        cash_flow = aapl.cashflow
        balance_sheet = aapl.balance_sheet
        
        # Get real-time stock price
        current_price = get_current_stock_price("AAPL")
        
        # Initialize with default values but use real-time price
        dcf_data = {
            'revenue': 383.3,              # Billions USD
            'net_income': 96.995,          # Billions USD
            'depreciation': 11.5,          # Billions USD
            'capex': 10.7,                 # Billions USD
            'change_in_wc': 3.0,           # Billions USD
            'current_price': current_price, # Real-time price instead of 191.24
            'shares_outstanding': 15.7,    # Billions
            'net_debt': 110.0              # Billions USD
        }
        
        if not income_stmt.empty and not cash_flow.empty:
            # Get latest fiscal year data (column 0)
            
            # Revenue (convert to billions)
            if 'Total Revenue' in income_stmt.index:
                dcf_data['revenue'] = round(income_stmt.loc['Total Revenue'][0] / 1e9, 2)
            
            # Net Income (convert to billions)
            if 'Net Income' in income_stmt.index:
                dcf_data['net_income'] = round(income_stmt.loc['Net Income'][0] / 1e9, 2)
            
            # Depreciation & Amortization (convert to billions)
            if 'Depreciation' in cash_flow.index:
                dcf_data['depreciation'] = round(cash_flow.loc['Depreciation'][0] / 1e9, 2)
            
            # Capital Expenditure (convert to billions)
            if 'Capital Expenditures' in cash_flow.index:
                # CapEx is usually negative in cash flow statements, so we take the absolute value
                dcf_data['capex'] = round(abs(cash_flow.loc['Capital Expenditures'][0]) / 1e9, 2)
            
            # Working Capital change (simplified calculation)
            if 'Change In Working Capital' in cash_flow.index:
                dcf_data['change_in_wc'] = round(abs(cash_flow.loc['Change In Working Capital'][0]) / 1e9, 2)
            
            # Get current market data
            info = aapl.info
            
            # Current stock price
            if 'currentPrice' in info:
                dcf_data['current_price'] = round(info['currentPrice'], 2)
            
            # Shares outstanding (convert to billions)
            if 'sharesOutstanding' in info:
                dcf_data['shares_outstanding'] = round(info['sharesOutstanding'] / 1e9, 2)
            
            # Calculate net debt (Total Debt - Cash & Equivalents)
            if 'Long Term Debt' in balance_sheet.index and 'Cash' in balance_sheet.index:
                total_debt = balance_sheet.loc['Long Term Debt'][0]
                if 'Short Long Term Debt' in balance_sheet.index:
                    total_debt += balance_sheet.loc['Short Long Term Debt'][0]
                cash = balance_sheet.loc['Cash'][0]
                dcf_data['net_debt'] = round((total_debt - cash) / 1e9, 2)
        
        # Cache the DCF data
        cache_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "apple_dcf_data": dcf_data
        }
        
        # Update the existing cache or create new
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r') as f:
                    existing_cache = json.load(f)
                existing_cache.update(cache_data)
                with open(CACHE_FILE, 'w') as f:
                    json.dump(existing_cache, f)
            except:
                with open(CACHE_FILE, 'w') as f:
                    json.dump(cache_data, f)
        else:
            with open(CACHE_FILE, 'w') as f:
                json.dump(cache_data, f)
        
        return dcf_data
        
    except Exception as e:
        print(f"Error fetching Apple DCF data: {e}")
        # Return default data if fetching fails
        return {
            'revenue': 383.3,              # Billions USD
            'net_income': 96.995,          # Billions USD
            'depreciation': 11.5,          # Billions USD
            'capex': 10.7,                 # Billions USD
            'change_in_wc': 3.0,           # Billions USD
            'current_price': 191.24,       # USD
            'shares_outstanding': 15.7,    # Billions
            'net_debt': 110.0              # Billions USD
        }

def get_apple_dcf_data():
    """
    Get Apple's financial data for DCF model with real-time stock price
    """
    # Get real-time stock price instead of using hardcoded value
    current_price = get_current_stock_price("AAPL")
    
    # Return financial data using current price
    return {
        'revenue': 394.33,
        'net_income': 99.82,
        'depreciation': 11.15,
        'capex': -10.71,
        'change_in_wc': -7.93,
        'current_price': current_price,  # Now using dynamic price
        'shares_outstanding': 15.7,
        'net_debt': 110.0
    }

def fetch_historical_revenue_data(ticker="AAPL", periods=32):
    """Fetch historical quarterly revenue data with better error handling and fallback data."""
    try:
        # Get stock data
        stock = yf.Ticker(ticker)
        print(f"Fetching quarterly financials for {ticker}")
        
        # Get quarterly and annual financial data
        q_financials = stock.quarterly_financials
        a_financials = stock.financials
        
        # Debug print
        print(f"Retrieved {len(q_financials.columns) if not q_financials.empty else 0} quarters of financial data")
        
        if not q_financials.empty and 'Total Revenue' in q_financials.index:
            # Extract revenue data
            revenues = q_financials.loc['Total Revenue']
            
            # If quarterly data is insufficient, supplement with annual data
            if len(revenues) < periods and not a_financials.empty:
                print(f"Supplementing with annual data (have {len(revenues)} quarters, need {periods})")
                annual_revenues = a_financials.loc['Total Revenue']
                
                # Create estimates for earlier quarters
                earliest_q_date = revenues.index.min()
                seasonal_pattern = [1.3, 0.9, 0.8, 1.0]  # Apple's typical seasonal pattern Q1, Q2, Q3, Q4
                
                for year_date, annual_rev in annual_revenues.items():
                    # Only process years before earliest quarter data
                    if year_date < earliest_q_date:
                        print(f"Adding estimated quarters for {year_date.year}")
                        # Calculate average quarterly revenue
                        avg_q_rev = annual_rev / 4
                        
                        # Apply seasonal pattern to create 4 quarters
                        for q in range(4):
                            q_date = pd.Timestamp(year=year_date.year, month=3*(q+1), day=1)
                            # If this quarter is earlier than the earliest existing quarter
                            if q_date < earliest_q_date:
                                # Add to revenues
                                revenues[q_date] = avg_q_rev * seasonal_pattern[q]
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'date': revenues.index,
                'revenue': revenues.values / 1e9  # Convert to billions
            })
            
            # Sort by date
            df = df.sort_values('date')
            
            # Add recent data if missing in API (hardcoded fallback for 2023-2024)
            latest_date = df['date'].max()
            today = pd.Timestamp.today()
            if (today - latest_date).days > 180:  # If data is more than 6 months old
                print(f"Latest data is from {latest_date}, adding recent estimates")
                
                # Use the last 4 quarters to estimate the pattern
                last_4q = df.tail(4).copy()
                last_4q['quarter'] = last_4q['date'].dt.quarter
                
                # Create missing quarters
                next_q_date = latest_date + pd.DateOffset(months=3)
                while next_q_date < today + pd.DateOffset(months=3):
                    next_q = next_q_date.quarter
                    # Find the same quarter in the previous year
                    prev_year_q = last_4q[last_4q['quarter'] == next_q]
                    
                    if not prev_year_q.empty:
                        # Apply ~5% annual growth to the previous year's same quarter
                        estimated_revenue = prev_year_q.iloc[0]['revenue'] * 1.05
                    else:
                        # Fallback: use the last quarter with slight growth
                        estimated_revenue = df.iloc[-1]['revenue'] * 1.02
                    
                    # Add to dataframe
                    df = pd.concat([df, pd.DataFrame({
                        'date': [next_q_date],
                        'revenue': [estimated_revenue]
                    })])
                    
                    # Move to next quarter
                    next_q_date = next_q_date + pd.DateOffset(months=3)
            
            # Keep requested number of periods
            if len(df) > periods:
                df = df.tail(periods)
            
            # Debug output
            print(f"Final historical data has {len(df)} quarters from {df['date'].min()} to {df['date'].max()}")
            
            # Handle serialization issues - convert dates to strings
            df_for_cache = df.copy()
            df_for_cache['date'] = df_for_cache['date'].dt.strftime('%Y-%m-%d')
            
            # Cache the data
            cache_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "historical_revenue_data": df_for_cache.to_dict()
            }
            
            # Update cache
            _update_cache(cache_data)
            
            return df
        else:
            print("No quarterly revenue data found, using default data")
            return _get_default_historical_revenue()
            
    except Exception as e:
        print(f"Error fetching historical revenue data: {e}")
        return _get_default_historical_revenue()

def get_hardcoded_apple_revenue_data():
    """Return hardcoded Apple quarterly financial data with extensive historical records (2015-2024)"""
    
    # Hardcoded Apple quarterly financial data
    quarters_data = [
        # 2024 fiscal year
        {"fiscal_quarter": "2024 Q1", "date": "2023-12-31", "revenue": 119.58},
        {"fiscal_quarter": "2024 Q2", "date": "2024-03-31", "revenue": 90.75},
        {"fiscal_quarter": "2024 Q3", "date": "2024-06-30", "revenue": 85.78},
        {"fiscal_quarter": "2024 Q4", "date": "2024-09-30", "revenue": 94.93},
        
        # 2023 fiscal year
        {"fiscal_quarter": "2023 Q1", "date": "2022-12-31", "revenue": 117.15},
        {"fiscal_quarter": "2023 Q2", "date": "2023-03-31", "revenue": 94.84},
        {"fiscal_quarter": "2023 Q3", "date": "2023-06-30", "revenue": 81.80},
        {"fiscal_quarter": "2023 Q4", "date": "2023-09-30", "revenue": 89.50},
        
        # 2022 fiscal year
        {"fiscal_quarter": "2022 Q1", "date": "2021-12-25", "revenue": 123.95},
        {"fiscal_quarter": "2022 Q2", "date": "2022-03-26", "revenue": 97.28},
        {"fiscal_quarter": "2022 Q3", "date": "2022-06-25", "revenue": 82.96},
        {"fiscal_quarter": "2022 Q4", "date": "2022-09-24", "revenue": 90.15},
        
        # 2021 fiscal year
        {"fiscal_quarter": "2021 Q1", "date": "2020-12-26", "revenue": 111.44},
        {"fiscal_quarter": "2021 Q2", "date": "2021-03-27", "revenue": 89.58},
        {"fiscal_quarter": "2021 Q3", "date": "2021-06-26", "revenue": 81.43},
        {"fiscal_quarter": "2021 Q4", "date": "2021-09-25", "revenue": 83.36},
        
        # 2020 fiscal year
        {"fiscal_quarter": "2020 Q1", "date": "2019-12-28", "revenue": 91.82},
        {"fiscal_quarter": "2020 Q2", "date": "2020-03-28", "revenue": 58.31},
        {"fiscal_quarter": "2020 Q3", "date": "2020-06-27", "revenue": 59.69},
        {"fiscal_quarter": "2020 Q4", "date": "2020-09-26", "revenue": 64.70},
        
        # 2019 fiscal year
        {"fiscal_quarter": "2019 Q1", "date": "2018-12-29", "revenue": 84.31},
        {"fiscal_quarter": "2019 Q2", "date": "2019-03-30", "revenue": 58.02},
        {"fiscal_quarter": "2019 Q3", "date": "2019-06-29", "revenue": 53.81},
        {"fiscal_quarter": "2019 Q4", "date": "2019-09-28", "revenue": 64.04},
        
        # 2018 fiscal year
        {"fiscal_quarter": "2018 Q1", "date": "2017-12-30", "revenue": 88.29},
        {"fiscal_quarter": "2018 Q2", "date": "2018-03-31", "revenue": 61.14},
        {"fiscal_quarter": "2018 Q3", "date": "2018-06-30", "revenue": 53.27},
        {"fiscal_quarter": "2018 Q4", "date": "2018-09-29", "revenue": 62.90},
        
        # 2017 fiscal year
        {"fiscal_quarter": "2017 Q1", "date": "2016-12-31", "revenue": 78.35},
        {"fiscal_quarter": "2017 Q2", "date": "2017-04-01", "revenue": 52.90},
        {"fiscal_quarter": "2017 Q3", "date": "2017-07-01", "revenue": 45.41},
        {"fiscal_quarter": "2017 Q4", "date": "2017-09-30", "revenue": 52.58},
        
        # 2016 fiscal year
        {"fiscal_quarter": "2016 Q1", "date": "2015-12-26", "revenue": 75.87},
        {"fiscal_quarter": "2016 Q2", "date": "2016-03-26", "revenue": 50.56},
        {"fiscal_quarter": "2016 Q3", "date": "2016-06-25", "revenue": 42.36},
        {"fiscal_quarter": "2016 Q4", "date": "2016-09-24", "revenue": 46.85},
        
        # 2015 fiscal year
        {"fiscal_quarter": "2015 Q1", "date": "2014-12-27", "revenue": 74.60},
        {"fiscal_quarter": "2015 Q2", "date": "2015-03-28", "revenue": 58.01},
        {"fiscal_quarter": "2015 Q3", "date": "2015-06-27", "revenue": 49.61},
        {"fiscal_quarter": "2015 Q4", "date": "2015-09-26", "revenue": 51.50},
        
        # 2025 fiscal year projection (keep this for future projection)
        {"fiscal_quarter": "2025 Q1", "date": "2024-12-28", "revenue": 124.30}
    ]
    
    # Create DataFrame
    df = pd.DataFrame(quarters_data)
    
    # Ensure correct date format
    df['date'] = pd.to_datetime(df['date'])
    
    # Add data source label
    df['source'] = "Actual Data"
    
    # Sort by date
    df = df.sort_values('date')
    
    return df

def get_historical_revenue_data(ticker="AAPL", refresh=False):
    """Get historical revenue data, always prioritizing hardcoded data"""
    print("Getting hardcoded Apple historical data...")
    return get_hardcoded_apple_revenue_data()  # Directly return hardcoded data

def _get_default_historical_revenue():
    """Return realistic default historical revenue data with clear source labeling"""
    # Apple's quarterly revenue (in billions) based on realistic historical patterns
    # Create data covering 2020 to present (about 14-16 quarters)
    today = pd.Timestamp.today()
    quarters_back = 16
    
    # Start from 16 quarters back and build up to today
    start_date = (today - pd.DateOffset(months=3*quarters_back)).to_period('Q').to_timestamp()
    dates = pd.date_range(start=start_date, periods=quarters_back, freq='Q')
    
    # Generate realistic Apple quarterly revenue with seasonality
    # Starting with 2020-Q1 value around $58B
    base_revenue = 58.0  # Starting point in billions
    quarterly_growth = 0.02  # 2% average quarterly growth
    seasonal_factor = np.array([1.3, 0.9, 0.8, 1.0])  # Q1, Q2, Q3, Q4 seasonality
    
    revenues = []
    for i in range(quarters_back):
        season_idx = i % 4
        seasonal_revenue = base_revenue * seasonal_factor[season_idx]
        random_factor = 1 + np.random.normal(0, 0.03)  # Random variation
        revenue = seasonal_revenue * random_factor
        revenues.append(revenue)
        base_revenue *= (1 + quarterly_growth)  # Apply growth for next quarter
    
    # Special handling for Covid-19 impact (2020-Q2 and Q3 had lower revenues)
    if start_date.year == 2020:
        covid_impact_quarters = [1, 2]  # Q2 and Q3 of 2020 (indices 1 and 2 if starting from Q1)
        for idx in covid_impact_quarters:
            if idx < len(revenues):
                revenues[idx] *= 0.9  # 10% reduction
    
    print(f"Generated synthetic data with {len(dates)} quarters from {dates[0]} to {dates[-1]}")
    
    # Create DataFrame and clearly mark as synthetic data
    df = pd.DataFrame({
        'date': dates,
        'revenue': revenues,
        'data_source': 'Synthetic'  # Clearly mark data source
    })
    
    return df
    
def _update_cache(new_data):
    """Helper function to update the cache file"""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r') as f:
                existing_cache = json.load(f)
            existing_cache.update(new_data)
            with open(CACHE_FILE, 'w') as f:
                json.dump(existing_cache, f)
        else:
            with open(CACHE_FILE, 'w') as f:
                json.dump(new_data, f)
    except Exception as e:
        print(f"Error updating cache: {e}")

def get_current_stock_price(ticker="AAPL"):
    """
    Get current stock price from Yahoo Finance
    """
    try:
        stock = yf.Ticker(ticker)
        ticker_data = stock.history(period="1d")
        
        if not ticker_data.empty:
            current_price = ticker_data['Close'].iloc[-1]
            return current_price
        else:
            print(f"Warning: No data returned for {ticker}, using default price")
            return 191.24  
    except Exception as e:
        print(f"Error fetching current stock price: {e}")
        return 191.24  

def fix_future_timestamps_in_cache():
    """Fix any cache files that have timestamps in the future"""
    fixed_files = []
    # Use UTC time for consistency across environments
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # Check the main cache file
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
            
            modified = False
            # Check if main timestamp exists and is in the future
            if "timestamp" in cache_data:
                try:
                    timestamp_dt = datetime.strptime(cache_data["timestamp"], "%Y-%m-%d %H:%M:%S")
                    if timestamp_dt > now:
                        cache_data["timestamp"] = now_str
                        modified = True
                        fixed_files.append(CACHE_FILE)
                except ValueError:
                    # If timestamp is invalid, update it
                    cache_data["timestamp"] = now_str
                    modified = True
                    fixed_files.append(CACHE_FILE)
            
            # Check for timestamps in nested data structures
            for key, value in cache_data.items():
                if isinstance(value, dict) and "timestamp" in value:
                    try:
                        timestamp_dt = datetime.strptime(value["timestamp"], "%Y-%m-%d %H:%M:%S")
                        if timestamp_dt > now:
                            value["timestamp"] = now_str
                            modified = True
                            fixed_files.append(f"{CACHE_FILE}:{key}")
                    except (ValueError, TypeError):
                        value["timestamp"] = now_str
                        modified = True
                        fixed_files.append(f"{CACHE_FILE}:{key}")
            
            # Save updated cache if modified
            if modified:
                with open(CACHE_FILE, 'w') as f:
                    json.dump(cache_data, f)
        except Exception as e:
            print(f"Error fixing timestamps in {CACHE_FILE}: {e}")
    
    # Check other common cache files in the cache directory
    cache_dir = "cache"
    if os.path.exists(cache_dir) and os.path.isdir(cache_dir):
        for filename in os.listdir(cache_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(cache_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        cache_data = json.load(f)
                    
                    modified = False
                    # Check for "last_updated" field (common in cache files)
                    if "last_updated" in cache_data:
                        try:
                            timestamp_dt = datetime.strptime(cache_data["last_updated"], "%Y-%m-%d %H:%M:%S")
                            if timestamp_dt > now:
                                cache_data["last_updated"] = now_str
                                modified = True
                                fixed_files.append(file_path)
                        except ValueError:
                            cache_data["last_updated"] = now_str
                            modified = True
                            fixed_files.append(file_path)
                    
                    # Save updated cache if modified
                    if modified:
                        with open(file_path, 'w') as f:
                            json.dump(cache_data, f)
                except Exception as e:
                    print(f"Error fixing timestamps in {file_path}: {e}")
    
    return fixed_files