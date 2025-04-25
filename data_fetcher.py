import pandas as pd
import yfinance as yf
from datetime import datetime
import os
import json

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

def fetch_peer_comparison_data():
    """Fetch latest financial data from Yahoo Finance"""
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
    
    try:
        for company, ticker in ticker_mapping.items():
            # Get data from Yahoo Finance
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract metrics
            metrics["Market Cap ($B)"].append(round(info.get('marketCap', 0) / 1e9, 1))
            metrics["P/E Ratio"].append(round(info.get('trailingPE', 0), 1) if info.get('trailingPE') else 0)
            
            # Calculate growth and margins
            try:
                # Use quarterly financials for recent data
                financials = stock.quarterly_financials
                balance_sheet = stock.quarterly_balance_sheet
                
                if not financials.empty and len(financials.columns) >= 2:
                    # Revenue Growth
                    current_revenue = financials.loc['Total Revenue', financials.columns[0]] if 'Total Revenue' in financials.index else 0
                    prev_revenue = financials.loc['Total Revenue', financials.columns[1]] if 'Total Revenue' in financials.index else 0
                    revenue_growth = (current_revenue / prev_revenue - 1) * 100 if prev_revenue else 0
                    metrics["Revenue Growth (%)"].append(round(revenue_growth, 1))
                    
                    # Net Margin
                    net_income = financials.loc['Net Income', financials.columns[0]] if 'Net Income' in financials.index else 0
                    net_margin = (net_income / current_revenue) * 100 if current_revenue else 0
                    metrics["Net Margin (%)"].append(round(net_margin, 1))
                    
                    # ROE
                    equity = balance_sheet.loc['Total Stockholder Equity', balance_sheet.columns[0]] if 'Total Stockholder Equity' in balance_sheet.index else 0
                    roe = (net_income / equity) * 100 if equity else 0
                    metrics["ROE (%)"].append(round(roe, 1))
                    
                    # EV/EBITDA - simplified calculation
                    enterprise_value = info.get('enterpriseValue', 0) / 1e9
                    ebitda = financials.loc['EBITDA', financials.columns[0]] / 1e9 if 'EBITDA' in financials.index else net_income / 1e9
                    ev_ebitda = enterprise_value / ebitda if ebitda else 0
                    metrics["EV/EBITDA"].append(round(ev_ebitda, 1))
                else:
                    # Fallback if financials not available
                    metrics["Revenue Growth (%)"].append(0)
                    metrics["Net Margin (%)"].append(0)
                    metrics["ROE (%)"].append(0)
                    metrics["EV/EBITDA"].append(0)
                    
            except Exception as e:
                print(f"Error processing financial data for {company}: {e}")
                # Use default values if calculation fails
                metrics["Revenue Growth (%)"].append(0)
                metrics["Net Margin (%)"].append(0)
                metrics["ROE (%)"].append(0)
                metrics["EV/EBITDA"].append(0)
        
        # Create DataFrame
        companies = list(ticker_mapping.keys())
        df = pd.DataFrame(metrics, index=companies)
        
        # Cache the data
        cache_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data": df.to_dict()
        }
        
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_data, f)
        
        return df
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        # Return default data if fetching fails
        return get_cached_or_default_data()

def get_last_update_time():
    """Get timestamp of last data update"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
                return cache_data["timestamp"]
        except:
            pass
    
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
        
        # Initialize with default values in case fetching fails
        dcf_data = {
            'revenue': 383.3,              # Billions USD
            'net_income': 96.995,          # Billions USD
            'depreciation': 11.5,          # Billions USD
            'capex': 10.7,                 # Billions USD
            'change_in_wc': 3.0,           # Billions USD
            'current_price': 191.24,       # USD
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
    """Get Apple DCF data from cache or default"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
                if "apple_dcf_data" in cache_data:
                    return cache_data["apple_dcf_data"]
        except:
            pass
    
    # Default Apple DCF data
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
