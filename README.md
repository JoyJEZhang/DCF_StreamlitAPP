# Advanced DCF Valuation Dashboard for Apple Inc.

This interactive Streamlit dashboard provides a comprehensive DCF (Discounted Cash Flow) valuation model for Apple Inc., with machine learning growth predictions and extensive historical data analysis. Designed for financial analysts, value engineering professionals, and investment decision-makers.

## Live Demo

**Access the live application: [https://investment-analysis-joy.streamlit.app/](https://investment-analysis-joy.streamlit.app/)**

## Features

- **ML-powered Growth Predictions**: Leverage multiple machine learning models to predict future revenue growth rates
- **Extended Historical Data**: Analyze Apple's quarterly revenue from 2015-2024 (10 years of data)
- **Multi-year DCF Modeling**: Project cash flows over a 5-year period with customizable parameters
- **Sensitivity Analysis**: Visualize how changes in WACC and terminal growth affect enterprise value
- **Industry Benchmarking**: Compare key metrics against tech industry peers
- **Interactive Visualizations**: Explore financial data through interactive charts and 3D plots

## Installation

If you prefer to run the application locally:

1. Clone this repository
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```
   Key dependencies include:
   - streamlit
   - pandas
   - numpy
   - plotly
   - scikit-learn
   - yfinance (optional)
   
3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Project Structure

- **`app.py`**: Main Streamlit interface and UI components
- **`data_processor.py`**: Data handling and visualization functions
- **`dcf_model.py`**: Core DCF model calculations and sensitivity analysis
- **`data_fetcher.py`**: Data retrieval and management for historical records
- **`ml_models.py`**: Machine learning models for revenue growth prediction

## How to Use the App

### Navigation
- Use the sidebar to navigate between different sections:
  - **Executive Dashboard**: High-level overview
  - **DCF Model**: Core valuation tool
  - **Sensitivity Analysis**: Test different scenarios
  - **Industry Comparison**: Competitive benchmarking

### DCF Model Page
1. **ML Growth Predictions**:
   - View the consensus prediction from multiple ML models
   - Use the "Model Comparison" tab to see individual model predictions
   - Adjust model weights using the sliders and click "Recalculate Consensus"
   - Click "Use ML Prediction in DCF Model" to apply the ML prediction to your DCF model

2. **Model Parameters**:
   - Adjust base financial metrics (Revenue, Net Income, etc.)
   - Set projection assumptions (Growth Rate, Profit Margin, WACC, etc.)
   - All changes automatically update the valuation results

3. **Model Validation**:
   - Use the "Model Validation" tab to assess model performance metrics
   - Review the "Forecast Comparison" chart to see projected quarterly revenue

4. **Historical Data Analysis**:
   - Examine extended historical data back to 2015
   - Click "Analyze Historical Data" to see patterns and seasonality

### Sensitivity Analysis Page
1. Use sliders to define ranges for key variables (WACC, Terminal Growth, etc.)
2. View the sensitivity table and 3D chart to understand value drivers
3. Review the "Key Insights" section for interpretation of results

### Industry Comparison Page
1. Compare Apple against tech peers on key financial metrics
2. Examine the radar chart to visualize competitive positioning
3. Review ROI analysis to assess investment potential

## Machine Learning Models

The application uses multiple ML models to predict future revenue growth:

- **Linear Regression**: Simple trend-based prediction
  - Best for capturing linear growth patterns
  - Easy to interpret and explain

- **Ridge Regression**: Enhanced linear model with regularization
  - Handles correlated features more effectively
  - Prevents overfitting through parameter shrinkage

- **Random Forest**: Tree-based ensemble model
  - Captures non-linear relationships
  - Handles seasonal patterns and product cycles
  - Provides feature importance information

- **Historical Average**: Baseline comparison model
  - Uses past growth patterns to predict future growth
  - Quarter-specific historical averages for seasonal patterns

## Value Engineering Capabilities

This dashboard demonstrates sophisticated value engineering capabilities:

- **ML-Driven Forecasting**: Uses data science to improve prediction accuracy
- **Scenario Modeling**: Test multiple growth, margin, and discount rate assumptions
- **Risk Assessment**: Identify sensitivities and evaluate downside scenarios
- **Competitive Analysis**: Benchmark against industry leaders on multiple dimensions
- **Investment Decision Support**: Quantify potential returns and risk-adjusted metrics

## Historical Data

The dashboard includes extensive historical quarterly revenue data for Apple Inc:
- Coverage: 2015-2024 (10 years)
- Granularity: Quarterly revenue figures
- Seasonality: Captures Apple's pronounced Q1 (holiday season) patterns
- Source: Hardcoded from reliable financial reports for consistency

## Data Sources
- **Financial Data**: Base financial metrics from Apple's FY 2023 reports
- **Historical Revenue**: Quarterly revenue from 2015-2024
- **Peer Comparison**: Latest financial data from major tech companies

## Customization

The modular architecture allows for easy customization:
- Modify data inputs in `data_fetcher.py`
- Adjust calculation methodology in `dcf_model.py`
- Extend the UI by adding new sections to `app.py`
- Add new ML models in `ml_models.py`

## Future Enhancements
- Additional ML models (LSTM, XGBoost)
- Expanded peer comparison with more companies
- Scenario modeling for different economic conditions
- Export functionality for reports and presentations 