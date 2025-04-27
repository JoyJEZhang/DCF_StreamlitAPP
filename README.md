# Advanced DCF Valuation Dashboard for Apple Inc.

This interactive Streamlit dashboard provides a comprehensive DCF (Discounted Cash Flow) valuation model for Apple Inc., with machine learning growth predictions and extensive historical data analysis. Designed for financial analysts, value engineering professionals, and investment decision-makers.

## Live Demo

**Access the live application: [https://dcf-analysis-joy.streamlit.app/](https://dcf-analysis-joy.streamlit.app/)**
## Features

- **Real-time Stock Price Integration**: Dynamic stock prices from Yahoo Finance for up-to-date valuations
- **ML-powered Growth Predictions**: Leverage multiple machine learning models (Linear, Ridge, Random Forest) to predict future revenue growth rates
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
   - yfinance
   
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
   - Adjust model weights using the sliders and click "Recalculate Consensus"
   - Click "Use ML Prediction in DCF Model" to apply the ML prediction to your DCF model

2. **Model Parameters**:
   - Adjust base financial metrics (Revenue, Net Income, etc.)
   - Set projection assumptions (Growth Rate, Profit Margin, WACC, etc.)
   - All changes automatically update the valuation results

### Sensitivity Analysis Page
- Use sliders to define ranges for key variables (WACC, Terminal Growth, etc.)
- View the sensitivity table and 3D chart to understand value drivers

### Industry Comparison Page
- Compare Apple against tech peers on key financial metrics
- Examine the radar chart to visualize competitive positioning

## Recent Updates

- Added real-time stock price integration with Yahoo Finance API
- Implemented ML-based growth rate prediction with multiple models
- Extended historical data to include 10 years of quarterly financials
- Fixed various bugs and improved overall performance

## Development

This project is built with Streamlit and uses Yahoo Finance API for real-time financial data. The machine learning models are implemented using scikit-learn.

## License

MIT License 