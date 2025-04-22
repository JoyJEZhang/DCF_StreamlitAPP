# Advanced DCF Valuation Dashboard

This interactive Streamlit dashboard provides a comprehensive DCF (Discounted Cash Flow) valuation model for Apple Inc., designed for financial analysts, value engineering professionals, and investment decision-makers.

## Features

- **Multi-year DCF Modeling**: Project cash flows over a 5-year period with adjustable growth rates and margins
- **Sensitivity Analysis**: Visualize how changes in WACC and terminal growth affect enterprise value
- **Industry Benchmarking**: Compare key metrics against tech industry peers
- **ROI Analysis**: Evaluate investment potential with predicted returns and upside calculations
- **Interactive Visualizations**: Explore financial data through charts, heatmaps, and 3D plots

## Project Structure

This application follows a modular design pattern with three main components:

1. **`app.py`**: Main Streamlit interface and UI components
   - Handles page navigation and user interface elements
   - Integrates data visualizations and model outputs
   - Provides interactive controls for model parameters

2. **`data_processor.py`**: Data handling and visualization functions
   - Generates charts, graphs, and visual representations
   - Formats and styles data for presentation
   - Implements peer comparison analytics

3. **`dcf_model.py`**: Core financial model logic
   - Implements DCF methodology and calculations
   - Performs sensitivity analysis across multiple variables
   - Calculates enterprise value, equity value, and investment metrics

## Installation

1. Clone this repository
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Dashboard Sections

- **Executive Dashboard**: High-level overview with key metrics and revenue forecasts
- **DCF Model**: Detailed financial projections and valuation calculations
- **Sensitivity Analysis**: Interactive tools to test different assumption scenarios
- **Industry Comparison**: Competitive benchmarking and relative performance metrics

## Value Engineering Capabilities

This dashboard demonstrates sophisticated value engineering capabilities:

- **Scenario Modeling**: Test multiple growth, margin, and discount rate assumptions
- **Risk Assessment**: Identify sensitivities and evaluate downside scenarios
- **Competitive Analysis**: Benchmark against industry leaders on multiple dimensions
- **Investment Decision Support**: Quantify potential returns and risk-adjusted metrics

## Data Sources

This dashboard uses Apple Inc.'s FY 2023 financial data as a baseline for projections and comparisons.

## Customization

The modular architecture allows for easy customization:
- Modify data inputs in the data processing module
- Adjust calculation methodology in the DCF model module
- Extend the UI by adding new sections to the main app 