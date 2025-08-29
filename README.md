# QSR Executive Enterprises Dashboard

A comprehensive Streamlit dashboard for analyzing delivery platform performance across DoorDash and UberEats.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. **Clone or download this repository**
   ```bash
   # If using git
   git clone <repository-url>
   cd QSR-EE
   ```

2. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   - The dashboard will automatically open at `http://localhost:8501`
   - If it doesn't open automatically, manually navigate to the URL shown in your terminal

## 📊 Dashboard Features

### Platform Selection
- **🚚 DoorDash**: Analyze financial data, marketing campaigns, and sales analytics
- **🚗 UberEats**: Explore ad campaigns, offers, and unified sales data

### Data Analysis Capabilities
- **Basic Data Exploration**: Row counts, column information, data completeness
- **Column Names Display**: View all column names for each dataset
- **Sales & Order Metrics**: Automatic identification of sales and order-related columns
- **Time Series Analysis**: Interactive charts for sales and order trends over time
- **Daily Aggregated Data**: View daily totals, counts, and averages

### Data Sources

#### DoorDash Data
- **Financial Data**: Payout summaries, transactions, error charges
- **Marketing Data**: Promotion campaigns, sponsored listings
- **Sales Data**: Order-level sales data with detailed metrics

#### UberEats Data
- **Ad Campaigns**: Performance metrics, impressions, clicks, ROAS
- **Offers**: Campaign performance, customer acquisition
- **Unified Sales**: Store performance, financial data, order tracking

## 📁 File Structure

```
QSR-EE/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── dashboard_description.txt       # Detailed dashboard documentation
├── all_ads.csv                     # UberEats ad campaign data
├── all_offers.csv                  # UberEats offers data
├── all_united.csv                  # UberEats unified sales data
├── SALES_viewByOrder_*.csv         # DoorDash sales data
├── financial_*/                    # DoorDash financial data folder
│   ├── FINANCIAL_PAYOUT_SUMMARY_*.csv
│   ├── FINANCIAL_SIMPLIFIED_TRANSACTIONS_*.csv
│   ├── FINANCIAL_DETAILED_TRANSACTIONS_*.csv
│   └── FINANCIAL_ERROR_CHARGES_AND_ADJUSTMENTS_*.csv
└── marketing_*/                    # DoorDash marketing data folder
    ├── MARKETING_PROMOTION_*.csv
    └── MARKETING_SPONSORED_LISTING_*.csv
```

## 🔧 Configuration

### Customizing the Dashboard

1. **Adding New Data Sources**
   - Place new CSV files in the appropriate folders
   - Update the data loading functions in `app.py`

2. **Modifying Visualizations**
   - Edit the `display_data_info()` function for basic analysis
   - Add new analysis functions for advanced features

3. **Styling Changes**
   - Modify the CSS in the `st.markdown()` section
   - Update colors, fonts, and layout as needed

## 📈 Usage Guide

### Getting Started
1. Launch the dashboard using `streamlit run app.py`
2. Select your desired platform (DoorDash, UberEats, or Both)
3. Explore the data analysis sections

### Understanding the Analysis
- **Data Completeness**: Percentage of non-null values across all columns
- **Sales Metrics**: Revenue, sales, and financial performance indicators
- **Order Metrics**: Order counts, order-related performance data
- **Time Series**: Daily aggregated trends and patterns

### Best Practices
- Select the appropriate platform for your analysis needs
- Check data completeness before analysis
- Use time series analysis to identify trends and patterns
- Review daily aggregated data for performance insights

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **File Not Found Errors**
   - Ensure all data files are in the correct directories
   - Check file permissions

3. **Memory Issues**
   - Close other applications to free up RAM
   - Consider using smaller data samples for testing

4. **Port Already in Use**
   ```bash
   streamlit run app.py --server.port 8502
   ```

### Performance Tips
- The dashboard uses caching to improve loading times
- Large files may take longer to load initially
- Consider data sampling for very large datasets

## 📝 Development

### Adding New Features
1. Create new functions for specific analysis
2. Add them to the main analysis functions
3. Update the dashboard description file
4. Test with different data scenarios

### Code Structure
- `load_doordash_data()`: Loads DoorDash CSV files
- `load_ubereats_data()`: Loads UberEats CSV files
- `display_data_info()`: Shows comprehensive data information
- `analyze_doordash_data()`: DoorDash-specific analysis
- `analyze_ubereats_data()`: UberEats-specific analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for internal use by QSR Executive Enterprises.

## 📞 Support

For technical support or questions about the dashboard:
- Check the troubleshooting section above
- Review the dashboard description file
- Contact the development team

---

**Last Updated**: December 2024
**Version**: 1.0.0
