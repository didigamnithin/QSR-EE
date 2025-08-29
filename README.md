# 🚚 DoorDash Marketing Analysis Dashboard

A specialized Streamlit dashboard for comprehensive marketing analytics on the DoorDash platform, focusing on campaign performance, time series analysis, and store-level insights.

## 🎯 Features

### 📈 Time Series Analysis
- **Time on X-axis**: Date-based visualization
- **Multiple metrics on Y-axis** with smooth curve graphs:
  - Orders
  - Sales
  - Customer discounts (Funded by you)
  - Customer discounts (Funded by DoorDash)
  - Marketing fees (including taxes)
  - DoorDash marketing credit
  - Average Order Value (AOV)
  - ROAS
  - Customers acquired
- **3x3 subplot layout** for comprehensive visualization
- **Smooth spline curves** for better visual appeal

### 🎯 Campaign Comparison
- **TODC Campaign** (Self-serve = True)
- **Corporate Campaign** (Self-serve = False)
- Side-by-side performance comparison
- Marketing costs breakdown
- Detailed comparison tables

### 🏪 Store Filtering
- **Multi-select dropdown** for store selection
- **Default view**: All stores
- **Real-time filtering** based on store selection
- Individual or multiple store analysis

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Required packages (see `requirements.txt`)

### Installation
1. Clone or download the project files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Dashboard
```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## 📊 Data Sources

### DoorDash Marketing Data
- **File**: `marketing_2025-06-22_2025-08-22_FLGlw_2025-08-28T17-46-33Z/MARKETING_PROMOTION_*.csv`
- **Content**: Campaign performance metrics, store-level data, self-serve vs corporate campaign classification

## 🎨 Dashboard Sections

### 1. Sidebar Filters
- Collapsible right-side sidebar
- Store filter with multi-select dropdown
- Campaign filter with multi-select dropdown
- Real-time data filtering

### 2. Marketing Performance Over Time
- 9 interactive time series charts
- Daily aggregated metrics
- Smooth curve visualization
- Summary statistics

### 3. Campaign Performance Comparison (4 Categories)
- Pre-TODC Corporate vs Pre-TODC TODC vs Post-TODC Corporate vs Post-TODC TODC
- Orders, Sales, Marketing Fees, and ROAS comparison
- Marketing costs breakdown by period and campaign type
- Detailed comparison table with all 4 categories
- Key insights and best performing categories

### 4. Pre-TODC vs Post-TODC Period Comparison
- Period comparison table with all metrics
- Delta and percentage delta calculations

## 🔧 Technical Details

### Built With
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation and analysis
- **Dark Mode**: Professional styling

### Key Functions
- `load_marketing_data()`: Loads DoorDash marketing data
- `create_time_series_charts()`: Generates time series visualizations
- `create_campaign_comparison()`: Creates campaign comparison analysis
- `filter_data_by_stores()`: Handles store filtering

## 📈 Analytics Insights

The dashboard provides insights into:
- **Marketing performance trends** over time
- **Campaign type effectiveness** comparison across periods
- **Store-level performance** analysis
- **Marketing cost breakdown** and analysis
- **Customer acquisition patterns**
- **ROAS and AOV trends**
- **Pre vs Post-TODC period performance** comparison
- **Period-over-period growth** and decline analysis
- **Four-category campaign analysis** (Pre-TODC Corporate, Pre-TODC TODC, Post-TODC Corporate, Post-TODC TODC)
- **Sidebar filtering** for stores and campaigns

## 🎯 Use Cases

- **Marketing Managers**: Track campaign performance and ROI
- **Store Managers**: Analyze store-specific marketing effectiveness
- **Business Analysts**: Identify trends and patterns in marketing data
- **Executives**: High-level performance overview and comparison

## 🔍 Troubleshooting

### Common Issues
1. **Data not loading**: Ensure the marketing CSV file is in the correct folder
2. **Charts not displaying**: Check if the data contains the required columns
3. **Store filter not working**: Verify the 'Store name' column exists in the data

### Data Requirements
- CSV file with marketing promotion data
- Required columns: Date, Store name, Is self serve campaign, Orders, Sales, etc.
- Date format should be parseable by pandas

## 📝 File Structure
```
QSR-EE/
├── app.py                          # Main dashboard application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── dashboard_description.txt       # Detailed dashboard documentation
├── test_dashboard.py              # Testing script
└── marketing_*/                    # Marketing data folder
    └── MARKETING_PROMOTION_*.csv   # Marketing data file
```

## 🤝 Contributing

To enhance the dashboard:
1. Review the current features in `dashboard_description.txt`
2. Test changes using `test_dashboard.py`
3. Update documentation as needed

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the dashboard description file
3. Run the test script to verify data loading

---

**Note**: This dashboard is specifically designed for DoorDash marketing analysis. For other platforms or data sources, the code may need modifications.
