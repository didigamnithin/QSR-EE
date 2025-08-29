import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import glob

# Page configuration
st.set_page_config(
    page_title="QSR Executive Enterprises Dashboard",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark mode styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #ffffff;
        margin-bottom: 2rem;
    }
    .platform-button {
        background-color: #1e1e1e;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #00ff88;
        margin: 0.5rem;
        color: #ffffff;
    }
    .metric-card {
        background-color: #2d2d2d;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #00ff88;
        margin: 0.5rem 0;
        color: #ffffff;
    }
    .data-info {
        background-color: #2d2d2d;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #404040;
        margin: 1rem 0;
        color: #ffffff;
    }
    .stApp {
        background-color: #0e1117;
    }
    .stButton > button {
        background-color: #00ff88;
        color: #000000;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #00cc6a;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_doordash_data():
    """Load all DoorDash data files"""
    data = {}
    
    # Load financial data
    financial_folder = "financial_2025-06-22_2025-08-22_SJnhV_2025-08-28T17-45-19Z"
    if os.path.exists(financial_folder):
        for file in os.listdir(financial_folder):
            if file.endswith('.csv'):
                try:
                    file_path = os.path.join(financial_folder, file)
                    df = pd.read_csv(file_path)
                    data[file] = df
                except Exception as e:
                    st.error(f"Error loading {file}: {str(e)}")
    
    # Load marketing data
    marketing_folder = "marketing_2025-06-22_2025-08-22_FLGlw_2025-08-28T17-46-33Z"
    if os.path.exists(marketing_folder):
        for file in os.listdir(marketing_folder):
            if file.endswith('.csv'):
                try:
                    file_path = os.path.join(marketing_folder, file)
                    df = pd.read_csv(file_path)
                    data[file] = df
                except Exception as e:
                    st.error(f"Error loading {file}: {str(e)}")
    
    # Load sales data
    sales_files = glob.glob("SALES_viewByOrder_*.csv")
    for file in sales_files:
        try:
            df = pd.read_csv(file)
            data[file] = df
        except Exception as e:
            st.error(f"Error loading {file}: {str(e)}")
    
    return data

@st.cache_data
def load_ubereats_data():
    """Load all UberEats data files"""
    data = {}
    
    ubereats_files = ['all_ads.csv', 'all_offers.csv', 'all_united.csv']
    
    for file in ubereats_files:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file)
                data[file] = df
            except Exception as e:
                st.error(f"Error loading {file}: {str(e)}")
    
    return data

def display_data_info(df, filename):
    """Display comprehensive data information"""
    st.markdown(f"### 📊 {filename}")
    
    # Display column names
    st.markdown("**Columns:**")
    columns_text = ", ".join(df.columns.tolist())
    st.text(columns_text)
    
    # Basic metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Rows", f"{len(df):,}")
    
    with col2:
        st.metric("Columns", f"{len(df.columns):,}")
    
    with col3:
        non_null_pct = (df.count().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Data Completeness", f"{non_null_pct:.1f}%")
    
    # Sample data
    st.markdown("**Sample Data (First 5 rows):**")
    st.dataframe(df.head(), use_container_width=True)

def analyze_sales_metrics(df, filename):
    """Analyze sales and order metrics over time"""
    st.markdown(f"### 📈 Sales Analysis - {filename}")
    
    # Identify date columns
    date_columns = []
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            date_columns.append(col)
    
    if not date_columns:
        st.warning("No date columns found for time series analysis")
        return
    
    # Identify sales/order related columns
    sales_columns = []
    order_columns = []
    revenue_columns = []
    
    for col in df.columns:
        col_lower = col.lower()
        # Check if column is numeric for better analysis
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        
        if 'sales' in col_lower or 'revenue' in col_lower or 'amount' in col_lower:
            if is_numeric:
                revenue_columns.append(col)
        elif 'order' in col_lower:
            if is_numeric:
                order_columns.append(col)
        elif any(term in col_lower for term in ['total', 'subtotal', 'net', 'gross']):
            if is_numeric:
                sales_columns.append(col)
    
    # Create 2-column layout for analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Available Metrics:**")
        if revenue_columns:
            st.markdown("💰 Revenue/Sales (Numeric):")
            for col in revenue_columns:
                st.markdown(f"- {col}")
        
        if order_columns:
            st.markdown("📦 Orders (Numeric):")
            for col in order_columns:
                st.markdown(f"- {col}")
        
        if sales_columns:
            st.markdown("💵 Sales Metrics (Numeric):")
            for col in sales_columns:
                st.markdown(f"- {col}")
        
        # Show data type information
        st.markdown("**Data Types:**")
        for col in df.columns:
            dtype = str(df[col].dtype)
            if pd.api.types.is_numeric_dtype(df[col]):
                st.markdown(f"- {col}: {dtype} (Numeric)")
            else:
                st.markdown(f"- {col}: {dtype} (Non-numeric)")
    
    with col2:
        st.markdown("**Date Columns:**")
        for col in date_columns:
            st.markdown(f"- {col}")
    
    # Time series analysis if we have date and metric columns
    if date_columns and (revenue_columns or order_columns):
        st.markdown("### 📊 Time Series Analysis")
        
        # Let user select date column and metric
        selected_date_col = st.selectbox("Select Date Column:", date_columns, key=f"date_{filename}")
        
        available_metrics = revenue_columns + order_columns + sales_columns
        if available_metrics:
            selected_metric = st.selectbox("Select Metric:", available_metrics, key=f"metric_{filename}")
            
            # Convert date column to datetime
            try:
                df_copy = df.copy()
                df_copy[selected_date_col] = pd.to_datetime(df_copy[selected_date_col], errors='coerce')
                df_copy = df_copy.dropna(subset=[selected_date_col, selected_metric])
                
                if len(df_copy) > 0:
                    # Check if the selected metric is numeric
                    if pd.api.types.is_numeric_dtype(df_copy[selected_metric]):
                        # Aggregate by date for numeric columns
                        daily_metrics = df_copy.groupby(df_copy[selected_date_col].dt.date)[selected_metric].agg(['sum', 'count', 'mean']).reset_index()
                        daily_metrics.columns = ['Date', 'Total', 'Count', 'Average']
                    else:
                        # For non-numeric columns, just count occurrences
                        daily_metrics = df_copy.groupby(df_copy[selected_date_col].dt.date)[selected_metric].agg(['count']).reset_index()
                        daily_metrics.columns = ['Date', 'Count']
                        daily_metrics['Total'] = daily_metrics['Count']  # For display purposes
                        daily_metrics['Average'] = daily_metrics['Count']  # For display purposes
                    
                    # Create time series chart
                    fig = px.line(daily_metrics, x='Date', y='Total', 
                                title=f'{selected_metric} Over Time',
                                labels={'Total': selected_metric, 'Date': 'Date'})
                    fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show aggregated data
                    st.markdown("**Daily Aggregated Data:**")
                    st.dataframe(daily_metrics, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total", f"{daily_metrics['Total'].sum():,.2f}")
                    with col2:
                        st.metric("Average Daily", f"{daily_metrics['Total'].mean():,.2f}")
                    with col3:
                        st.metric("Max Daily", f"{daily_metrics['Total'].max():,.2f}")
                    with col4:
                        st.metric("Min Daily", f"{daily_metrics['Total'].min():,.2f}")
                else:
                    st.warning("No valid data found after date conversion")
            except Exception as e:
                st.error(f"Error processing time series: {str(e)}")

def analyze_doordash_data(data):
    """Analyze DoorDash data"""
    st.markdown("## 🚚 DoorDash Data Analysis")
    
    if not data:
        st.warning("No DoorDash data found!")
        return
    
    # Display data information for each file
    for filename, df in data.items():
        display_data_info(df, filename)
        analyze_sales_metrics(df, filename)
        st.markdown("---")

def analyze_ubereats_data(data):
    """Analyze UberEats data"""
    st.markdown("## 🚗 UberEats Data Analysis")
    
    if not data:
        st.warning("No UberEats data found!")
        return
    
    # Display data information for each file
    for filename, df in data.items():
        display_data_info(df, filename)
        analyze_sales_metrics(df, filename)
        st.markdown("---")

def main():
    # Main header
    st.markdown('<h1 class="main-header">🍔 QSR Executive Enterprises Dashboard</h1>', unsafe_allow_html=True)
    
    # Platform selection buttons
    st.markdown("### Select Platform for Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        doordash_selected = st.button("🚚 DoorDash", use_container_width=True, key="doordash")
    
    with col2:
        ubereats_selected = st.button("🚗 UberEats", use_container_width=True, key="ubereats")
    
    # Load data based on selection
    if doordash_selected:
        with st.spinner("Loading DoorDash data..."):
            doordash_data = load_doordash_data()
        analyze_doordash_data(doordash_data)
    
    elif ubereats_selected:
        with st.spinner("Loading UberEats data..."):
            ubereats_data = load_ubereats_data()
        analyze_ubereats_data(ubereats_data)
    
    else:
        # Default view - show overview
        st.markdown("## 📋 Dashboard Overview")
        st.markdown("""
        Welcome to the QSR Executive Enterprises Dashboard! This dashboard provides comprehensive analytics 
        for your delivery platform performance across DoorDash and UberEats.
        
        **Select a platform above to begin your analysis:**
        
        - **🚚 DoorDash**: Financial data, marketing campaigns, and sales analytics
        - **🚗 UberEats**: Ad campaigns, offers, and unified sales data
        
        **Available Analysis:**
        - 📊 Basic data exploration (row counts, column info)
        - 📈 Time series analysis of sales and order metrics
        - 📋 Sample data preview
        - 📊 Daily aggregated metrics
        """)
        
        # Show summary statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🚚 DoorDash Data Files")
            try:
                doordash_data = load_doordash_data()
                if doordash_data:
                    for filename in doordash_data.keys():
                        st.markdown(f"- ✅ {filename}")
                else:
                    st.markdown("- ❌ No data files found")
            except:
                st.markdown("- ❌ Error loading data")
        
        with col2:
            st.markdown("### 🚗 UberEats Data Files")
            try:
                ubereats_data = load_ubereats_data()
                if ubereats_data:
                    for filename in ubereats_data.keys():
                        st.markdown(f"- ✅ {filename}")
                else:
                    st.markdown("- ❌ No data files found")
            except:
                st.markdown("- ❌ Error loading data")

if __name__ == "__main__":
    main()
