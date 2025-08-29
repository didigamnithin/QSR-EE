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
    page_title="QSR Executive Enterprises - DoorDash Marketing Analysis",
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
def load_marketing_data():
    """Load DoorDash marketing data"""
    data = {}
    
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
    
    return data

@st.cache_data
def load_financial_data():
    """Load DoorDash financial data"""
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
    
    return data

def get_available_stores(df):
    """Get list of available stores"""
    if 'Store name' in df.columns:
        stores = sorted(df['Store name'].unique())
        return ['All Stores'] + stores
    return ['All Stores']

def get_available_campaigns(df, selected_stores):
    """Get list of available campaigns based on selected stores"""
    if 'Campaign name' in df.columns:
        # Filter data by selected stores first
        filtered_df = filter_data_by_stores(df, selected_stores)
        campaigns = sorted(filtered_df['Campaign name'].unique())
        return ['All Campaigns'] + campaigns
    return ['All Campaigns']

def filter_data_by_stores(df, selected_stores):
    """Filter data by selected stores"""
    if 'All Stores' in selected_stores or not selected_stores:
        return df
    else:
        return df[df['Store name'].isin(selected_stores)]

def filter_data_by_campaigns(df, selected_campaigns):
    """Filter data by selected campaigns"""
    if 'All Campaigns' in selected_campaigns or not selected_campaigns:
        return df
    else:
        return df[df['Campaign name'].isin(selected_campaigns)]

def create_time_series_charts(df, selected_stores, selected_campaigns, data_type="Marketing"):
    """Create time series charts for marketing/sponsored metrics"""
    st.markdown(f"## 📈 {data_type} Performance Over Time")
    
    # Filter data by selected stores and campaigns
    filtered_df = filter_data_by_stores(df, selected_stores)
    filtered_df = filter_data_by_campaigns(filtered_df, selected_campaigns)
    
    # Convert date column to datetime
    if 'Date' in filtered_df.columns:
        filtered_df['Date'] = pd.to_datetime(filtered_df['Date'], errors='coerce')
        filtered_df = filtered_df.dropna(subset=['Date'])
        
        if len(filtered_df) == 0:
            st.warning("No valid data found after date conversion")
            return
        
        # Define metrics based on data type
        if data_type == "Marketing":
            metrics = [
                'Orders', 'Sales', 'Customer discounts from marketing | (Funded by you)',
                'Customer discounts from marketing | (Funded by DoorDash)', 
                'Marketing fees | (including any applicable taxes)', 'DoorDash marketing credit',
                'Average order value', 'ROAS', 'Total customers acquired'
            ]
        else:  # Sponsored
            metrics = [
                'Impressions', 'Clicks', 'Orders', 'Sales', 
                'Marketing fees | (including any applicable taxes)', 'DoorDash marketing credit',
                'Third-party contribution', 'Average order value', 'Average CPA', 'ROAS',
                'New customers acquired', 'Existing customers acquired', 'Total customers acquired'
            ]
        
        # Aggregate data by date
        agg_dict = {}
        for metric in metrics:
            if metric in filtered_df.columns:
                if metric in ['Average order value', 'Average CPA', 'ROAS']:
                    agg_dict[metric] = 'mean'
                else:
                    agg_dict[metric] = 'sum'
        
        daily_metrics = filtered_df.groupby('Date').agg(agg_dict).reset_index()
        
        # Create subplots - adjust based on number of metrics
        if data_type == "Marketing":
            rows, cols = 3, 3
            subplot_titles = ('Orders', 'Sales', 'Customer Discounts (You)', 
                            'Customer Discounts (DoorDash)', 'Marketing Fees', 'Marketing Credit',
                            'Average Order Value', 'ROAS', 'Customers Acquired')
        else:  # Sponsored
            rows, cols = 4, 3
            subplot_titles = ('Impressions', 'Clicks', 'Orders', 'Sales', 'Marketing Fees', 'Marketing Credit',
                            'Third-party Contribution', 'Average Order Value', 'Average CPA', 'ROAS',
                            'New Customers', 'Total Customers')
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=subplot_titles,
            specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)]
        )
        
        # Add traces for each metric
        colors = ['#00ff88', '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', 
                 '#ff9ff3', '#54a0ff', '#5f27cd', '#ff7675', '#74b9ff', '#a29bfe']
        
        # Only plot metrics that exist in the data and fit within the subplot grid
        available_metrics = [metric for metric in metrics if metric in daily_metrics.columns]
        max_plots = rows * cols
        
        for i, metric in enumerate(available_metrics[:max_plots]):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            fig.add_trace(
                    go.Scatter(
                        x=daily_metrics['Date'],
                        y=daily_metrics[metric],
                        mode='lines',
                        name=metric,
                        line=dict(color=colors[i % len(colors)], width=2, shape='spline'),
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            height=900 if data_type == "Marketing" else 1200,
            title_text=f"{data_type} Performance Metrics Over Time",
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        # Update all subplot axes
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)', row=i, col=j)
                fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)', row=i, col=j)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show summary statistics
        st.markdown("### 📊 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'Orders' in daily_metrics.columns:
                st.metric("Total Orders", f"{daily_metrics['Orders'].sum():,.0f}")
            elif 'Impressions' in daily_metrics.columns:
                st.metric("Total Impressions", f"{daily_metrics['Impressions'].sum():,.0f}")
        with col2:
            if 'Sales' in daily_metrics.columns:
                st.metric("Total Sales", f"${daily_metrics['Sales'].sum():,.2f}")
            elif 'Clicks' in daily_metrics.columns:
                st.metric("Total Clicks", f"{daily_metrics['Clicks'].sum():,.0f}")
        with col3:
            if 'Marketing fees | (including any applicable taxes)' in daily_metrics.columns:
                st.metric("Total Marketing Fees", f"${daily_metrics['Marketing fees | (including any applicable taxes)'].sum():,.2f}")
            elif 'ROAS' in daily_metrics.columns:
                st.metric("Average ROAS", f"{daily_metrics['ROAS'].mean():,.2f}")
        with col4:
            if 'Total customers acquired' in daily_metrics.columns:
                st.metric("Total Customers Acquired", f"{daily_metrics['Total customers acquired'].sum():,.0f}")
            elif 'Average CPA' in daily_metrics.columns:
                st.metric("Average CPA", f"${daily_metrics['Average CPA'].mean():,.2f}")

def create_campaign_comparison(df, selected_stores, selected_campaigns, data_type="Marketing"):
    """Create campaign comparison analysis for both periods"""
    st.markdown(f"## 🎯 {data_type} Campaign Performance Comparison (Pre-TODC vs Post-TODC)")
    
    # Filter data by selected stores and campaigns
    filtered_df = filter_data_by_stores(df, selected_stores)
    filtered_df = filter_data_by_campaigns(filtered_df, selected_campaigns)
    
    # Convert date column to datetime
    if 'Date' in filtered_df.columns:
        filtered_df['Date'] = pd.to_datetime(filtered_df['Date'], errors='coerce')
        filtered_df = filtered_df.dropna(subset=['Date'])
        
        if len(filtered_df) == 0:
            st.warning("No valid data found after date conversion")
            return
        
        # Define periods
        pre_todc_start = pd.to_datetime('2025-06-22')
        pre_todc_end = pd.to_datetime('2025-07-21')
        post_todc_start = pd.to_datetime('2025-07-22')
        post_todc_end = pd.to_datetime('2025-08-22')
        
        # Filter data for each period
        pre_todc_data = filtered_df[(filtered_df['Date'] >= pre_todc_start) & (filtered_df['Date'] <= pre_todc_end)]
        post_todc_data = filtered_df[(filtered_df['Date'] >= post_todc_start) & (filtered_df['Date'] <= post_todc_end)]
        
        # Map campaign types
        pre_todc_data['Campaign Type'] = pre_todc_data['Is self serve campaign'].map({
            True: 'TODC Campaign',
            False: 'Corporate Campaign'
        })
        post_todc_data['Campaign Type'] = post_todc_data['Is self serve campaign'].map({
            True: 'TODC Campaign',
            False: 'Corporate Campaign'
        })
        
        # Add period labels
        pre_todc_data['Period'] = 'Pre-TODC'
        post_todc_data['Period'] = 'Post-TODC'
        
        # Combine data
        combined_data = pd.concat([pre_todc_data, post_todc_data], ignore_index=True)
        
        # Create combined category
        combined_data['Period_Campaign'] = combined_data['Period'] + ' - ' + combined_data['Campaign Type']
        
        # Define metrics based on data type
        if data_type == "Marketing":
            metrics = [
                'Orders', 'Sales', 'Customer discounts from marketing | (Funded by you)',
                'Customer discounts from marketing | (Funded by DoorDash)', 
                'Marketing fees | (including any applicable taxes)', 'DoorDash marketing credit',
                'Average order value', 'ROAS', 'Total customers acquired'
            ]
        else:  # Sponsored
            metrics = [
                'Impressions', 'Clicks', 'Orders', 'Sales', 
                'Marketing fees | (including any applicable taxes)', 'DoorDash marketing credit',
                'Third-party contribution', 'Average order value', 'Average CPA', 'ROAS',
                'New customers acquired', 'Existing customers acquired', 'Total customers acquired'
            ]
        
        # Aggregate by period and campaign type
        agg_dict = {}
        for metric in metrics:
            if metric in combined_data.columns:
                if metric in ['Average order value', 'Average CPA', 'ROAS']:
                    agg_dict[metric] = 'mean'
                else:
                    agg_dict[metric] = 'sum'
        
        campaign_metrics = combined_data.groupby(['Period', 'Campaign Type']).agg(agg_dict).reset_index()
        
        # Create combined category for display
        campaign_metrics['Period_Campaign'] = campaign_metrics['Period'] + ' - ' + campaign_metrics['Campaign Type']
    
    # Create comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Orders comparison across all 4 categories
        fig1 = go.Figure()
        
        # Define colors for each category
        colors = {
            'Pre-TODC - Corporate Campaign': '#ff6b6b',
            'Pre-TODC - TODC Campaign': '#4ecdc4',
            'Post-TODC - Corporate Campaign': '#45b7d1',
            'Post-TODC - TODC Campaign': '#00ff88'
        }
        
        metric_name = 'Orders' if 'Orders' in campaign_metrics.columns else 'Impressions'
        
        for period_campaign in campaign_metrics['Period_Campaign'].unique():
            data = campaign_metrics[campaign_metrics['Period_Campaign'] == period_campaign]
            if len(data) > 0 and metric_name in data.columns:
                fig1.add_trace(go.Bar(
                    x=[metric_name],
                    y=data[metric_name],
                    name=period_campaign,
                    marker_color=colors.get(period_campaign, '#cccccc')
                ))
        
        fig1.update_layout(
            title=f'{metric_name} by Period and Campaign Type',
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Sales comparison across all 4 categories
        fig2 = go.Figure()
        
        metric_name = 'Sales' if 'Sales' in campaign_metrics.columns else 'Clicks'
        
        for period_campaign in campaign_metrics['Period_Campaign'].unique():
            data = campaign_metrics[campaign_metrics['Period_Campaign'] == period_campaign]
            if len(data) > 0 and metric_name in data.columns:
                fig2.add_trace(go.Bar(
                    x=[metric_name],
                    y=data[metric_name],
                    name=period_campaign,
                    marker_color=colors.get(period_campaign, '#cccccc')
                ))
        
        fig2.update_layout(
            title=f'{metric_name} by Period and Campaign Type',
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    # Create additional charts for marketing metrics
    st.markdown("### 📊 Marketing Costs Comparison")
    col1, col2 = st.columns(2)
    
    with col1:
        # Marketing fees comparison
        fig3 = go.Figure()
        
        for period_campaign in campaign_metrics['Period_Campaign'].unique():
            data = campaign_metrics[campaign_metrics['Period_Campaign'] == period_campaign]
            if len(data) > 0 and 'Marketing fees | (including any applicable taxes)' in data.columns:
                fig3.add_trace(go.Bar(
                    x=['Marketing Fees'],
                    y=data['Marketing fees | (including any applicable taxes)'],
                    name=period_campaign,
                    marker_color=colors.get(period_campaign, '#cccccc')
                ))
        
        fig3.update_layout(
            title='Marketing Fees by Period and Campaign Type',
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # ROAS comparison
        fig4 = go.Figure()
        
        for period_campaign in campaign_metrics['Period_Campaign'].unique():
            data = campaign_metrics[campaign_metrics['Period_Campaign'] == period_campaign]
            if len(data) > 0 and 'ROAS' in data.columns:
                fig4.add_trace(go.Bar(
                    x=['ROAS'],
                    y=data['ROAS'],
                    name=period_campaign,
                    marker_color=colors.get(period_campaign, '#cccccc')
                ))
        
        fig4.update_layout(
            title='ROAS by Period and Campaign Type',
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig4, use_container_width=True)
    
    # Show detailed comparison table
    st.markdown("### 📋 Detailed Campaign Comparison (4 Categories)")
    
    # Format the comparison table
    comparison_table = campaign_metrics.copy()
    
    # Format numeric columns
    comparison_table = comparison_table.copy()  # Create a copy to avoid SettingWithCopyWarning
    for i, row in comparison_table.iterrows():
        for col in comparison_table.columns:
            if col not in ['Period', 'Campaign Type', 'Period_Campaign']:
                value = row[col]
                if pd.notna(value):
                    if any(keyword in col.lower() for keyword in ['sales', 'discounts', 'fees', 'credit', 'value', 'contribution']):
                        comparison_table.loc[i, col] = f"${value:,.2f}"
                    elif col in ['ROAS', 'Average CPA']:
                        comparison_table.loc[i, col] = f"{value:.2f}"
                    elif pd.notna(value) and value == int(value):
                        comparison_table.loc[i, col] = f"{value:,.0f}"
                    else:
                        comparison_table.loc[i, col] = f"{value:,.2f}"
    
    # Reorder columns for better display
    display_columns = ['Period', 'Campaign Type', 'Period_Campaign'] + [col for col in comparison_table.columns if col not in ['Period', 'Campaign Type', 'Period_Campaign']]
    comparison_table = comparison_table[display_columns]
    
    st.dataframe(comparison_table, use_container_width=True)
    

def create_period_comparison(df, selected_stores, selected_campaigns, data_type="Marketing"):
    """Create Pre-TODC vs Post-TODC period comparison"""
    st.markdown(f"## 📊 {data_type} Pre-TODC vs Post-TODC Period Comparison")
    
    # Filter data by selected stores and campaigns
    filtered_df = filter_data_by_stores(df, selected_stores)
    filtered_df = filter_data_by_campaigns(filtered_df, selected_campaigns)
    
    # Convert date column to datetime
    if 'Date' in filtered_df.columns:
        filtered_df['Date'] = pd.to_datetime(filtered_df['Date'], errors='coerce')
        filtered_df = filtered_df.dropna(subset=['Date'])
        
        if len(filtered_df) == 0:
            st.warning("No valid data found after date conversion")
            return
        
        # Define periods
        pre_todc_start = pd.to_datetime('2025-06-22')
        pre_todc_end = pd.to_datetime('2025-07-21')
        post_todc_start = pd.to_datetime('2025-07-22')
        post_todc_end = pd.to_datetime('2025-08-22')
        
        # Filter data for each period
        pre_todc_data = filtered_df[(filtered_df['Date'] >= pre_todc_start) & (filtered_df['Date'] <= pre_todc_end)]
        post_todc_data = filtered_df[(filtered_df['Date'] >= post_todc_start) & (filtered_df['Date'] <= post_todc_end)]
        
        # Define metrics based on data type
        if data_type == "Marketing":
            metrics = [
                'Orders', 'Sales', 'Customer discounts from marketing | (Funded by you)',
                'Customer discounts from marketing | (Funded by DoorDash)', 
                'Marketing fees | (including any applicable taxes)', 'DoorDash marketing credit',
                'Average order value', 'ROAS', 'Total customers acquired'
            ]
        elif data_type == "Sponsored":
            metrics = [
                'Impressions', 'Clicks', 'Orders', 'Sales', 
                'Marketing fees | (including any applicable taxes)', 'DoorDash marketing credit',
                'Third-party contribution', 'Average order value', 'Average CPA', 'ROAS',
                'New customers acquired', 'Existing customers acquired', 'Total customers acquired'
            ]
        else:  # Financials
            metrics = [
                'Gross sales', 'Net sales', 'Commission', 'Promotional credits',
                'Net payout', 'Total orders', 'Average order value'
            ]
        
        # Calculate metrics for each period
        comparison_data = []
        
        for metric in metrics:
            if metric in filtered_df.columns:
                if metric in ['Average order value', 'Average CPA', 'ROAS']:
                    pre_value = pre_todc_data[metric].mean()
                    post_value = post_todc_data[metric].mean()
                else:
                    pre_value = pre_todc_data[metric].sum()
                    post_value = post_todc_data[metric].sum()
                
                # Calculate delta and percentage delta
                delta = post_value - pre_value
                pct_delta = (delta / pre_value * 100) if pre_value != 0 else 0
                
                comparison_data.append({
                    'Metric': metric,
                    'Pre-TODC (6/22-7/21)': pre_value,
                    'Post-TODC (7/22-8/22)': post_value,
                    'Delta': delta,
                    '% Delta': pct_delta
                })
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame(comparison_data)
        
        # Format the display table
        display_df = comparison_df.copy()
        
        # Format numeric columns
        display_df = display_df.copy()  # Create a copy to avoid SettingWithCopyWarning
        for i, row in display_df.iterrows():
            metric = row['Metric']
            for col in ['Pre-TODC (6/22-7/21)', 'Post-TODC (7/22-8/22)', 'Delta']:
                value = row[col]
                if any(keyword in metric.lower() for keyword in ['sales', 'discounts', 'fees', 'credit', 'value', 'contribution', 'payout', 'commission']):
                    display_df.loc[i, col] = f"${value:,.2f}"
                elif pd.notna(value) and value == int(value):
                    display_df.loc[i, col] = f"{value:,.0f}"
                else:
                    display_df.loc[i, col] = f"{value:,.2f}"
        
        display_df['% Delta'] = display_df['% Delta'].apply(lambda x: f"{x:+.1f}%" if x != 0 else "0.0%")
        
        # Show the comparison table
        st.markdown("### 📈 Period Comparison Table")
        st.dataframe(display_df, use_container_width=True)

def create_financial_analysis(df, selected_stores, selected_campaigns):
    """Create financial analysis with detailed transactions data"""
    st.markdown("## 💰 Financial Analysis - Detailed Transactions")
    
    # Filter data by selected stores
    filtered_df = filter_data_by_stores(df, selected_stores)
    
    # Convert Timestamp UTC date to datetime
    if 'Timestamp UTC date' in filtered_df.columns:
        filtered_df['Timestamp UTC date'] = pd.to_datetime(filtered_df['Timestamp UTC date'], errors='coerce')
        filtered_df = filtered_df.dropna(subset=['Timestamp UTC date'])
        
        if len(filtered_df) == 0:
            st.warning("No valid data found after date conversion")
            return
        
        # Extract date from timestamp
        filtered_df['Date'] = filtered_df['Timestamp UTC date'].dt.date
        filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])
        
        # Show available columns for debugging
        st.markdown("### 📋 Available Financial Columns")
        st.write("Available columns in the financial data:")
        st.write(list(filtered_df.columns))
        
        # 1. ORDER TRANSACTIONS ANALYSIS
        st.markdown("## 📦 Order Transactions Analysis")
        
        # Filter for Order transactions
        order_df = filtered_df[filtered_df['Transaction type'] == 'Order'].copy()
        
        if len(order_df) > 0:
            # Define order metrics
            order_metrics = ['Subtotal', 'Commission', 'Marketing fees', 'Discounts', 'Net pay']
            available_order_metrics = [col for col in order_metrics if col in order_df.columns]
            
            # Aggregate order data by date
            agg_dict = {}
            for metric in available_order_metrics:
                if metric in order_df.columns and pd.api.types.is_numeric_dtype(order_df[metric]):
                    agg_dict[metric] = 'sum'
            
            # Add count of orders
            agg_dict['Orders'] = 'count'
            
            if agg_dict:
                daily_orders = order_df.groupby('Date').agg(agg_dict).reset_index()
                
                # Show order summary statistics
                st.markdown("### 📊 Order Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Orders", f"{daily_orders['Orders'].sum():,.0f}")
                with col2:
                    if 'Subtotal' in daily_orders.columns:
                        st.metric("Total Subtotal", f"${daily_orders['Subtotal'].sum():,.2f}")
                with col3:
                    if 'Net pay' in daily_orders.columns:
                        st.metric("Total Net Pay", f"${daily_orders['Net pay'].sum():,.2f}")
                with col4:
                    if 'Commission' in daily_orders.columns:
                        st.metric("Total Commission", f"${daily_orders['Commission'].sum():,.2f}")
                
                # Create time series chart for order metrics
                st.markdown("### 📈 Order Metrics Over Time")
                
                # Create subplots for order metrics
                display_metrics = ['Orders'] + [col for col in available_order_metrics if col in daily_orders.columns]
                
                if len(display_metrics) <= 3:
                    rows, cols = 1, len(display_metrics)
                elif len(display_metrics) <= 6:
                    rows, cols = 2, 3
                else:
                    rows, cols = 3, 2
                
                fig = make_subplots(
                    rows=rows, cols=cols,
                    subplot_titles=display_metrics,
                    specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)]
                )
                
                # Add traces for each metric
                colors = ['#00ff88', '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
                
                for i, metric in enumerate(display_metrics):
                    row = (i // cols) + 1
                    col = (i % cols) + 1
                    
                    fig.add_trace(
                        go.Scatter(
                            x=daily_orders['Date'],
                            y=daily_orders[metric],
                            mode='lines',
                            name=metric,
                            line=dict(color=colors[i % len(colors)], width=2, shape='spline'),
                            showlegend=False
                        ),
                        row=row, col=col
                    )
                
                fig.update_layout(
                    height=300 * rows,
                    title_text="Order Metrics Over Time",
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                # Update all subplot axes
                for i in range(1, rows + 1):
                    for j in range(1, cols + 1):
                        fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)', row=i, col=j)
                        fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)', row=i, col=j)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Pre-TODC vs Post-TODC Analysis for Orders
                st.markdown("### 📊 Pre-TODC vs Post-TODC Order Analysis")
                
                # Define periods
                pre_todc_start = pd.to_datetime('2025-06-22')
                pre_todc_end = pd.to_datetime('2025-07-21')
                post_todc_start = pd.to_datetime('2025-07-22')
                post_todc_end = pd.to_datetime('2025-08-22')
                
                # Filter data for each period
                pre_todc_orders = daily_orders[(daily_orders['Date'] >= pre_todc_start) & (daily_orders['Date'] <= pre_todc_end)]
                post_todc_orders = daily_orders[(daily_orders['Date'] >= post_todc_start) & (daily_orders['Date'] <= post_todc_end)]
                
                # Calculate comparison data
                comparison_data = []
                
                for metric in display_metrics:
                    if metric in daily_orders.columns:
                        pre_value = pre_todc_orders[metric].sum()
                        post_value = post_todc_orders[metric].sum()
                        
                        # Calculate delta and percentage delta
                        delta = post_value - pre_value
                        pct_delta = (delta / pre_value * 100) if pre_value != 0 else 0
                        
                        comparison_data.append({
                            'Metric': metric,
                            'Pre-TODC (6/22-7/21)': pre_value,
                            'Post-TODC (7/22-8/22)': post_value,
                            'Delta': delta,
                            '% Delta': pct_delta
                        })
                
                # Create comparison dataframe
                comparison_df = pd.DataFrame(comparison_data)
                
                # Format the display table
                display_df = comparison_df.copy()
                
                # Format numeric columns
                display_df = display_df.copy()
                for i, row in display_df.iterrows():
                    metric = row['Metric']
                    for col in ['Pre-TODC (6/22-7/21)', 'Post-TODC (7/22-8/22)', 'Delta']:
                        value = row[col]
                        if any(keyword in metric.lower() for keyword in ['subtotal', 'commission', 'fees', 'discounts', 'pay', 'amount']):
                            display_df.loc[i, col] = f"${value:,.2f}"
                        elif pd.notna(value) and value == int(value):
                            display_df.loc[i, col] = f"{value:,.0f}"
                        else:
                            display_df.loc[i, col] = f"{value:,.2f}"
                
                display_df['% Delta'] = display_df['% Delta'].apply(lambda x: f"{x:+.1f}%" if x != 0 else "0.0%")
                
                # Show the comparison table
                st.dataframe(display_df, use_container_width=True)
                
                # Show daily order data
                st.markdown("### 📋 Daily Order Data")
                
                # Format the table for display
                display_df = daily_orders.copy()
                
                # Format numeric columns
                display_df = display_df.copy()
                for i, row in display_df.iterrows():
                    for col in display_df.columns:
                        if col != 'Date':
                            value = row[col]
                            if pd.notna(value):
                                if any(keyword in col.lower() for keyword in ['subtotal', 'commission', 'fees', 'discounts', 'pay', 'amount']):
                                    display_df.loc[i, col] = f"${value:,.2f}"
                                elif pd.notna(value) and value == int(value):
                                    display_df.loc[i, col] = f"{value:,.0f}"
                                else:
                                    display_df.loc[i, col] = f"{value:,.2f}"
                
                st.dataframe(display_df, use_container_width=True)
        else:
            st.warning("No Order transactions found in the data")
        
        # 2. ERROR CHARGES AND ADJUSTMENTS ANALYSIS
        st.markdown("## ⚠️ Error Charges and Adjustments Analysis")
        
        # Filter for Error Charge and Adjustment transactions
        error_df = filtered_df[filtered_df['Transaction type'].isin(['Error Charge', 'Adjustment'])].copy()
        
        if len(error_df) > 0:
            # Define error metrics
            error_metrics = ['Subtotal', 'Commission', 'Marketing fees', 'Discounts', 'Net pay']
            available_error_metrics = [col for col in error_metrics if col in error_df.columns]
            
            # Aggregate error data by date
            agg_dict = {}
            for metric in available_error_metrics:
                if metric in error_df.columns and pd.api.types.is_numeric_dtype(error_df[metric]):
                    agg_dict[metric] = 'sum'
            
            # Add count of errors
            agg_dict['Error Count'] = 'count'
            
            if agg_dict:
                daily_errors = error_df.groupby('Date').agg(agg_dict).reset_index()
                
                # Show error summary statistics
                st.markdown("### 📊 Error Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Errors", f"{daily_errors['Error Count'].sum():,.0f}")
                with col2:
                    if 'Subtotal' in daily_errors.columns:
                        st.metric("Total Error Amount", f"${daily_errors['Subtotal'].sum():,.2f}")
                with col3:
                    if 'Net pay' in daily_errors.columns:
                        st.metric("Total Net Impact", f"${daily_errors['Net pay'].sum():,.2f}")
                with col4:
                    if 'Commission' in daily_errors.columns:
                        st.metric("Total Commission Impact", f"${daily_errors['Commission'].sum():,.2f}")
                
                # Create time series chart for error metrics
                st.markdown("### 📈 Error Metrics Over Time")
                
                # Create subplots for error metrics
                display_metrics = ['Error Count'] + [col for col in available_error_metrics if col in daily_errors.columns]
                
                if len(display_metrics) <= 3:
                    rows, cols = 1, len(display_metrics)
                elif len(display_metrics) <= 6:
                    rows, cols = 2, 3
                else:
                    rows, cols = 3, 2
                
                fig = make_subplots(
                    rows=rows, cols=cols,
                    subplot_titles=display_metrics,
                    specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)]
                )
                
                # Add traces for each metric
                colors = ['#ff6b6b', '#ff4757', '#ff3838', '#ff5252', '#ff1744', '#d50000']
                
                for i, metric in enumerate(display_metrics):
                    row = (i // cols) + 1
                    col = (i % cols) + 1
                    
                    fig.add_trace(
                        go.Scatter(
                            x=daily_errors['Date'],
                            y=daily_errors[metric],
                            mode='lines',
                            name=metric,
                            line=dict(color=colors[i % len(colors)], width=2, shape='spline'),
                            showlegend=False
                        ),
                        row=row, col=col
                    )
                
                fig.update_layout(
                    height=300 * rows,
                    title_text="Error Metrics Over Time",
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                # Update all subplot axes
                for i in range(1, rows + 1):
                    for j in range(1, cols + 1):
                        fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)', row=i, col=j)
                        fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)', row=i, col=j)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Pre-TODC vs Post-TODC Analysis for Errors
                st.markdown("### 📊 Pre-TODC vs Post-TODC Error Analysis")
                
                # Filter data for each period
                pre_todc_errors = daily_errors[(daily_errors['Date'] >= pre_todc_start) & (daily_errors['Date'] <= pre_todc_end)]
                post_todc_errors = daily_errors[(daily_errors['Date'] >= post_todc_start) & (daily_errors['Date'] <= post_todc_end)]
                
                # Calculate comparison data
                comparison_data = []
                
                for metric in display_metrics:
                    if metric in daily_errors.columns:
                        pre_value = pre_todc_errors[metric].sum()
                        post_value = post_todc_errors[metric].sum()
                        
                        # Calculate delta and percentage delta
                        delta = post_value - pre_value
                        pct_delta = (delta / pre_value * 100) if pre_value != 0 else 0
                        
                        comparison_data.append({
                            'Metric': metric,
                            'Pre-TODC (6/22-7/21)': pre_value,
                            'Post-TODC (7/22-8/22)': post_value,
                            'Delta': delta,
                            '% Delta': pct_delta
                        })
                
                # Create comparison dataframe
                comparison_df = pd.DataFrame(comparison_data)
                
                # Format the display table
                display_df = comparison_df.copy()
                
                # Format numeric columns
                display_df = display_df.copy()
                for i, row in display_df.iterrows():
                    metric = row['Metric']
                    for col in ['Pre-TODC (6/22-7/21)', 'Post-TODC (7/22-8/22)', 'Delta']:
                        value = row[col]
                        if any(keyword in metric.lower() for keyword in ['subtotal', 'commission', 'fees', 'discounts', 'pay', 'amount']):
                            display_df.loc[i, col] = f"${value:,.2f}"
                        elif pd.notna(value) and value == int(value):
                            display_df.loc[i, col] = f"{value:,.0f}"
                        else:
                            display_df.loc[i, col] = f"{value:,.2f}"
                
                display_df['% Delta'] = display_df['% Delta'].apply(lambda x: f"{x:+.1f}%" if x != 0 else "0.0%")
                
                # Show the comparison table
                st.dataframe(display_df, use_container_width=True)
        else:
            st.warning("No Error Charge or Adjustment transactions found in the data")
        
        # 3. FINAL ORDER STATUS ANALYSIS
        st.markdown("## 📋 Final Order Status Analysis")
        
        if 'Final order status' in filtered_df.columns:
            # Daily order status counts
            status_df = filtered_df[filtered_df['Transaction type'] == 'Order'].copy()
            
            if len(status_df) > 0:
                # Get daily status counts
                daily_status = status_df.groupby(['Date', 'Final order status']).size().reset_index(name='Count')
                
                # Pivot to get status as columns
                status_pivot = daily_status.pivot(index='Date', columns='Final order status', values='Count').fillna(0)
                status_pivot = status_pivot.reset_index()
                
                # Show status summary
                st.markdown("### 📊 Order Status Summary")
                total_by_status = status_df['Final order status'].value_counts()
                
                col1, col2, col3, col4 = st.columns(4)
                status_list = list(total_by_status.index)
                
                for i, status in enumerate(status_list[:4]):
                    with [col1, col2, col3, col4][i]:
                        st.metric(f"Total {status}", f"{total_by_status[status]:,.0f}")
                
                # Create time series chart for order status
                st.markdown("### 📈 Order Status Over Time")
                
                # Create subplots for each status
                status_columns = [col for col in status_pivot.columns if col != 'Date']
                
                if len(status_columns) <= 3:
                    rows, cols = 1, len(status_columns)
                elif len(status_columns) <= 6:
                    rows, cols = 2, 3
                else:
                    rows, cols = 3, 2
                
                fig = make_subplots(
                    rows=rows, cols=cols,
                    subplot_titles=status_columns,
                    specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows)]
                )
                
                # Add traces for each status
                colors = ['#00ff88', '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57']
                
                for i, status in enumerate(status_columns):
                    row = (i // cols) + 1
                    col = (i % cols) + 1
                    
                    fig.add_trace(
                        go.Scatter(
                            x=status_pivot['Date'],
                            y=status_pivot[status],
                            mode='lines',
                            name=status,
                            line=dict(color=colors[i % len(colors)], width=2, shape='spline'),
                            showlegend=False
                        ),
                        row=row, col=col
                    )
                
                fig.update_layout(
                    height=300 * rows,
                    title_text="Order Status Over Time",
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                
                # Update all subplot axes
                for i in range(1, rows + 1):
                    for j in range(1, cols + 1):
                        fig.update_xaxes(gridcolor='rgba(128,128,128,0.2)', row=i, col=j)
                        fig.update_yaxes(gridcolor='rgba(128,128,128,0.2)', row=i, col=j)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show daily status data
                st.markdown("### 📋 Daily Order Status Data")
                st.dataframe(status_pivot, use_container_width=True)
            else:
                st.warning("No Order transactions found for status analysis")
        else:
            st.warning("'Final order status' column not found in the data")
    else:
        st.error("'Timestamp UTC date' column not found in the financial data")

def main():
    # Main header
    st.markdown('<h1 class="main-header">🚚 DoorDash Marketing Analysis</h1>', unsafe_allow_html=True)
    
    # Load data based on type
    with st.spinner("Loading data..."):
        marketing_data = load_marketing_data()
        financial_data = load_financial_data()
    
    if not marketing_data and not financial_data:
        st.error("No data found!")
        return
    
    # Sidebar filters
    with st.sidebar:
        st.markdown("## 🔧 Filters")
        
        # Data type selector
        st.markdown("### 📊 Data Type")
        data_type = st.selectbox(
            "Select Data Type",
            options=["Marketing", "Sponsored", "Financials"],
            help="Choose between Marketing Promotion data, Sponsored Listing data, or Financial data."
        )
        
        # Get the appropriate data file
        if data_type in ["Marketing", "Sponsored"]:
            if data_type == "Marketing":
                data_file = None
                for filename in marketing_data.keys():
                    if 'PROMOTION' in filename:
                        data_file = filename
                        break
            else:  # Sponsored
                data_file = None
                for filename in marketing_data.keys():
                    if 'SPONSORED' in filename:
                        data_file = filename
                        break
            
            if not data_file:
                st.error(f"{data_type} data not found!")
                return
            
            df = marketing_data[data_file]
        else:  # Financials
            data_file = None
            for filename in financial_data.keys():
                if 'FINANCIAL_DETAILED_TRANSACTIONS' in filename:
                    data_file = filename
                    break
            
            if not data_file:
                st.error("Financial data not found!")
                return
            
            df = financial_data[data_file]
        
        # Store filter
        st.markdown("### 🏪 Store Filter")
        available_stores = get_available_stores(df)
        selected_stores = st.multiselect(
            "Select Stores",
            options=available_stores,
            default=['All Stores'],
            help="Select specific stores to filter the analysis."
        )
        
        if not selected_stores:
            selected_stores = ['All Stores']
        
        # Campaign filter (only for marketing/sponsored data)
        if data_type in ["Marketing", "Sponsored"]:
            st.markdown("### 🎯 Campaign Filter")
            available_campaigns = get_available_campaigns(df, selected_stores)
            selected_campaigns = st.multiselect(
                "Select Campaigns",
                options=available_campaigns,
                default=['All Campaigns'],
                help="Select specific campaigns to filter the analysis."
            )
            
            if not selected_campaigns:
                selected_campaigns = ['All Campaigns']
        else:  # Financials
            st.markdown("### 🏪 Store Filter (Financials)")
            # For financials, we might have different store column names
            if 'Store name' in df.columns:
                available_stores = get_available_stores(df)
            elif 'Store' in df.columns:
                stores = sorted(df['Store'].unique())
                available_stores = ['All Stores'] + stores
            else:
                available_stores = ['All Stores']
            
            selected_stores = st.multiselect(
                "Select Stores",
                options=available_stores,
                default=['All Stores'],
                help="Select specific stores to filter the financial analysis."
            )
            
            if not selected_stores:
                selected_stores = ['All Stores']
            
            selected_campaigns = ['All Campaigns']  # Not applicable for financials
    
    # Show tables first, then graphs
    if data_type == "Financials":
        # Financial analysis
        create_financial_analysis(df, selected_stores, selected_campaigns)
    else:
        # Marketing/Sponsored analysis - show tables first
        create_period_comparison(df, selected_stores, selected_campaigns, data_type)
        create_campaign_comparison(df, selected_stores, selected_campaigns, data_type)
        
        # Then show graphs
        create_time_series_charts(df, selected_stores, selected_campaigns, data_type)

if __name__ == "__main__":
    main()
