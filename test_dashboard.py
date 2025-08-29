#!/usr/bin/env python3
"""
Test script for QSR Executive Enterprises Dashboard
This script tests the data loading and basic functionality
"""

import pandas as pd
import os
import glob

def test_data_loading():
    """Test data loading functions"""
    print("🧪 Testing Dashboard Data Loading...")
    print("=" * 50)
    
    # Test UberEats data
    print("\n🚗 Testing UberEats Data:")
    ubereats_files = ['all_ads.csv', 'all_offers.csv', 'all_united.csv']
    ubereats_data = {}
    
    for file in ubereats_files:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file)
                ubereats_data[file] = df
                print(f"✅ {file}: {len(df):,} rows, {len(df.columns)} columns")
            except Exception as e:
                print(f"❌ {file}: Error - {str(e)}")
        else:
            print(f"❌ {file}: File not found")
    
    # Test DoorDash data
    print("\n🚚 Testing DoorDash Data:")
    
    # Financial data
    financial_folder = "financial_2025-06-22_2025-08-22_SJnhV_2025-08-28T17-45-19Z"
    if os.path.exists(financial_folder):
        print(f"📁 Financial folder: {financial_folder}")
        for file in os.listdir(financial_folder):
            if file.endswith('.csv'):
                try:
                    file_path = os.path.join(financial_folder, file)
                    df = pd.read_csv(file_path)
                    print(f"✅ {file}: {len(df):,} rows, {len(df.columns)} columns")
                except Exception as e:
                    print(f"❌ {file}: Error - {str(e)}")
    else:
        print(f"❌ Financial folder not found: {financial_folder}")
    
    # Marketing data
    marketing_folder = "marketing_2025-06-22_2025-08-22_FLGlw_2025-08-28T17-46-33Z"
    if os.path.exists(marketing_folder):
        print(f"📁 Marketing folder: {marketing_folder}")
        for file in os.listdir(marketing_folder):
            if file.endswith('.csv'):
                try:
                    file_path = os.path.join(marketing_folder, file)
                    df = pd.read_csv(file_path)
                    print(f"✅ {file}: {len(df):,} rows, {len(df.columns)} columns")
                except Exception as e:
                    print(f"❌ {file}: Error - {str(e)}")
    else:
        print(f"❌ Marketing folder not found: {marketing_folder}")
    
    # Sales data
    sales_files = glob.glob("SALES_viewByOrder_*.csv")
    if sales_files:
        print(f"📁 Sales files found: {len(sales_files)}")
        for file in sales_files:
            try:
                df = pd.read_csv(file)
                print(f"✅ {file}: {len(df):,} rows, {len(df.columns)} columns")
            except Exception as e:
                print(f"❌ {file}: Error - {str(e)}")
    else:
        print("❌ No sales files found")
    
    return ubereats_data

def test_data_analysis():
    """Test basic data analysis functions"""
    print("\n📊 Testing Data Analysis Functions...")
    print("=" * 50)
    
    # Test with UberEats data
    if os.path.exists('all_ads.csv'):
        df = pd.read_csv('all_ads.csv')
        print(f"\n📈 Sample Analysis for all_ads.csv:")
        print(f"   - Total rows: {len(df):,}")
        print(f"   - Total columns: {len(df.columns)}")
        print(f"   - Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        print(f"   - Missing values: {df.isnull().sum().sum()}")
        print(f"   - Data completeness: {(df.count().sum() / (len(df) * len(df.columns))) * 100:.1f}%")
        
        # Show column info
        print(f"\n   - Column types:")
        for col, dtype in df.dtypes.items():
            print(f"     {col}: {dtype}")
    
    print("\n✅ Data analysis test completed!")

def test_requirements():
    """Test if all required packages are installed"""
    print("\n📦 Testing Required Packages...")
    print("=" * 50)
    
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 
        'matplotlib', 'seaborn', 'openpyxl', 'xlrd'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}: Installed")
        except ImportError:
            print(f"❌ {package}: Not installed")
    
    print("\n✅ Package test completed!")

def main():
    """Main test function"""
    print("🍔 QSR Executive Enterprises Dashboard - Test Suite")
    print("=" * 60)
    
    # Test requirements
    test_requirements()
    
    # Test data loading
    test_data_loading()
    
    # Test data analysis
    test_data_analysis()
    
    print("\n🎉 All tests completed!")
    print("\nTo run the dashboard:")
    print("   streamlit run app.py")

if __name__ == "__main__":
    main()
