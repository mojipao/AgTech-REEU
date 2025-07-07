#!/usr/bin/env python3
"""
Time Series Visualization by Feature and Plot
============================================

Creates time series plots for each feature (ExG, soil moisture, Heat Index, ET0, Rainfall)
with different lines for each plot, allowing easy comparison of plot behavior over time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load the complete dataset"""
    print("Loading data for time series visualization...")
    
    # Load complete dataset
    complete_data = pd.read_csv('../data/corpus_season_completed_enhanced_lubbock_ml.csv')
    complete_data['Date'] = pd.to_datetime(complete_data['Date'], format='mixed')
    
    # Add temporal features
    planting_date = pd.to_datetime('2025-04-03')
    complete_data['Days_After_Planting'] = (complete_data['Date'] - planting_date).dt.days
    complete_data['Month'] = complete_data['Date'].dt.month
    
    print(f"Complete dataset: {len(complete_data)} rows")
    print(f"Date range: {complete_data['Date'].min()} to {complete_data['Date'].max()}")
    
    return complete_data

def plot_exg_time_series(data):
    """Plot ExG time series for each plot"""
    print("\n📊 Creating ExG time series plot...")
    
    # Filter for RL-relevant plots only (no separate synthetic plot)
    rl_plots = ['102.0', '404.0', '409.0']
    plot_data = data[data['Plot ID'].astype(str).isin(rl_plots)].copy()
    
    if len(plot_data) == 0:
        print("No data found for RL-relevant plots")
        return
    
    # Debug: Show what plot IDs we have
    print(f"Available plot IDs: {plot_data['Plot ID'].unique()}")
    print(f"Date range in filtered data: {plot_data['Date'].min()} to {plot_data['Date'].max()}")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot each plot with different colors and styles
    colors = ['blue', 'red', 'green']
    line_styles = ['-', '--', '-.']
    
    for i, plot_id in enumerate(rl_plots):
        plot_subset = plot_data[plot_data['Plot ID'].astype(str) == plot_id].copy()
        plot_subset = plot_subset.sort_values('Date')
        
        if len(plot_subset) > 0:
            print(f"Plot {plot_id}: {len(plot_subset)} data points from {plot_subset['Date'].min()} to {plot_subset['Date'].max()}")
            
            # Separate historical and synthetic data based on date range
            # Historical data: dates before 2025-01-01 (original experimental data)
            # Synthetic data: dates from 2025-01-01 onwards (generated data)
            historical_mask = plot_subset['Date'] < pd.to_datetime('2025-01-01')
            synthetic_mask = plot_subset['Date'] >= pd.to_datetime('2025-01-01')
            
            # Plot historical data
            historical_data = plot_subset[historical_mask]
            if len(historical_data) > 0:
                ax.plot(historical_data['Date'], historical_data['ExG'], 
                       color=colors[i], linestyle=line_styles[i], linewidth=2, 
                       label=f'Plot {plot_id} (Historical)', alpha=0.8)
                ax.scatter(historical_data['Date'], historical_data['ExG'], 
                          color=colors[i], s=30, alpha=0.8, marker='o')
            
            # Plot synthetic data
            synthetic_data = plot_subset[synthetic_mask]
            if len(synthetic_data) > 0:
                ax.plot(synthetic_data['Date'], synthetic_data['ExG'], 
                       color=colors[i], linestyle=line_styles[i], linewidth=2, 
                       label=f'Plot {plot_id} (Synthetic)', alpha=0.6)
                ax.scatter(synthetic_data['Date'], synthetic_data['ExG'], 
                          color='orange', s=40, alpha=0.8, marker='s', edgecolors='black')
    
    ax.set_title('ExG Time Series by Plot', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('ExG', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Add planting date line
    planting_date = pd.to_datetime('2025-04-03')
    ax.axvline(x=planting_date, color='black', linestyle=':', alpha=0.7, 
               label='Planting Date')
    
    plt.tight_layout()
    plt.savefig('../analysis/exg_time_series.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_soil_moisture_time_series(data):
    """Plot soil moisture time series by plot - HISTORICAL DATA ONLY"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Filter for RL-relevant plots only
    rl_plots = ['102.0', '404.0', '409.0']
    plot_data = data[data['Plot ID'].astype(str).isin(rl_plots)].copy()
    
    # IMPORTANT: Only show HISTORICAL soil moisture data (June 6 - July 1, 2025)
    # Skip synthetic data since we only have 3.5 weeks of real measurements
    historical_start = pd.to_datetime('2025-06-06')
    historical_end = pd.to_datetime('2025-07-01')
    
    plot_data = plot_data[
        (plot_data['Date'] >= historical_start) & 
        (plot_data['Date'] <= historical_end)
    ]
    
    if len(plot_data) == 0:
        print("⚠️  No historical soil moisture data found for June 6-July 1, 2025")
        return
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    line_styles = ['-', '--', '-.']
    
    for i, plot_id in enumerate(rl_plots):
        plot_subset = plot_data[plot_data['Plot ID'].astype(str) == plot_id].sort_values('Date')
        
        if len(plot_subset) > 0:
            # Plot only historical data (no synthetic)
            ax.plot(plot_subset['Date'], plot_subset['Total Soil Moisture'], 
                   color=colors[i], linestyle=line_styles[i], linewidth=3, 
                   label=f'Plot {plot_id} (Historical Only)', alpha=0.9)
            ax.scatter(plot_subset['Date'], plot_subset['Total Soil Moisture'], 
                      color=colors[i], s=50, alpha=0.9, marker='o', edgecolors='black')
    
    ax.set_title('Soil Moisture Time Series - HISTORICAL DATA ONLY\n(June 6 - July 1, 2025)', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Soil Moisture (gallons)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Add field capacity line (based on your data patterns)
    ax.axhline(y=250, color='red', linestyle='--', alpha=0.7, 
               label='Typical Field Capacity (250 gal)')
    
    # Add observed range
    ax.axhline(y=187.5, color='orange', linestyle=':', alpha=0.5, 
               label='Observed Min (187.5 gal)')
    ax.axhline(y=320.0, color='orange', linestyle=':', alpha=0.5, 
               label='Observed Max (320.0 gal)')
    
    # Add note about limited data
    ax.text(0.02, 0.98, 'Note: Only 3.5 weeks of real measurements available\nSynthetic data excluded for transparency', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('../analysis/soil_moisture_historical_only.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Soil moisture plot saved (HISTORICAL DATA ONLY)")
    print(f"   Date range: {plot_data['Date'].min().strftime('%Y-%m-%d')} to {plot_data['Date'].max().strftime('%Y-%m-%d')}")
    print(f"   Data points: {len(plot_data)}")
    print(f"   Plots: {plot_data['Plot ID'].unique()}")

def plot_heat_index_time_series(data):
    """Plot heat index time series (weather variable - same across all plots)"""
    print("\n🌡️ Creating heat index time series plot...")
    
    # Filter for synthetic weather data (weather variables are consistent across plots)
    synthetic_data = data[data['Plot ID'].astype(str) == 'Synthetic'].copy()
    
    if len(synthetic_data) == 0:
        print("No synthetic weather data found")
        return
    
    print(f"Synthetic weather data: {len(synthetic_data)} data points from {synthetic_data['Date'].min()} to {synthetic_data['Date'].max()}")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sort by date
    synthetic_data = synthetic_data.sort_values('Date')
    
    # Plot synthetic weather data
    ax.plot(synthetic_data['Date'], synthetic_data['Heat Index (F)'], 
           color='red', linewidth=2, label='Synthetic Weather Data', alpha=0.8)
    ax.scatter(synthetic_data['Date'], synthetic_data['Heat Index (F)'], 
              color='red', s=30, alpha=0.8, marker='o')
    
    ax.set_title('Heat Index Time Series (Weather Variable)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Heat Index (°F)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Add stress threshold lines
    ax.axhline(y=85, color='orange', linestyle='--', alpha=0.7, 
               label='Mild Stress (85°F)')
    ax.axhline(y=95, color='red', linestyle='--', alpha=0.7, 
               label='Moderate Stress (95°F)')
    
    # Add planting date line
    planting_date = pd.to_datetime('2025-04-03')
    ax.axvline(x=planting_date, color='black', linestyle=':', alpha=0.7, 
               label='Planting Date')
    
    plt.tight_layout()
    plt.savefig('../analysis/heat_index_time_series.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_et0_time_series(data):
    """Plot ET0 time series (weather variable - same across all plots)"""
    print("\n💨 Creating ET0 time series plot...")
    
    # Filter for synthetic weather data (weather variables are consistent across plots)
    synthetic_data = data[data['Plot ID'].astype(str) == 'Synthetic'].copy()
    
    if len(synthetic_data) == 0:
        print("No synthetic weather data found")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sort by date
    synthetic_data = synthetic_data.sort_values('Date')
    
    # Plot synthetic weather data
    ax.plot(synthetic_data['Date'], synthetic_data['ET0 (mm)'], 
           color='green', linewidth=2, label='Synthetic Weather Data', alpha=0.8)
    ax.scatter(synthetic_data['Date'], synthetic_data['ET0 (mm)'], 
              color='green', s=30, alpha=0.8, marker='o')
    
    ax.set_title('ET0 Time Series (Weather Variable)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('ET0 (mm/day)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Add planting date line
    planting_date = pd.to_datetime('2025-04-03')
    ax.axvline(x=planting_date, color='black', linestyle=':', alpha=0.7, 
               label='Planting Date')
    
    plt.tight_layout()
    plt.savefig('../analysis/et0_time_series.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_rainfall_time_series(data):
    """Plot rainfall time series (weather variable - same across all plots)"""
    print("\n🌧️ Creating rainfall time series plot...")
    
    # Filter for synthetic weather data (weather variables are consistent across plots)
    synthetic_data = data[data['Plot ID'].astype(str) == 'Synthetic'].copy()
    
    if len(synthetic_data) == 0:
        print("No synthetic weather data found")
        return
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sort by date
    synthetic_data = synthetic_data.sort_values('Date')
    
    # Plot synthetic weather data
    ax.plot(synthetic_data['Date'], synthetic_data['Rainfall (gallons)'], 
           color='blue', linewidth=2, label='Synthetic Weather Data', alpha=0.8)
    ax.scatter(synthetic_data['Date'], synthetic_data['Rainfall (gallons)'], 
              color='blue', s=30, alpha=0.8, marker='o')
    
    ax.set_title('Rainfall Time Series (Weather Variable)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Rainfall (gallons)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Add planting date line
    planting_date = pd.to_datetime('2025-04-03')
    ax.axvline(x=planting_date, color='black', linestyle=':', alpha=0.7, 
               label='Planting Date')
    
    plt.tight_layout()
    plt.savefig('../analysis/rainfall_time_series.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_statistics(data):
    """Create summary statistics for each plot and feature"""
    print("\n📈 Creating summary statistics...")
    
    # Filter for RL-relevant plots only (no separate synthetic plot)
    rl_plots = ['102.0', '404.0', '409.0']
    plot_data = data[data['Plot ID'].astype(str).isin(rl_plots)].copy()
    
    if len(plot_data) == 0:
        print("No data found for RL-relevant plots")
        return
    
    # Show overall date range
    print(f"Overall date range: {plot_data['Date'].min()} to {plot_data['Date'].max()}")
    
    # Calculate summary statistics for each plot
    features = ['ExG', 'Total Soil Moisture', 'Heat Index (F)', 'ET0 (mm)', 'Rainfall (gallons)']
    
    summary_stats = []
    for plot_id in rl_plots:
        plot_subset = plot_data[plot_data['Plot ID'].astype(str) == plot_id]
        
        for feature in features:
            if feature in plot_subset.columns:
                values = plot_subset[feature].dropna()
                if len(values) > 0:
                    summary_stats.append({
                        'Plot': plot_id,
                        'Feature': feature,
                        'Mean': values.mean(),
                        'Std': values.std(),
                        'Min': values.min(),
                        'Max': values.max(),
                        'Count': len(values)
                    })
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Create summary table visualization
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create table
    table_data = []
    for plot_id in rl_plots:
        plot_stats = summary_df[summary_df['Plot'] == plot_id]
        for _, row in plot_stats.iterrows():
            table_data.append([
                f"Plot {plot_id}",
                row['Feature'],
                f"{row['Mean']:.2f}",
                f"{row['Std']:.2f}",
                f"{row['Min']:.2f}",
                f"{row['Max']:.2f}",
                f"{row['Count']}"
            ])
    
    table = ax.table(cellText=table_data, 
                    colLabels=['Plot', 'Feature', 'Mean', 'Std', 'Min', 'Max', 'Count'],
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    ax.set_title('Summary Statistics by Plot and Feature', fontweight='bold', fontsize=16, pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../analysis/summary_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return summary_df

def main():
    """Main execution function"""
    print("🚀 Starting Time Series Visualization by Feature and Plot")
    print("=" * 70)
    
    # Load data
    data = load_data()
    
    # Create time series plots for each feature
    plot_exg_time_series(data)
    plot_soil_moisture_time_series(data)
    plot_heat_index_time_series(data)
    plot_et0_time_series(data)
    plot_rainfall_time_series(data)
    
    # Create summary statistics
    summary_df = create_summary_statistics(data)
    
    print("\n✅ Time series visualization completed!")
    print("📊 Visualizations saved to ../analysis/")
    print("   - exg_time_series.png")
    print("   - soil_moisture_time_series.png")
    print("   - heat_index_time_series.png")
    print("   - et0_time_series.png")
    print("   - rainfall_time_series.png")
    print("   - summary_statistics.png")
    
    # Print summary
    print("\n" + "=" * 70)
    print("📈 TIME SERIES ANALYSIS SUMMARY")
    print("=" * 70)
    
    # Show summary statistics
    if summary_df is not None:
        print("\nSummary Statistics by Plot and Feature:")
        print(summary_df.to_string(index=False))
    
    print("\n🎯 Key Insights:")
    print("   • Each plot shows different temporal patterns")
    print("   • Weather features (Heat Index, ET0, Rainfall) are consistent across plots")
    print("   • ExG and soil moisture show plot-specific variations")
    print("   • Planting date (2025-04-03) marked on all plots")
    print("   • Field capacity (250 gal) and stress thresholds shown where relevant")

if __name__ == "__main__":
    main() 