#!/usr/bin/env python3
"""
Hybrid ML + Physics Visualization
=================================

Simple visualization of the hybrid approach showing:
1. ML model accuracy (Heat Index, ET0, Rainfall)
2. Physics-based calculations (ExG, Soil Moisture)
3. Combined hybrid approach results
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
    """Load all datasets"""
    print("Loading data for hybrid visualization...")
    
    # Load historical data
    historical_data = pd.read_csv('../data/Model Input - Corpus.csv')
    historical_data['Date'] = pd.to_datetime(historical_data['Date'])
    
    # Load complete dataset
    complete_data = pd.read_csv('../data/corpus_season_completed_enhanced_lubbock_ml.csv')
    complete_data['Date'] = pd.to_datetime(complete_data['Date'], format='mixed')
    
    # Separate synthetic from historical
    historical_dates = set(historical_data['Date'].dt.date)
    synthetic_data = complete_data[
        ~complete_data['Date'].dt.date.isin(historical_dates)
    ].copy()
    
    # Add temporal features
    planting_date = pd.to_datetime('2025-04-03')
    for df in [historical_data, synthetic_data, complete_data]:
        df['Days_After_Planting'] = (df['Date'] - planting_date).dt.days
        df['Month'] = df['Date'].dt.month
    
    print(f"Historical data: {len(historical_data)} rows")
    print(f"Synthetic data: {len(synthetic_data)} rows")
    
    return historical_data, synthetic_data, complete_data

def create_ml_performance_summary():
    """Create ML performance summary visualization"""
    print("\n📊 Creating ML performance summary...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Hybrid ML + Physics Approach - ML Model Performance', fontsize=16, fontweight='bold')
    
    # ML R² scores
    ml_vars = ['Heat Index', 'ET0', 'Rainfall']
    ml_r2 = [0.959, 0.890, 0.057]
    ml_cv_r2 = [0.959, 0.865, -0.024]
    ml_status = ['Excellent', 'Excellent', 'Poor']
    
    # ML RMSE scores (from training output)
    ml_rmse = [3.045, 1.114, 780.007]
    ml_cv_rmse = [3.044, 1.165, 907.474]
    
    x = np.arange(len(ml_vars))
    width = 0.35
    
    # R² plot
    bars1_r2 = ax1.bar(x - width/2, ml_cv_r2, width, label='CV R²', alpha=0.7, color='blue')
    bars2_r2 = ax1.bar(x + width/2, ml_r2, width, label='Final R²', alpha=0.7, color='orange')
    
    ax1.set_title('R² Performance')
    ax1.set_ylabel('R² Score')
    ax1.set_xticks(x)
    ax1.set_xticklabels(ml_vars)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.2, 1.1)
    
    # Add R² value labels
    for i, (bar1, bar2, cv, final, status) in enumerate(zip(bars1_r2, bars2_r2, ml_cv_r2, ml_r2, ml_status)):
        ax1.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.02,
                f'{cv:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax1.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.02,
                f'{final:.3f}\n({status})', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # RMSE plot
    bars1_rmse = ax2.bar(x - width/2, ml_cv_rmse, width, label='CV RMSE', alpha=0.7, color='blue')
    bars2_rmse = ax2.bar(x + width/2, ml_rmse, width, label='Final RMSE', alpha=0.7, color='orange')
    
    ax2.set_title('RMSE Performance')
    ax2.set_ylabel('RMSE')
    ax2.set_xticks(x)
    ax2.set_xticklabels(ml_vars)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add RMSE value labels
    for i, (bar1, bar2, cv, final) in enumerate(zip(bars1_rmse, bars2_rmse, ml_cv_rmse, ml_rmse)):
        ax2.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height() + 0.02,
                f'{cv:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax2.text(bar2.get_x() + bar2.get_width()/2, bar2.get_height() + 0.02,
                f'{final:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../analysis/ml_performance_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_hybrid_workflow():
    """Create hybrid workflow diagram"""
    print("\n🔄 Creating hybrid workflow diagram...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create workflow diagram
    steps = [
        ('ML Models\n(Heat Index, ET0, Rainfall)', 0.2, 0.8),
        ('Physics Models\n(ExG, Soil Moisture)', 0.8, 0.8),
        ('ML Predictions\nas Inputs', 0.2, 0.5),
        ('Physics\nCalculations', 0.8, 0.5),
        ('Hybrid\nSynthetic Data', 0.5, 0.2)
    ]
    
    for step, x, y in steps:
        ax.text(x, y, step, ha='center', va='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7),
               fontweight='bold', fontsize=12)
    
    # Add arrows
    ax.annotate('', xy=(0.3, 0.7), xytext=(0.2, 0.7),
               arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(0.7, 0.7), xytext=(0.8, 0.7),
               arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(0.4, 0.4), xytext=(0.3, 0.4),
               arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(0.6, 0.4), xytext=(0.7, 0.4),
               arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(0.5, 0.3), xytext=(0.4, 0.3),
               arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=(0.5, 0.3), xytext=(0.6, 0.3),
               arrowprops=dict(arrowstyle='->', lw=2))
    
    ax.set_title('Hybrid ML + Physics Workflow', fontweight='bold', fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../analysis/hybrid_workflow_diagram.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_synthetic_data_analysis(historical_data, synthetic_data):
    """Create synthetic data analysis visualization"""
    print("\n📈 Creating synthetic data analysis...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Hybrid Approach - Synthetic Data Analysis', fontsize=16, fontweight='bold')
    
    # 1. Heat Index ML Performance
    if len(synthetic_data) > 0:
        ax1.plot(synthetic_data['Date'], synthetic_data['Heat Index (F)'], 
               'r-', linewidth=2, alpha=0.8, label='ML Predicted')
        ax1.set_title('Heat Index ML Performance\n(R² = 0.959)', fontweight='bold')
        ax1.set_ylabel('Heat Index (°F)')
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. ET0 ML Performance
    if len(synthetic_data) > 0:
        ax2.plot(synthetic_data['Date'], synthetic_data['ET0 (mm)'], 
               'g-', linewidth=2, alpha=0.8, label='ML Predicted')
        ax2.set_title('ET0 ML Performance\n(R² = 0.890)', fontweight='bold')
        ax2.set_ylabel('ET0 (mm/day)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. ExG Physics Performance
    if len(synthetic_data) > 0:
        ax3.scatter(synthetic_data['Days_After_Planting'], synthetic_data['ExG'], 
                  alpha=0.6, color='purple', s=20, label='Physics-Based')
        
        # Add trend line
        z = np.polyfit(synthetic_data['Days_After_Planting'], synthetic_data['ExG'], 2)
        p = np.poly1d(z)
        ax3.plot(synthetic_data['Days_After_Planting'], 
               p(synthetic_data['Days_After_Planting']), 
               "r--", alpha=0.8, linewidth=2, label='Trend')
        
        ax3.set_title('ExG Physics Performance\n(No Overfitting)', fontweight='bold')
        ax3.set_xlabel('Days After Planting')
        ax3.set_ylabel('ExG')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Soil Moisture Physics Performance
    if len(synthetic_data) > 0:
        ax4.plot(synthetic_data['Date'], synthetic_data['Total Soil Moisture'], 
               'brown', linewidth=2, alpha=0.8, label='Physics-Based')
        ax4.set_title('Soil Moisture Physics Performance\n(Water Balance)', fontweight='bold')
        ax4.set_ylabel('Soil Moisture (gallons)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../analysis/synthetic_data_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_comprehensive_summary():
    """Create comprehensive summary visualization"""
    print("\n📋 Creating comprehensive summary...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create summary table
    summary_data = [
        ['Variable', 'Method', 'R² Score', 'Status', 'Benefits'],
        ['Heat Index', 'ML Model', '0.959', 'Excellent', 'High accuracy, 27-year data'],
        ['ET0', 'ML Model', '0.890', 'Excellent', 'Strong weather patterns'],
        ['Rainfall', 'ML Model', '0.057', 'Poor', 'Inherently difficult'],
        ['ExG', 'Physics', 'N/A', 'Robust', 'No overfitting, interpretable'],
        ['Soil Moisture', 'Physics', 'N/A', 'Robust', 'Water balance, realistic']
    ]
    
    # Create table
    table = ax.table(cellText=summary_data[1:], colLabels=summary_data[0],
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Color code cells
    for i in range(1, len(summary_data)):
        if summary_data[i][3] == 'Excellent':
            for j in range(len(summary_data[i])):
                table[(i, j)].set_facecolor('lightgreen')
        elif summary_data[i][3] == 'Poor':
            for j in range(len(summary_data[i])):
                table[(i, j)].set_facecolor('lightcoral')
        else:  # Robust
            for j in range(len(summary_data[i])):
                table[(i, j)].set_facecolor('lightblue')
    
    ax.set_title('Hybrid ML + Physics Approach - Comprehensive Summary', 
                fontweight='bold', fontsize=16, pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('../analysis/comprehensive_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main execution function"""
    print("🚀 Starting Hybrid ML + Physics Visualization")
    print("=" * 60)
    
    # Load data
    historical_data, synthetic_data, complete_data = load_data()
    
    # Create visualizations
    create_ml_performance_summary()
    create_hybrid_workflow()
    create_synthetic_data_analysis(historical_data, synthetic_data)
    create_comprehensive_summary()
    
    print("\n✅ Hybrid visualization completed!")
    print("📊 Visualizations saved to ../analysis/")
    print("   - ml_performance_summary.png")
    print("   - hybrid_workflow_diagram.png")
    print("   - synthetic_data_analysis.png")
    print("   - comprehensive_summary.png")
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 HYBRID ML + PHYSICS APPROACH SUMMARY")
    print("=" * 60)
    print("🔬 ML Models (Reliable Variables):")
    print("   • Heat Index: R² = 0.959 (Excellent)")
    print("   • ET0: R² = 0.890 (Excellent)")
    print("   • Rainfall: R² = 0.057 (Poor, but expected)")
    print()
    print("⚛️  Physics Models (Complex Variables):")
    print("   • ExG: Stress-based growth model (No overfitting)")
    print("   • Soil Moisture: Water balance equation (Realistic)")
    print()
    print("🎯 Hybrid Benefits:")
    print("   • Combines ML accuracy with physics realism")
    print("   • Eliminates overfitting in complex variables")
    print("   • Maintains interpretability and scientific soundness")
    print("   • Provides robust synthetic data for RL training")

if __name__ == "__main__":
    main() 