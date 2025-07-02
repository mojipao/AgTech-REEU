#!/usr/bin/env python3
"""
Comprehensive Accuracy Analysis for Cotton Irrigation Synthetic Data Generator
==============================================================================

This unified script provides comprehensive evaluation of the ML models including:

Standard ML Analysis:
1. R² and RMSE analysis with cross-validation
2. SHAP feature importance analysis  
3. Distribution comparisons (synthetic vs historical)
4. Treatment-specific accuracy metrics
5. Seasonal pattern validation
6. Residual analysis

Agricultural-Specific Analysis:
7. Cotton growth stage accuracy validation
8. Agricultural threshold accuracy (irrigation decisions, heat stress, vigor)
9. Temporal consistency analysis
10. Cotton physiology validation (ExG ranges by growth stage)
11. Treatment efficacy differentiation
12. Enhanced SHAP visualizations for cotton-specific features

Author: Mohriz Murad
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP (optional)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

# Import our generator
from corpus_ml import EnhancedMLCottonSyntheticGenerator

class AccuracyAnalyzer:
    """Comprehensive accuracy analysis for synthetic data generation"""
    
    def __init__(self):
        self.generator = None
        self.historical_data = None
        self.synthetic_data = None
        self.complete_data = None
        
    def load_and_prepare_data(self):
        """Load all datasets for analysis"""
        print("Loading data for analysis...")
        
        # Load historical data
        self.historical_data = pd.read_csv('../data/Model Input - Corpus.csv')
        self.historical_data['Date'] = pd.to_datetime(self.historical_data['Date'])
        
        # Load complete synthetic dataset 
        self.complete_data = pd.read_csv('../data/corpus_season_completed_enhanced_lubbock_ml.csv')
        self.complete_data['Date'] = pd.to_datetime(self.complete_data['Date'])
        
        # Separate synthetic from historical
        historical_dates = set(self.historical_data['Date'].dt.date)
        self.synthetic_data = self.complete_data[
            ~self.complete_data['Date'].dt.date.isin(historical_dates)
        ].copy()
        
        # Add cotton growth stages and temporal features
        planting_date = pd.to_datetime('2025-04-03')  # Corrected planting date
        for df in [self.historical_data, self.synthetic_data, self.complete_data]:
            df['Days_After_Planting'] = (df['Date'] - planting_date).dt.days
            df['Growth_Stage'] = df['Days_After_Planting'].apply(self._get_growth_stage)
            df['Week'] = df['Date'].dt.isocalendar().week
            df['Month'] = df['Date'].dt.month
        
        print(f"Historical data: {len(self.historical_data)} rows")
        print(f"Synthetic data: {len(self.synthetic_data)} rows") 
        print(f"Complete dataset: {len(self.complete_data)} rows")
        
        return self.historical_data, self.synthetic_data, self.complete_data
    
    def _get_growth_stage(self, days):
        """Map days after planting to cotton growth stages"""
        if days < 10:
            return 'Seeding'
        elif days < 40:
            return 'Squaring'
        elif days < 60:
            return 'Early Bloom'
        elif days < 90:
            return 'Peak Bloom'
        elif days < 125:
            return 'Boll Development'
        elif days < 150:
            return 'Maturity'
        else:
            return 'Harvest'
    
    def train_models_for_analysis(self):
        """Train models and prepare for detailed analysis"""
        print("\nTraining models for analysis...")
        
        self.generator = EnhancedMLCottonSyntheticGenerator()
        self.generator.load_data()
        
        # Get training data
        (X_exg, y_exg, weights_exg, X_soil, y_soil, weights_soil,
         X_heat, y_heat, weights_heat, X_et0, y_et0, weights_et0,
         X_rainfall, y_rainfall, weights_rainfall) = self.generator.prepare_enhanced_training_data()
        
        # Train models
        self.generator.train_enhanced_models(
            X_exg, y_exg, weights_exg, X_soil, y_soil, weights_soil,
            X_heat, y_heat, weights_heat, X_et0, y_et0, weights_et0,
            X_rainfall, y_rainfall, weights_rainfall
        )
        
        # Store training data for analysis
        self.training_data = {
            'exg': (X_exg, y_exg, weights_exg),
            'soil': (X_soil, y_soil, weights_soil),
            'heat': (X_heat, y_heat, weights_heat),
            'et0': (X_et0, y_et0, weights_et0),
            'rainfall': (X_rainfall, y_rainfall, weights_rainfall)
        }
        
        print("Models trained successfully!")
        return self.generator
    
    def comprehensive_model_evaluation(self):
        """Comprehensive evaluation of all models"""
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*80)
        
        models = {
            'ExG': (self.generator.exg_model, self.generator.scaler_exg),
            'Soil Moisture': (self.generator.soil_model, self.generator.scaler_soil),
            'Heat Index': (self.generator.heat_index_model, self.generator.scaler_heat),
            'ET0': (self.generator.et0_model, self.generator.scaler_et0),
            'Rainfall': (self.generator.rainfall_model, self.generator.scaler_rainfall)
        }
        
        results = {}
        
        for var_name, (model, scaler) in models.items():
            if model is None:
                continue
                
            print(f"\n--- {var_name} Model Analysis ---")
            
            # Get training data
            if var_name == 'ExG':
                X, y, weights = self.training_data['exg']
            elif var_name == 'Soil Moisture':
                X, y, weights = self.training_data['soil']
            elif var_name == 'Heat Index':
                X, y, weights = self.training_data['heat']
            elif var_name == 'ET0':
                X, y, weights = self.training_data['et0']
            elif var_name == 'Rainfall':
                X, y, weights = self.training_data['rainfall']
            
            X_scaled = scaler.transform(X)
            
            # Basic metrics
            y_pred = model.predict(X_scaled)
            r2 = r2_score(y, y_pred, sample_weight=weights)
            rmse = np.sqrt(mean_squared_error(y, y_pred, sample_weight=weights))
            mae = mean_absolute_error(y, y_pred, sample_weight=weights)
            
            print(f"R²: {r2:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
            
            # Cross-validation (without weights for compatibility)
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
            print(f"CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            
            # Store results
            results[var_name] = {
                'R2': r2,
                'RMSE': rmse,
                'MAE': mae,
                'CV_R2_mean': cv_scores.mean(),
                'CV_R2_std': cv_scores.std(),
                'y_true': y,
                'y_pred': y_pred,
                'weights': weights
            }
        
        return results
    
    def shap_analysis(self):
        """SHAP analysis for feature importance"""
        if not SHAP_AVAILABLE:
            print("\nSHAP analysis skipped - SHAP not installed")
            return {}
            
        print("\n" + "="*80)
        print("SHAP FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        # Feature names for different models
        feature_names = {
            'exg': ['Days After Planting', 'Heat Index', 'ET0', 'Soil Moisture', 'Rainfall',
                   'Rainfed', 'Half Irrigation', 'Full Irrigation', 'Kc Coefficient', 
                   'Month', 'Day of Year', 'Is Corpus', 'Sin Seasonal', 'Cos Seasonal'],
            'soil': ['Days After Planting', 'Heat Index', 'ET0', 'Rainfall', 'Irrigation',
                    'Rainfed', 'Half Irrigation', 'Full Irrigation', 'Kc Coefficient',
                    'Month', 'Day of Year', 'Is Corpus', 'Sin Seasonal', 'Cos Seasonal'],
            'environmental': ['Days After Planting', 'Month', 'Day of Year', 'Weekday',
                            'Sin Annual', 'Cos Annual', 'Sin Semi-Annual', 'Cos Semi-Annual',
                            'Kc Coefficient', 'Is Corpus', 'Distance from Summer', 'Is Summer']
        }
        
        models_to_analyze = [
            ('ExG', self.generator.exg_model, self.generator.scaler_exg, 'exg'),
            ('Soil Moisture', self.generator.soil_model, self.generator.scaler_soil, 'soil'),
            ('Heat Index', self.generator.heat_index_model, self.generator.scaler_heat, 'environmental')
        ]
        
        shap_results = {}
        
        for model_name, model, scaler, data_key in models_to_analyze:
            if model is None:
                continue
                
            print(f"\n--- SHAP Analysis: {model_name} ---")
            
            # Get appropriate data
            if data_key == 'exg':
                X, y, weights = self.training_data['exg']
            elif data_key == 'soil':
                X, y, weights = self.training_data['soil']
            else:  # environmental
                X, y, weights = self.training_data['heat']
            
            X_scaled = scaler.transform(X)
            
            # Sample data for SHAP (for performance)
            sample_size = min(100, len(X_scaled))
            sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
            X_sample = X_scaled[sample_indices]
            
            try:
                # Create SHAP explainer
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_sample)
                
                # Feature importance
                feature_importance = np.abs(shap_values).mean(0)
                features = feature_names.get(data_key, [f'Feature_{i}' for i in range(len(feature_importance))])
                
                # Display top features
                importance_df = pd.DataFrame({
                    'Feature': features[:len(feature_importance)],
                    'Importance': feature_importance
                }).sort_values('Importance', ascending=False)
                
                print("Top 5 Most Important Features:")
                print(importance_df.head().to_string(index=False))
                
                shap_results[model_name] = {
                    'explainer': explainer,
                    'shap_values': shap_values,
                    'feature_names': features[:len(feature_importance)],
                    'importance_df': importance_df,
                    'X_sample': X_sample
                }
                
            except Exception as e:
                print(f"SHAP analysis failed for {model_name}: {e}")
                continue
        
        return shap_results
    
    def treatment_specific_accuracy(self):
        """Analyze accuracy by treatment type"""
        print("\n" + "="*80)
        print("TREATMENT-SPECIFIC ACCURACY ANALYSIS")
        print("="*80)
        
        treatments = ['R_F', 'H_I', 'F_I']
        variables = ['ExG', 'Total Soil Moisture', 'Heat Index (F)', 'ET0 (mm)', 'Rainfall (gallons)']
        
        for treatment in treatments:
            print(f"\n--- Treatment: {treatment} ---")
            
            # Get historical and synthetic data for this treatment
            hist_treatment = self.historical_data[self.historical_data['Treatment Type'] == treatment]
            synth_treatment = self.synthetic_data[self.synthetic_data['Treatment Type'] == treatment]
            
            print(f"Historical samples: {len(hist_treatment)}")
            print(f"Synthetic samples: {len(synth_treatment)}")
            
            for var in variables:
                if var in hist_treatment.columns and var in synth_treatment.columns:
                    hist_vals = hist_treatment[var].dropna()
                    synth_vals = synth_treatment[var].dropna()
                    
                    if len(hist_vals) > 0 and len(synth_vals) > 0:
                        # Basic statistics comparison
                        hist_mean, hist_std = hist_vals.mean(), hist_vals.std()
                        synth_mean, synth_std = synth_vals.mean(), synth_vals.std()
                        
                        print(f"\n  {var}:")
                        print(f"    Historical: μ={hist_mean:.3f}, σ={hist_std:.3f}")
                        print(f"    Synthetic:  μ={synth_mean:.3f}, σ={synth_std:.3f}")
                        print(f"    Mean Diff:  {abs(hist_mean - synth_mean):.3f}")
                        print(f"    Std Diff:   {abs(hist_std - synth_std):.3f}")
                        
                        # Statistical test for distribution similarity
                        if len(hist_vals) >= 3 and len(synth_vals) >= 3:
                            try:
                                ks_stat, ks_p = stats.ks_2samp(hist_vals, synth_vals)
                                print(f"    KS Test:    stat={ks_stat:.4f}, p={ks_p:.4f}")
                            except:
                                pass
    
    def seasonal_pattern_validation(self):
        """Validate seasonal patterns in synthetic data"""
        print("\n" + "="*80)
        print("SEASONAL PATTERN VALIDATION")
        print("="*80)
        
        # Add month to datasets
        self.historical_data['Month'] = self.historical_data['Date'].dt.month
        self.synthetic_data['Month'] = self.synthetic_data['Date'].dt.month
        
        variables = ['ExG', 'Total Soil Moisture', 'Heat Index (F)', 'ET0 (mm)']
        
        seasonal_results = {}
        
        for var in variables:
            if var in self.historical_data.columns and var in self.synthetic_data.columns:
                print(f"\n--- {var} Seasonal Patterns ---")
                
                # Monthly averages
                hist_monthly = self.historical_data.groupby('Month')[var].agg(['mean', 'std']).round(3)
                synth_monthly = self.synthetic_data.groupby('Month')[var].agg(['mean', 'std']).round(3)
                
                print("Historical Monthly Patterns:")
                print(hist_monthly.to_string())
                print("\nSynthetic Monthly Patterns:")
                print(synth_monthly.to_string())
                
                # Calculate correlation between seasonal patterns
                common_months = set(hist_monthly.index) & set(synth_monthly.index)
                if len(common_months) >= 2:
                    hist_means = [hist_monthly.loc[m, 'mean'] for m in sorted(common_months)]
                    synth_means = [synth_monthly.loc[m, 'mean'] for m in sorted(common_months)]
                    
                    seasonal_corr = np.corrcoef(hist_means, synth_means)[0, 1]
                    print(f"\nSeasonal Pattern Correlation: {seasonal_corr:.4f}")
                    
                    seasonal_results[var] = {
                        'hist_monthly': hist_monthly,
                        'synth_monthly': synth_monthly,
                        'seasonal_correlation': seasonal_corr
                    }
        
        return seasonal_results
    
    def generate_visualizations(self, model_results, shap_results, seasonal_results):
        """Generate comprehensive visualization plots"""
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        # Create multiple figures
        fig_size = (15, 10)
        
        # 1. Model Performance Summary
        plt.figure(figsize=fig_size)
        
        models = list(model_results.keys())
        r2_scores = [model_results[m]['R2'] for m in models]
        rmse_scores = [model_results[m]['RMSE'] for m in models]
        
        plt.subplot(2, 3, 1)
        plt.bar(models, r2_scores)
        plt.title('R² Scores by Model')
        plt.ylabel('R²')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        plt.subplot(2, 3, 2)
        plt.bar(models, rmse_scores)
        plt.title('RMSE by Model')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45)
        
        # 2. Residual plots for key variables
        for i, (model_name, results) in enumerate(list(model_results.items())[:4]):
            plt.subplot(2, 3, i+3)
            residuals = results['y_true'] - results['y_pred']
            plt.scatter(results['y_pred'], residuals, alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title(f'{model_name} Residuals')
            plt.xlabel('Predicted')
            plt.ylabel('Residuals')
        
        plt.tight_layout()
        plt.savefig('../analysis/model_performance_analysis.png', dpi=300, bbox_inches='tight')
        print("Model performance plot saved: ../analysis/model_performance_analysis.png")
        
        # 3. Distribution comparisons
        if len(self.historical_data) > 0 and len(self.synthetic_data) > 0:
            plt.figure(figsize=fig_size)
            
            variables = ['ExG', 'Total Soil Moisture', 'Heat Index (F)', 'ET0 (mm)']
            for i, var in enumerate(variables):
                if var in self.historical_data.columns and var in self.synthetic_data.columns:
                    plt.subplot(2, 2, i+1)
                    
                    hist_vals = self.historical_data[var].dropna()
                    synth_vals = self.synthetic_data[var].dropna()
                    
                    plt.hist(hist_vals, alpha=0.7, label='Historical', bins=20)
                    plt.hist(synth_vals, alpha=0.7, label='Synthetic', bins=20)
                    plt.title(f'{var} Distribution')
                    plt.legend()
                    plt.xlabel(var)
                    plt.ylabel('Frequency')
            
            plt.tight_layout()
            plt.savefig('../analysis/distribution_comparison.png', dpi=300, bbox_inches='tight')
            print("Distribution comparison plot saved: ../analysis/distribution_comparison.png")
        
        # 4. Seasonal patterns plot
        if seasonal_results:
            plt.figure(figsize=fig_size)
            
            for i, (var, results) in enumerate(list(seasonal_results.items())[:4]):
                plt.subplot(2, 2, i+1)
                
                months = sorted(results['hist_monthly'].index)
                hist_means = [results['hist_monthly'].loc[m, 'mean'] for m in months]
                synth_means = [results['synth_monthly'].loc[m, 'mean'] for m in months if m in results['synth_monthly'].index]
                synth_months = [m for m in months if m in results['synth_monthly'].index]
                
                plt.plot(months, hist_means, 'o-', label='Historical', linewidth=2)
                plt.plot(synth_months, synth_means, 's-', label='Synthetic', linewidth=2)
                plt.title(f'{var} Seasonal Pattern\n(r={results["seasonal_correlation"]:.3f})')
                plt.xlabel('Month')
                plt.ylabel(var)
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('../analysis/seasonal_patterns.png', dpi=300, bbox_inches='tight')
            print("Seasonal patterns plot saved: ../analysis/seasonal_patterns.png")
    
    def agricultural_accuracy_metrics(self):
        """Compute agriculture-specific accuracy metrics"""
        print("\n" + "="*80)
        print("AGRICULTURAL ACCURACY METRICS")
        print("="*80)
        
        treatments = ['R_F', 'H_I', 'F_I']
        variables = ['ExG', 'Total Soil Moisture', 'Heat Index (F)', 'ET0 (mm)']
        
        results = {}
        
        for treatment in treatments:
            print(f"\n--- Treatment: {treatment} (Cotton Irrigation Protocol) ---")
            
            hist_treat = self.historical_data[self.historical_data['Treatment Type'] == treatment]
            synth_treat = self.synthetic_data[self.synthetic_data['Treatment Type'] == treatment]
            
            treatment_results = {}
            
            for var in variables:
                if var in hist_treat.columns and var in synth_treat.columns:
                    hist_vals = hist_treat[var].dropna()
                    synth_vals = synth_treat[var].dropna()
                    
                    if len(hist_vals) > 0 and len(synth_vals) > 0:
                        # Agricultural relevance metrics
                        results_dict = self._compute_agricultural_metrics(
                            hist_vals, synth_vals, var, treatment
                        )
                        treatment_results[var] = results_dict
                        
                        print(f"\n  {var}:")
                        for metric, value in results_dict.items():
                            if isinstance(value, float):
                                print(f"    {metric}: {value:.4f}")
                            else:
                                print(f"    {metric}: {value}")
            
            results[treatment] = treatment_results
        
        return results
    
    def _compute_agricultural_metrics(self, hist_vals, synth_vals, variable, treatment):
        """Compute agriculture-specific metrics"""
        metrics = {}
        
        # Basic statistics
        metrics['Historical_Mean'] = hist_vals.mean()
        metrics['Synthetic_Mean'] = synth_vals.mean()
        metrics['Mean_Absolute_Difference'] = abs(hist_vals.mean() - synth_vals.mean())
        metrics['Relative_Error_Percent'] = (abs(hist_vals.mean() - synth_vals.mean()) / hist_vals.mean()) * 100
        
        # Agricultural thresholds
        if variable == 'ExG':
            # Vegetation vigor thresholds
            metrics['Historical_High_Vigor_Percent'] = (hist_vals > 0.6).mean() * 100
            metrics['Synthetic_High_Vigor_Percent'] = (synth_vals > 0.6).mean() * 100
            metrics['Vigor_Classification_Accuracy'] = self._classification_accuracy(
                hist_vals > 0.6, synth_vals > 0.6
            )
        
        elif variable == 'Total Soil Moisture':
            # Irrigation decision thresholds
            dry_threshold = 200 if treatment == 'F_I' else 180
            metrics['Historical_Dry_Percent'] = (hist_vals < dry_threshold).mean() * 100
            metrics['Synthetic_Dry_Percent'] = (synth_vals < dry_threshold).mean() * 100
            metrics['Irrigation_Decision_Accuracy'] = self._classification_accuracy(
                hist_vals < dry_threshold, synth_vals < dry_threshold
            )
        
        elif variable == 'Heat Index (F)':
            # Heat stress thresholds
            stress_threshold = 95
            metrics['Historical_Heat_Stress_Percent'] = (hist_vals > stress_threshold).mean() * 100
            metrics['Synthetic_Heat_Stress_Percent'] = (synth_vals > stress_threshold).mean() * 100
            metrics['Heat_Stress_Detection_Accuracy'] = self._classification_accuracy(
                hist_vals > stress_threshold, synth_vals > stress_threshold
            )
        
        # Distribution similarity
        if len(hist_vals) >= 3 and len(synth_vals) >= 3:
            ks_stat, ks_p = stats.ks_2samp(hist_vals, synth_vals)
            metrics['KS_Test_Statistic'] = ks_stat
            metrics['KS_Test_P_Value'] = ks_p
            metrics['Distribution_Similar'] = ks_p > 0.05
        
        return metrics
    
    def _classification_accuracy(self, true_binary, pred_binary):
        """Compute binary classification accuracy"""
        if len(true_binary) != len(pred_binary):
            return np.nan
        return (true_binary == pred_binary).mean()
    
    def cotton_growth_stage_analysis(self):
        """Analyze accuracy by cotton growth stages"""
        print("\n" + "="*80)
        print("COTTON GROWTH STAGE ACCURACY ANALYSIS")
        print("="*80)
        
        growth_stages = ['Seeding', 'Squaring', 'Early Bloom', 'Peak Bloom', 'Boll Development']
        variables = ['ExG', 'Total Soil Moisture']
        
        for stage in growth_stages:
            print(f"\n--- Growth Stage: {stage} ---")
            
            hist_stage = self.historical_data[self.historical_data['Growth_Stage'] == stage]
            synth_stage = self.synthetic_data[self.synthetic_data['Growth_Stage'] == stage]
            
            if len(hist_stage) == 0:
                print("  No historical data available for this stage")
                continue
                
            print(f"  Historical samples: {len(hist_stage)}")
            print(f"  Synthetic samples: {len(synth_stage)}")
            
            for var in variables:
                if var in hist_stage.columns and len(hist_stage[var].dropna()) > 0:
                    hist_vals = hist_stage[var].dropna()
                    
                    if len(synth_stage) > 0 and var in synth_stage.columns:
                        synth_vals = synth_stage[var].dropna()
                        
                        if len(synth_vals) > 0:
                            accuracy = mean_absolute_error([hist_vals.mean()], [synth_vals.mean()])
                            print(f"    {var} MAE: {accuracy:.4f}")
                            
                            # Cotton physiology validation
                            if var == 'ExG':
                                expected_range = self._get_expected_exg_range(stage)
                                synth_in_range = ((synth_vals >= expected_range[0]) & 
                                                (synth_vals <= expected_range[1])).mean()
                                print(f"    {var} physiologically valid: {synth_in_range*100:.1f}%")
    
    def _get_expected_exg_range(self, growth_stage):
        """Get expected ExG range for cotton growth stage"""
        ranges = {
            'Seeding': (0.0, 0.3),
            'Squaring': (0.1, 0.5),
            'Early Bloom': (0.3, 0.7),
            'Peak Bloom': (0.5, 1.0),
            'Boll Development': (0.4, 0.9),
            'Maturity': (0.3, 0.7),
            'Harvest': (0.2, 0.5)
        }
        return ranges.get(growth_stage, (0.0, 1.0))
    
    def temporal_accuracy_analysis(self):
        """Analyze accuracy over time periods"""
        print("\n" + "="*80)
        print("TEMPORAL ACCURACY ANALYSIS")
        print("="*80)
        
        # Weekly accuracy for synthetic data
        synthetic_weekly = self.synthetic_data.groupby(['Week', 'Treatment Type']).agg({
            'ExG': ['mean', 'std'],
            'Total Soil Moisture': ['mean', 'std'],
            'Heat Index (F)': ['mean', 'std']
        }).round(3)
        
        print("Weekly Synthetic Data Patterns (Sample):")
        print(synthetic_weekly.head(10).to_string())
        
        # Check for temporal consistency
        for treatment in ['R_F', 'H_I', 'F_I']:
            synth_treat = self.synthetic_data[self.synthetic_data['Treatment Type'] == treatment]
            
            if len(synth_treat) > 10:
                # ExG temporal trend
                exg_trend = synth_treat.groupby('Week')['ExG'].mean()
                
                # Check if trend is realistic (should generally increase then decrease)
                if len(exg_trend) >= 5:
                    peak_week = exg_trend.idxmax()
                    trend_realistic = (exg_trend.iloc[-1] < exg_trend.iloc[2:].max())  # Decreases from peak
                    
                    print(f"\nTreatment {treatment} ExG Temporal Pattern:")
                    print(f"  Peak ExG at week: {peak_week}")
                    print(f"  Realistic seasonal decline: {trend_realistic}")
    
    def create_enhanced_shap_visualizations(self):
        """Create detailed SHAP visualizations"""
        if not SHAP_AVAILABLE:
            print("\nSHAP visualizations skipped - SHAP not installed")
            return
            
        print("\n" + "="*80)
        print("CREATING ENHANCED SHAP VISUALIZATIONS")
        print("="*80)
        
        # Feature names
        exg_features = ['Days After Planting', 'Heat Index', 'ET0', 'Soil Moisture', 'Rainfall',
                       'Rainfed', 'Half Irrigation', 'Full Irrigation', 'Kc Coefficient', 
                       'Month', 'Day of Year', 'Is Corpus', 'Sin Seasonal', 'Cos Seasonal']
        
        # Create SHAP explainer for ExG model
        if self.generator.exg_model is not None:
            try:
                X, y, weights = self.training_data['exg']
                X_scaled = self.generator.scaler_exg.transform(X[:100])  # Sample for performance
                explainer = shap.TreeExplainer(self.generator.exg_model)
                shap_values = explainer.shap_values(X_scaled)
                
                # Create SHAP summary plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_scaled, feature_names=exg_features, show=False)
                plt.title('SHAP Feature Importance - ExG Prediction (Cotton Vigor)', fontsize=14, pad=20)
                plt.tight_layout()
                plt.savefig('../analysis/shap_exg_summary.png', dpi=300, bbox_inches='tight')
                plt.close()
                
                print("SHAP visualizations saved:")
                print("- ../analysis/shap_exg_summary.png")
                
            except Exception as e:
                print(f"SHAP visualization failed: {e}")
    
    def run_complete_analysis(self):
        """Run the complete analysis suite"""
        print("="*80)
        print("COTTON IRRIGATION SYNTHETIC DATA - COMPREHENSIVE ACCURACY ANALYSIS")
        print("="*80)
        
        # Load data
        self.load_and_prepare_data()
        
        # Train models
        self.train_models_for_analysis()
        
        # Run analyses
        model_results = self.comprehensive_model_evaluation()
        shap_results = self.shap_analysis()
        seasonal_results = self.seasonal_pattern_validation()
        
        # Treatment-specific analysis
        self.treatment_specific_accuracy()
        
        # Agricultural-specific analyses
        agricultural_results = self.agricultural_accuracy_metrics()
        self.cotton_growth_stage_analysis()
        self.temporal_accuracy_analysis()
        
        # Enhanced SHAP visualizations
        self.create_enhanced_shap_visualizations()
        
        # Generate visualizations
        self.generate_visualizations(model_results, shap_results, seasonal_results)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print("Generated files:")
        print("- ../analysis/model_performance_analysis.png")
        print("- ../analysis/distribution_comparison.png") 
        print("- ../analysis/seasonal_patterns.png")
        print("- ../analysis/shap_exg_summary.png (if SHAP available)")
        
        return {
            'model_results': model_results,
            'shap_results': shap_results,
            'seasonal_results': seasonal_results,
            'agricultural_results': agricultural_results
        }

def main():
    """Run the complete accuracy analysis"""
    analyzer = AccuracyAnalyzer()
    results = analyzer.run_complete_analysis()
    return results

if __name__ == "__main__":
    main() 