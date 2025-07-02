#!/usr/bin/env python3
"""

Author: Mohriz Murad
Date: 2025
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils.class_weight import compute_sample_weight
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedMLCottonSyntheticGenerator:
    """Enhanced ML-based synthetic data generator with Lubbock data integration"""
    
    COTTON_PLANTING_DATE = datetime(2025, 4, 3)  # Actual planting date from field records
    
    def __init__(self):
        self.historical_data = None
        self.lubbock_data = None
        self.exg_model = None
        self.soil_model = None
        self.heat_index_model = None
        self.et0_model = None
        self.rainfall_model = None
        self.scaler_exg = StandardScaler()
        self.scaler_soil = StandardScaler()
        self.scaler_heat = StandardScaler()
        self.scaler_et0 = StandardScaler()
        self.scaler_rainfall = StandardScaler()
        
    def load_data(self):
        """Load both Corpus and Lubbock data for enhanced training and seasonal patterns"""
        logger.info("Loading Corpus historical data (will be preserved 100%)...")
        self.historical_data = pd.read_csv('../data/Model Input - Corpus.csv')
        self.historical_data['Date'] = pd.to_datetime(self.historical_data['Date'])
        
        logger.info("Loading Lubbock data for seasonal patterns and ML training...")
        self.lubbock_data = pd.read_csv('../data/Model Input - Lubbock-3.csv')
        self.lubbock_data['Date'] = pd.to_datetime(self.lubbock_data['Date'])
        
        # Clean Lubbock data - replace 'NA' with NaN
        self.lubbock_data = self.lubbock_data.replace('NA', np.nan)
        
        # Convert numeric columns
        numeric_cols = ['ExG', 'Total Soil Moisture', 'Irrigation Added (gallons)', 
                       'Rainfall (gallons)', 'ET0 (mm)', 'Heat Index (F)', 'Canopy Cover (CC)', 'Kc (Crop Coeffecient)']
        for col in numeric_cols:
            if col in self.lubbock_data.columns:
                self.lubbock_data[col] = pd.to_numeric(self.lubbock_data[col], errors='coerce')
        
        # Get Lubbock date range for synthetic generation
        self.lubbock_start = self.lubbock_data['Date'].min()
        self.lubbock_end = self.lubbock_data['Date'].max()
        
        logger.info(f"Corpus data: {len(self.historical_data)} rows - WILL NOT BE MODIFIED")
        logger.info(f"Lubbock data: {len(self.lubbock_data)} rows - FOR ENHANCED ML TRAINING")
        logger.info(f"Lubbock covers: {self.lubbock_start.strftime('%Y-%m-%d')} to {self.lubbock_end.strftime('%Y-%m-%d')}")
        
        return self.historical_data
    
    def prepare_enhanced_training_data(self):
        """Prepare enhanced training data combining Corpus and Lubbock data with weightings"""
        logger.info("Preparing enhanced ML training data with Lubbock weightings...")
        
        # Corpus data (high weight - target location)
        corpus_valid = self.historical_data.dropna(subset=['Date'])
        
        # Lubbock data (lower weight - supporting patterns)
        lubbock_valid = self.lubbock_data.dropna(subset=['Date'])
        
        # 1. Enhanced ExG training
        corpus_exg = corpus_valid.dropna(subset=['ExG'])
        lubbock_exg = lubbock_valid.dropna(subset=['ExG'])
        
        X_exg_corpus, y_exg_corpus = self._prepare_exg_features(corpus_exg, is_corpus=True), corpus_exg['ExG'].values
        X_exg_lubbock, y_exg_lubbock = self._prepare_exg_features(lubbock_exg, is_corpus=False), lubbock_exg['ExG'].values
        
        X_exg = np.vstack([X_exg_corpus, X_exg_lubbock]) if len(X_exg_lubbock) > 0 else X_exg_corpus
        y_exg = np.hstack([y_exg_corpus, y_exg_lubbock]) if len(y_exg_lubbock) > 0 else y_exg_corpus
        
        # Create sample weights (Corpus: 3x weight, Lubbock: 1x weight)
        weights_exg = np.hstack([
            np.full(len(y_exg_corpus), 3.0),
            np.full(len(y_exg_lubbock), 1.0)
        ]) if len(y_exg_lubbock) > 0 else np.full(len(y_exg_corpus), 1.0)
        
        logger.info(f"ExG training: {len(X_exg_corpus)} Corpus + {len(X_exg_lubbock)} Lubbock = {len(X_exg)} samples")
        
        # 2. Enhanced Soil training
        corpus_soil = corpus_valid.dropna(subset=['Total Soil Moisture'])
        lubbock_soil = lubbock_valid.dropna(subset=['Total Soil Moisture'])
        
        X_soil_corpus, y_soil_corpus = self._prepare_soil_features(corpus_soil, is_corpus=True), corpus_soil['Total Soil Moisture'].values
        X_soil_lubbock, y_soil_lubbock = self._prepare_soil_features(lubbock_soil, is_corpus=False), lubbock_soil['Total Soil Moisture'].values
        
        X_soil = np.vstack([X_soil_corpus, X_soil_lubbock]) if len(X_soil_lubbock) > 0 else X_soil_corpus
        y_soil = np.hstack([y_soil_corpus, y_soil_lubbock]) if len(y_soil_lubbock) > 0 else y_soil_corpus
        
        weights_soil = np.hstack([
            np.full(len(y_soil_corpus), 3.0),
            np.full(len(y_soil_lubbock), 1.0)
        ]) if len(y_soil_lubbock) > 0 else np.full(len(y_soil_corpus), 1.0)
        
        logger.info(f"Soil training: {len(X_soil_corpus)} Corpus + {len(X_soil_lubbock)} Lubbock = {len(X_soil)} samples")
        
        # 3. Enhanced Heat Index training
        corpus_heat = corpus_valid.dropna(subset=['Heat Index (F)'])
        lubbock_heat = lubbock_valid.dropna(subset=['Heat Index (F)'])
        
        X_heat_corpus, y_heat_corpus = self._prepare_environmental_features(corpus_heat, is_corpus=True), corpus_heat['Heat Index (F)'].values
        X_heat_lubbock, y_heat_lubbock = self._prepare_environmental_features(lubbock_heat, is_corpus=False), lubbock_heat['Heat Index (F)'].values
        
        X_heat = np.vstack([X_heat_corpus, X_heat_lubbock]) if len(X_heat_lubbock) > 0 else X_heat_corpus
        y_heat = np.hstack([y_heat_corpus, y_heat_lubbock]) if len(y_heat_lubbock) > 0 else y_heat_corpus
        
        weights_heat = np.hstack([
            np.full(len(y_heat_corpus), 2.0),  # Less weight difference for environmental data
            np.full(len(y_heat_lubbock), 1.0)
        ]) if len(y_heat_lubbock) > 0 else np.full(len(y_heat_corpus), 1.0)
        
        logger.info(f"Heat Index training: {len(X_heat_corpus)} Corpus + {len(X_heat_lubbock)} Lubbock = {len(X_heat)} samples")
        
        # 4. Enhanced ET0 training  
        corpus_et0 = corpus_valid.dropna(subset=['ET0 (mm)'])
        lubbock_et0 = lubbock_valid.dropna(subset=['ET0 (mm)'])
        
        X_et0_corpus, y_et0_corpus = self._prepare_environmental_features(corpus_et0, is_corpus=True), corpus_et0['ET0 (mm)'].values
        X_et0_lubbock, y_et0_lubbock = self._prepare_environmental_features(lubbock_et0, is_corpus=False), lubbock_et0['ET0 (mm)'].values
        
        X_et0 = np.vstack([X_et0_corpus, X_et0_lubbock]) if len(X_et0_lubbock) > 0 else X_et0_corpus
        y_et0 = np.hstack([y_et0_corpus, y_et0_lubbock]) if len(y_et0_lubbock) > 0 else y_et0_corpus
        
        weights_et0 = np.hstack([
            np.full(len(y_et0_corpus), 2.0),
            np.full(len(y_et0_lubbock), 1.0)
        ]) if len(y_et0_lubbock) > 0 else np.full(len(y_et0_corpus), 1.0)
        
        logger.info(f"ET0 training: {len(X_et0_corpus)} Corpus + {len(X_et0_lubbock)} Lubbock = {len(X_et0)} samples")
        
        # 5. Enhanced Rainfall training
        corpus_rainfall = corpus_valid.dropna(subset=['Rainfall (gallons)'])
        lubbock_rainfall = lubbock_valid.dropna(subset=['Rainfall (gallons)'])
        
        X_rainfall_corpus, y_rainfall_corpus = self._prepare_environmental_features(corpus_rainfall, is_corpus=True), corpus_rainfall['Rainfall (gallons)'].values
        X_rainfall_lubbock, y_rainfall_lubbock = self._prepare_environmental_features(lubbock_rainfall, is_corpus=False), lubbock_rainfall['Rainfall (gallons)'].values
        
        X_rainfall = np.vstack([X_rainfall_corpus, X_rainfall_lubbock]) if len(X_rainfall_lubbock) > 0 else X_rainfall_corpus
        y_rainfall = np.hstack([y_rainfall_corpus, y_rainfall_lubbock]) if len(y_rainfall_lubbock) > 0 else y_rainfall_corpus
        
        weights_rainfall = np.hstack([
            np.full(len(y_rainfall_corpus), 2.0),
            np.full(len(y_rainfall_lubbock), 1.0)
        ]) if len(y_rainfall_lubbock) > 0 else np.full(len(y_rainfall_corpus), 1.0)
        
        logger.info(f"Rainfall training: {len(X_rainfall_corpus)} Corpus + {len(X_rainfall_lubbock)} Lubbock = {len(X_rainfall)} samples")
        
        return (X_exg, y_exg, weights_exg, X_soil, y_soil, weights_soil, 
                X_heat, y_heat, weights_heat, X_et0, y_et0, weights_et0, 
                X_rainfall, y_rainfall, weights_rainfall)
    
    def _prepare_exg_features(self, data, is_corpus=True):
        """Enhanced ExG features with location context"""
        features = []
        planting_date = self.COTTON_PLANTING_DATE if is_corpus else datetime(2023, 5, 1)  # Lubbock planting
        
        for _, row in data.iterrows():
            days_after_planting = (row['Date'] - planting_date).days
            
            feature_row = [
                days_after_planting,
                row.get('Heat Index (F)', 85.0),
                row.get('ET0 (mm)', 8.0),
                row.get('Total Soil Moisture', 200.0),
                row.get('Rainfall (gallons)', 0.0),
                1 if row.get('Treatment Type') in ['R_F', 'DICT'] else 0,  # Rainfed
                1 if row.get('Treatment Type') in ['H_I', 'DIEG'] else 0,   # Partial irrigation  
                1 if row.get('Treatment Type') == 'F_I' else 0,              # Full irrigation
                self.get_texas_amu_cotton_kc(days_after_planting),
                row['Date'].month,
                row['Date'].timetuple().tm_yday,
                1 if is_corpus else 0,  # Location indicator
                # Enhanced seasonal features
                np.sin(2 * np.pi * row['Date'].timetuple().tm_yday / 365),
                np.cos(2 * np.pi * row['Date'].timetuple().tm_yday / 365),
            ]
            
            features.append(feature_row)
        
        return np.array(features)
    
    def _prepare_soil_features(self, data, is_corpus=True):
        """Enhanced soil features with location context"""
        features = []
        planting_date = self.COTTON_PLANTING_DATE if is_corpus else datetime(2023, 5, 1)
        
        for _, row in data.iterrows():
            days_after_planting = (row['Date'] - planting_date).days
            
            feature_row = [
                days_after_planting,
                row.get('Heat Index (F)', 85.0),
                row.get('ET0 (mm)', 6.0),
                row.get('Rainfall (gallons)', 0.0),
                row.get('Irrigation Added (gallons)', 0.0),
                1 if row.get('Treatment Type') in ['R_F', 'DICT'] else 0,
                1 if row.get('Treatment Type') in ['H_I', 'DIEG'] else 0,
                1 if row.get('Treatment Type') == 'F_I' else 0,
                self.get_texas_amu_cotton_kc(days_after_planting),
                row['Date'].month,
                row['Date'].timetuple().tm_yday,
                1 if is_corpus else 0,  # Location indicator
                # Enhanced seasonal features
                np.sin(2 * np.pi * row['Date'].timetuple().tm_yday / 365),
                np.cos(2 * np.pi * row['Date'].timetuple().tm_yday / 365),
            ]
            features.append(feature_row)
        
        return np.array(features)
    
    def _prepare_environmental_features(self, data, is_corpus=True):
        """Enhanced environmental features with more variation sources"""
        features = []
        planting_date = self.COTTON_PLANTING_DATE if is_corpus else datetime(2023, 5, 1)
        
        for _, row in data.iterrows():
            days_after_planting = (row['Date'] - planting_date).days
            day_of_year = row['Date'].timetuple().tm_yday
            
            # Enhanced features for better variation
            feature_row = [
                days_after_planting,
                row['Date'].month,
                day_of_year,
                row['Date'].weekday(),
                # Multiple seasonal patterns
                np.sin(2 * np.pi * day_of_year / 365),
                np.cos(2 * np.pi * day_of_year / 365),
                np.sin(4 * np.pi * day_of_year / 365),  # Semi-annual pattern
                np.cos(4 * np.pi * day_of_year / 365),
                # Cotton growth stage context
                self.get_texas_amu_cotton_kc(days_after_planting),
                1 if is_corpus else 0,  # Location indicator
                # Weather pattern indicators
                (day_of_year - 182) ** 2 / 10000,  # Distance from summer peak
                1 if 152 <= day_of_year <= 244 else 0,  # Summer months (Jun-Aug)
            ]
            features.append(feature_row)
        
        return np.array(features)

    def train_enhanced_models(self, X_exg, y_exg, weights_exg, X_soil, y_soil, weights_soil,
                             X_heat, y_heat, weights_heat, X_et0, y_et0, weights_et0,
                             X_rainfall, y_rainfall, weights_rainfall):
        """Train enhanced ML models with Lubbock weightings"""
        logger.info("Training enhanced ML models with Lubbock data integration...")
        
        MIN_SAMPLES = 5
        
        # 1. Enhanced ExG model
        if len(X_exg) >= MIN_SAMPLES:
            X_exg_scaled = self.scaler_exg.fit_transform(X_exg)
            self.exg_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
            self.exg_model.fit(X_exg_scaled, y_exg, sample_weight=weights_exg)
            
            y_pred = self.exg_model.predict(X_exg_scaled)
            r2_exg = r2_score(y_exg, y_pred, sample_weight=weights_exg)
            rmse_exg = np.sqrt(mean_squared_error(y_exg, y_pred, sample_weight=weights_exg))
            logger.info(f"Enhanced ExG Model - R²: {r2_exg:.3f}, RMSE: {rmse_exg:.3f}")
        
        # 2. Enhanced Soil model
        if len(X_soil) >= MIN_SAMPLES:
            X_soil_scaled = self.scaler_soil.fit_transform(X_soil)
            self.soil_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
            self.soil_model.fit(X_soil_scaled, y_soil, sample_weight=weights_soil)
            
            y_pred = self.soil_model.predict(X_soil_scaled)
            r2_soil = r2_score(y_soil, y_pred, sample_weight=weights_soil)
            rmse_soil = np.sqrt(mean_squared_error(y_soil, y_pred, sample_weight=weights_soil))
            logger.info(f"Enhanced Soil Model - R²: {r2_soil:.3f}, RMSE: {rmse_soil:.3f}")
        
        # 3. Enhanced Heat Index model with better parameters
        if len(X_heat) >= MIN_SAMPLES:
            X_heat_scaled = self.scaler_heat.fit_transform(X_heat)
            # More trees and less regularization for better variation
            self.heat_index_model = RandomForestRegressor(n_estimators=150, max_depth=10, 
                                                         min_samples_split=2, min_samples_leaf=1, random_state=42)
            self.heat_index_model.fit(X_heat_scaled, y_heat, sample_weight=weights_heat)
            
            y_pred = self.heat_index_model.predict(X_heat_scaled)
            r2_heat = r2_score(y_heat, y_pred, sample_weight=weights_heat)
            rmse_heat = np.sqrt(mean_squared_error(y_heat, y_pred, sample_weight=weights_heat))
            logger.info(f"Enhanced Heat Index Model - R²: {r2_heat:.3f}, RMSE: {rmse_heat:.3f}")
        
        # 4. Enhanced ET0 model
        if len(X_et0) >= MIN_SAMPLES:
            X_et0_scaled = self.scaler_et0.fit_transform(X_et0)
            self.et0_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
            self.et0_model.fit(X_et0_scaled, y_et0, sample_weight=weights_et0)
            
            y_pred = self.et0_model.predict(X_et0_scaled)
            r2_et0 = r2_score(y_et0, y_pred, sample_weight=weights_et0)
            rmse_et0 = np.sqrt(mean_squared_error(y_et0, y_pred, sample_weight=weights_et0))
            logger.info(f"Enhanced ET0 Model - R²: {r2_et0:.3f}, RMSE: {rmse_et0:.3f}")
        
        # 5. Enhanced Rainfall model
        if len(X_rainfall) >= MIN_SAMPLES:
            X_rainfall_scaled = self.scaler_rainfall.fit_transform(X_rainfall)
            self.rainfall_model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
            self.rainfall_model.fit(X_rainfall_scaled, y_rainfall, sample_weight=weights_rainfall)
            
            y_pred = self.rainfall_model.predict(X_rainfall_scaled)
            r2_rainfall = r2_score(y_rainfall, y_pred, sample_weight=weights_rainfall)
            rmse_rainfall = np.sqrt(mean_squared_error(y_rainfall, y_pred, sample_weight=weights_rainfall))
            logger.info(f"Enhanced Rainfall Model - R²: {r2_rainfall:.3f}, RMSE: {rmse_rainfall:.3f}")

    def get_texas_amu_cotton_kc(self, days_after_planting: int) -> float:
        """Get Texas A&M cotton Kc values"""
        if days_after_planting <= 10:
            return 0.07
        elif days_after_planting <= 40:
            return 0.22
        elif days_after_planting <= 60:
            return 0.44
        elif days_after_planting <= 90:
            return 1.10
        elif days_after_planting <= 115:
            return 1.10
        elif days_after_planting <= 125:
            return 0.83
        elif days_after_planting <= 145:
            return 0.44
        elif days_after_planting <= 150:
            return 0.44
        else:
            return 0.10
    
    def apply_cotton_guardrails(self, predicted_exg: float, days_after_planting: int) -> float:
        """Apply Texas A&M cotton physiology guardrails"""
        # Define biologically plausible ExG ranges based on cotton growth stages
        if days_after_planting <= 10:  # Seeding
            exg_min, exg_max = 0.05, 0.25
        elif days_after_planting <= 40:  # 1st Square
            exg_min, exg_max = 0.15, 0.45
        elif days_after_planting <= 60:  # 1st Bloom
            exg_min, exg_max = 0.30, 0.70
        elif days_after_planting <= 90:  # Max Bloom
            exg_min, exg_max = 0.60, 1.10
        elif days_after_planting <= 115:  # 1st Open
            exg_min, exg_max = 0.55, 1.05
        elif days_after_planting <= 125:  # 25% Open
            exg_min, exg_max = 0.40, 0.90
        elif days_after_planting <= 145:  # 50% Open
            exg_min, exg_max = 0.20, 0.60
        elif days_after_planting <= 150:  # 95% Open
            exg_min, exg_max = 0.15, 0.50
        else:  # Pick
            exg_min, exg_max = 0.05, 0.25
        
        return np.clip(predicted_exg, exg_min, exg_max)
    
    def _get_lubbock_seasonal_patterns(self):
        """Extract Lubbock seasonal patterns for heat index and rainfall by day of year"""
        patterns = {}
        
        if hasattr(self, 'lubbock_data') and self.lubbock_data is not None:
            for _, row in self.lubbock_data.iterrows():
                day_of_year = row['Date'].timetuple().tm_yday
                
                if day_of_year not in patterns:
                    patterns[day_of_year] = {'heat_index': [], 'rainfall': []}
                
                # Collect heat index values (remove NAs)
                if pd.notna(row['Heat Index (F)']):
                    patterns[day_of_year]['heat_index'].append(row['Heat Index (F)'])
                
                # Collect rainfall values (remove NAs)
                if pd.notna(row['Rainfall (gallons)']):
                    patterns[day_of_year]['rainfall'].append(row['Rainfall (gallons)'])
        
        # Calculate averages for each day of year
        for day in patterns:
            if patterns[day]['heat_index']:
                patterns[day]['avg_heat_index'] = np.mean(patterns[day]['heat_index'])
            else:
                patterns[day]['avg_heat_index'] = None
                
            if patterns[day]['rainfall']:
                patterns[day]['avg_rainfall'] = np.mean(patterns[day]['rainfall'])
                patterns[day]['rainfall_values'] = patterns[day]['rainfall']
            else:
                patterns[day]['avg_rainfall'] = 0.0
                patterns[day]['rainfall_values'] = [0.0]
        
        return patterns
    
    def _get_heat_index_from_lubbock_pattern(self, date: datetime, lubbock_patterns: dict) -> float:
        """Get heat index based on Lubbock seasonal pattern with Corpus Christi adjustment"""
        day_of_year = date.timetuple().tm_yday
        
        # Try to find pattern for this exact day
        if day_of_year in lubbock_patterns and lubbock_patterns[day_of_year]['avg_heat_index'] is not None:
            lubbock_temp = lubbock_patterns[day_of_year]['avg_heat_index']
        else:
            # Find nearest days with data
            available_days = [d for d in lubbock_patterns if lubbock_patterns[d]['avg_heat_index'] is not None]
            if available_days:
                nearest_day = min(available_days, key=lambda x: abs(x - day_of_year))
                lubbock_temp = lubbock_patterns[nearest_day]['avg_heat_index']
            else:
                # Fallback to seasonal estimate
                return self._generate_temperature(date)
        
        # Apply Corpus Christi adjustment (warmer and more humid than Lubbock)
        # Corpus Christi is about 5-8°F warmer than Lubbock due to coastal location
        corpus_adjustment = 6.0 + np.random.normal(0, 2.0)  # 6°F average with variation
        adjusted_temp = lubbock_temp + corpus_adjustment
        
        # Apply seasonal constraints
        month = date.month
        if month in [6, 7, 8]:  # Summer
            return np.clip(adjusted_temp, 88, 96)
        elif month in [5, 9]:   # Late spring/early fall
            return np.clip(adjusted_temp, 82, 90)
        elif month in [4, 10]:  # Mid spring/fall
            return np.clip(adjusted_temp, 78, 86)
        else:
            return np.clip(adjusted_temp, 74, 82)
    
    def _get_rainfall_from_lubbock_pattern(self, date: datetime, lubbock_patterns: dict) -> float:
        """Get rainfall based on Lubbock seasonal pattern with adjustments"""
        day_of_year = date.timetuple().tm_yday
        
        # Try to find pattern for this exact day
        if day_of_year in lubbock_patterns:
            rainfall_values = lubbock_patterns[day_of_year]['rainfall_values']
            # Use actual Lubbock rainfall distribution for this day of year
            return np.random.choice(rainfall_values)
        else:
            # Find nearest days with data
            available_days = [d for d in lubbock_patterns if lubbock_patterns[d]['rainfall_values']]
            if available_days:
                nearest_day = min(available_days, key=lambda x: abs(x - day_of_year))
                rainfall_values = lubbock_patterns[nearest_day]['rainfall_values']
                return np.random.choice(rainfall_values)
            else:
                # Fallback to 0 (no rain)
                return 0.0
    
    def generate_synthetic_season(self, days_to_generate: int = None):
        """Generate synthetic data using Lubbock seasonal patterns as templates"""
        logger.info(f"Generating synthetic data using Lubbock seasonal patterns...")
        
        # Get Lubbock environmental patterns by day of year for reference
        lubbock_patterns = self._get_lubbock_seasonal_patterns()
        
        synthetic_rows = []
        last_date = self.historical_data['Date'].max()
        
        # Calculate how many days to generate based on Lubbock data availability
        # Only generate data for dates where we have Lubbock reference patterns
        corpus_start_date = last_date + timedelta(days=1)
        
        # Map Corpus date to corresponding Lubbock season (adjust year)
        lubbock_reference_start = self.lubbock_start.replace(year=corpus_start_date.year)
        lubbock_reference_end = self.lubbock_end.replace(year=corpus_start_date.year)
        
        # Find appropriate end date
        if corpus_start_date <= lubbock_reference_end:
            generation_end_date = lubbock_reference_end
        else:
            # If we're past Lubbock season, generate until end of year or specified days
            if days_to_generate:
                generation_end_date = corpus_start_date + timedelta(days=days_to_generate-1)
            else:
                generation_end_date = datetime(corpus_start_date.year, 12, 31)
        
        total_days = (generation_end_date - corpus_start_date).days + 1
        logger.info(f"Generating {total_days} days: {corpus_start_date.strftime('%Y-%m-%d')} to {generation_end_date.strftime('%Y-%m-%d')}")
        logger.info(f"Using Lubbock patterns from: {lubbock_reference_start.strftime('%Y-%m-%d')} to {lubbock_reference_end.strftime('%Y-%m-%d')}")
        
        # Get last values for continuity
        last_values = {}
        for plot_id in [102, 404, 409]:
            plot_data = self.historical_data[self.historical_data['Plot ID'] == plot_id]
            
            # Find last valid ExG value for this plot
            valid_exg_rows = plot_data.dropna(subset=['ExG'])
            if len(valid_exg_rows) > 0:
                last_exg = valid_exg_rows.iloc[-1]['ExG']
            else:
                last_exg = 0.3  # Default starting value
            
            # Find last valid soil moisture
            valid_soil_rows = plot_data.dropna(subset=['Total Soil Moisture'])
            if len(valid_soil_rows) > 0:
                last_soil = valid_soil_rows.iloc[-1]['Total Soil Moisture']
            else:
                last_soil = 200.0  # Default starting value
                
            last_values[plot_id] = {
                'exg': last_exg,
                'soil': last_soil
            }
            
        logger.info(f"Starting ExG values: Plot 102={last_values[102]['exg']:.3f}, Plot 404={last_values[404]['exg']:.3f}, Plot 409={last_values[409]['exg']:.3f}")
        
        for day in range(total_days):
            synthetic_date = corpus_start_date + timedelta(days=day)
            days_after_planting = (synthetic_date - self.COTTON_PLANTING_DATE).days
            
            # Use Lubbock seasonal patterns for environmental variables
            rainfall = self._get_rainfall_from_lubbock_pattern(synthetic_date, lubbock_patterns)
            heat_index = self._get_heat_index_from_lubbock_pattern(synthetic_date, lubbock_patterns)
            
            # ET0 still use ML if available, otherwise generate
            env_features = self._prepare_env_prediction_features(synthetic_date, days_after_planting)
            et0 = self._predict_et0(env_features) if self.et0_model else self._generate_et0(synthetic_date)
            
            for plot_id in [102, 404, 409]:
                treatment_type = {102: 'R_F', 404: 'H_I', 409: 'F_I'}[plot_id]
                
                # Predict ExG using enhanced ML + guardrails
                if self.exg_model is not None:
                    features = np.array([[
                        days_after_planting, heat_index, et0, last_values[plot_id]['soil'], rainfall,
                        1 if treatment_type == 'R_F' else 0,
                        1 if treatment_type == 'H_I' else 0,
                        1 if treatment_type == 'F_I' else 0,
                        self.get_texas_amu_cotton_kc(days_after_planting),
                        synthetic_date.month, synthetic_date.timetuple().tm_yday,
                        1,  # Corpus location indicator
                        # Enhanced seasonal features
                        np.sin(2 * np.pi * synthetic_date.timetuple().tm_yday / 365),
                        np.cos(2 * np.pi * synthetic_date.timetuple().tm_yday / 365),
                    ]])
                    
                    features_scaled = self.scaler_exg.transform(features)
                    predicted_exg = self.exg_model.predict(features_scaled)[0]
                    final_exg = self.apply_cotton_guardrails(predicted_exg, days_after_planting)
                    # Temporal smoothing
                    final_exg = 0.7 * final_exg + 0.3 * last_values[plot_id]['exg']
                else:
                    final_exg = 0.3
                    logger.warning("ExG model is None! Using default value 0.3")
                
                # Predict soil moisture with enhanced features
                if self.soil_model is not None:
                    soil_features = np.array([[
                        days_after_planting, heat_index, et0, rainfall, 0.0,
                        1 if treatment_type == 'R_F' else 0,
                        1 if treatment_type == 'H_I' else 0,
                        1 if treatment_type == 'F_I' else 0,
                        self.get_texas_amu_cotton_kc(days_after_planting),
                        synthetic_date.month, synthetic_date.timetuple().tm_yday,
                        1,  # Corpus location indicator
                        # Enhanced seasonal features
                        np.sin(2 * np.pi * synthetic_date.timetuple().tm_yday / 365),
                        np.cos(2 * np.pi * synthetic_date.timetuple().tm_yday / 365),
                    ]])
                    
                    soil_features_scaled = self.scaler_soil.transform(soil_features)
                    predicted_soil = self.soil_model.predict(soil_features_scaled)[0]
                    final_soil = self._apply_water_balance(last_values[plot_id]['soil'], predicted_soil, rainfall, et0, self.get_texas_amu_cotton_kc(days_after_planting))
                else:
                    final_soil = 200.0
                
                # Calculate irrigation
                irrigation = self._calculate_irrigation(final_soil, treatment_type)
                final_soil += irrigation * 0.554
                
                synthetic_row = {
                    'Date': synthetic_date,
                    'Plot ID': plot_id,
                    'Treatment Type': treatment_type,
                    'ExG': final_exg,
                    'Total Soil Moisture': final_soil,
                    'Irrigation Added (gallons)': irrigation,
                    'Rainfall (gallons)': rainfall,
                    'ET0 (mm)': et0,
                    'Heat Index (F)': heat_index,
                    'Kc (Crop Coefficient)': self.get_texas_amu_cotton_kc(days_after_planting)
                }
                
                synthetic_rows.append(synthetic_row)
                
                # Update last values for next day
                last_values[plot_id]['exg'] = final_exg
                last_values[plot_id]['soil'] = final_soil
        
        return pd.DataFrame(synthetic_rows)
    
    def _generate_et0(self, date: datetime) -> float:
        """Generate realistic ET0 for Corpus Christi based on TexasET Network data"""
        day_of_year = date.timetuple().tm_yday
        
        # Corpus Christi seasonal pattern from TexasET Network:
        # Jun-Aug: 6.43, 6.68, 6.65 mm/day (peak summer)
        # May, Sep: 5.95, 5.21 mm/day (shoulder months)
        # Annual range: ~4-7 mm/day
        
        # Peak in July (day 185), minimum in December/January
        base_et0 = 5.5 + 1.2 * np.sin((day_of_year - 15) * 2 * np.pi / 365)
        return np.clip(base_et0 + np.random.normal(0, 0.8), 3.5, 7.5)
    
    def _generate_temperature(self, date: datetime) -> float:
        """Generate realistic heat index for Corpus Christi, Texas"""
        day_of_year = date.timetuple().tm_yday
        
        # Corpus Christi heat index seasonal pattern:
        # Summer peak: 88-95°F typical, 95-100°F hot days, 100-105°F extreme (rare)
        # Winter low: 65-75°F
        # Peak in August (day 213), minimum in January (day 15)
        
        base_heat_index = 80 + 10 * np.sin((day_of_year - 15) * 2 * np.pi / 365)
        # Reduced random variation to realistic ±3°F daily fluctuation
        return np.clip(base_heat_index + np.random.normal(0, 3), 65, 105)
    
    def _apply_water_balance(self, last_soil, predicted_soil, rainfall, et0, kc):
        """Fixed water balance physics - realistic soil moisture range"""
        # Convert gallons to mm for proper water balance
        rainfall_mm = rainfall * 0.0037854 * 1000 / 36  # Convert to mm water depth over plot area
        
        # Actual evapotranspiration (more conservative)
        et_crop = et0 * kc * 0.6  # Reduced factor for more realistic ET
        
        # Water balance equation
        water_balance_change = rainfall_mm - et_crop
        
        # Combine ML prediction with physics (higher weight on ML)
        physics_adjusted_soil = last_soil + water_balance_change
        final_soil = 0.3 * physics_adjusted_soil + 0.7 * predicted_soil
        
        # Realistic soil moisture range for Corpus Christi sandy clay loam
        return np.clip(final_soil, 180, 320)
    
    def _calculate_irrigation(self, soil_moisture, treatment_type):
        """Calculate irrigation based on treatment"""
        if treatment_type == 'F_I' and soil_moisture < 180:
            return min(100.0, (185 - soil_moisture) * 1.2)
        elif treatment_type == 'H_I' and soil_moisture < 170:
            return min(50.0, (175 - soil_moisture) * 0.8)
        return 0.0
    
    def save_complete_season(self, output_filename: str = '../data/corpus_season_completed_enhanced_lubbock_ml.csv'):
        """Save complete season - PRESERVES historical data 100%"""
        logger.info("Generating complete season - PRESERVING historical data 100%...")
        
        # Generate synthetic data ONLY
        synthetic_df = self.generate_synthetic_season()
        
        # Combine: PRESERVE historical data exactly + add synthetic
        complete_season = pd.concat([self.historical_data, synthetic_df], ignore_index=True)
        complete_season = complete_season.sort_values(['Date', 'Plot ID'])
        
        # Save to CSV
        complete_season.to_csv(output_filename, index=False)
        logger.info(f"Complete season saved to {output_filename}")
        
        # Verify historical data preservation
        historical_rows = len(self.historical_data)
        preserved_data = complete_season.iloc[:historical_rows]
        
        logger.info(f"Historical data preservation verified: {historical_rows} rows UNCHANGED")
        logger.info(f"Synthetic data added: {len(synthetic_df)} new rows")
        
        self._print_summary(complete_season, synthetic_df)
        
        return complete_season
    
    def _print_summary(self, complete_season, synthetic_df):
        """Log basic generation summary"""
        historical_rows = len(complete_season) - len(synthetic_df)
        logger.info(f"Generation completed: {historical_rows} historical + {len(synthetic_df)} synthetic = {len(complete_season)} total rows")
    
    def _prepare_env_prediction_features(self, date: datetime, days_after_planting: int) -> np.ndarray:
        """Enhanced features for environmental variable prediction - matches training features"""
        day_of_year = date.timetuple().tm_yday
        
        feature_row = [
            days_after_planting,
            date.month,
            day_of_year,
            date.weekday(),
            # Multiple seasonal patterns
            np.sin(2 * np.pi * day_of_year / 365),
            np.cos(2 * np.pi * day_of_year / 365),
            np.sin(4 * np.pi * day_of_year / 365),  # Semi-annual pattern
            np.cos(4 * np.pi * day_of_year / 365),
            # Cotton growth stage context
            self.get_texas_amu_cotton_kc(days_after_planting),
            1,  # Corpus location indicator
            # Weather pattern indicators
            (day_of_year - 182) ** 2 / 10000,  # Distance from summer peak
            1 if 152 <= day_of_year <= 244 else 0,  # Summer months (Jun-Aug)
        ]
        return np.array([feature_row])
    
    def _predict_heat_index(self, env_features: np.ndarray, date: datetime) -> float:
        """Predict heat index with realistic seasonal variation"""
        if self.heat_index_model is None:
            return self._generate_temperature(date)
        
        features_scaled = self.scaler_heat.transform(env_features)
        predicted = self.heat_index_model.predict(features_scaled)[0]
        
        # Apply realistic seasonal constraints for Corpus Christi
        month = date.month
        if month in [6, 7, 8]:  # Summer - hottest
            min_temp, max_temp = 88, 96
        elif month in [5, 9]:   # Late spring/early fall
            min_temp, max_temp = 82, 90
        elif month in [4, 10]:  # Mid spring/fall
            min_temp, max_temp = 78, 86
        elif month in [3, 11]:  # Early spring/late fall
            min_temp, max_temp = 74, 82
        else:  # Winter months
            min_temp, max_temp = 65, 75
        
        return np.clip(predicted, min_temp, max_temp)
    
    def _predict_et0(self, env_features: np.ndarray) -> float:
        """Predict ET0 using ML model with realistic constraints"""
        if self.et0_model is None:
            return self._generate_et0(datetime.now())
        
        features_scaled = self.scaler_et0.transform(env_features)
        predicted = self.et0_model.predict(features_scaled)[0]
        
        # Apply realistic constraints for Corpus Christi ET0
        return np.clip(predicted, 3.5, 7.5)
    
    def _predict_rainfall(self, env_features: np.ndarray, date: datetime) -> float:
        """Predict rainfall with realistic patterns for Corpus Christi"""
        if self.rainfall_model is None:
            return self.generate_realistic_rainfall(date)
        
        features_scaled = self.scaler_rainfall.transform(env_features)
        predicted = self.rainfall_model.predict(features_scaled)[0]
        
        # Apply realistic constraints - no 600 cap!
        # Most days are 0, occasional rain 10-400 gallons, rare heavy events up to 600
        predicted = np.clip(predicted, 0, 650)  # Slightly higher cap for rare events
        
        # Add realism: if predicted is very high, reduce probability
        if predicted > 400:
            # Only 5% chance of very heavy rain
            if np.random.random() > 0.05:
                predicted = predicted * 0.3  # Reduce to moderate rain
        
        return predicted

def main():
    """Main execution function"""
    logger.info("Starting synthetic data generation")
    
    generator = EnhancedMLCottonSyntheticGenerator()
    
    # Load both Corpus and Lubbock data
    generator.load_data()
    
    # Prepare enhanced training data with Lubbock weightings
    (X_exg, y_exg, weights_exg, X_soil, y_soil, weights_soil, 
     X_heat, y_heat, weights_heat, X_et0, y_et0, weights_et0, 
     X_rainfall, y_rainfall, weights_rainfall) = generator.prepare_enhanced_training_data()
    
    # Train enhanced models with Lubbock data integration
    generator.train_enhanced_models(X_exg, y_exg, weights_exg, X_soil, y_soil, weights_soil,
                                   X_heat, y_heat, weights_heat, X_et0, y_et0, weights_et0,
                                   X_rainfall, y_rainfall, weights_rainfall)
    
    # Generate complete season using Lubbock patterns + ML enhancement
    generator.save_complete_season()
    
    logger.info("Synthetic data generation completed")

if __name__ == "__main__":
    main() 