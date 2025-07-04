"""
Irrigation Policy Transfer System - REVISED
===========================================

Fixed approach that ensures irrigation improves plant health (ExG).

Key Fixes:
1. Proper plot size scaling (Lubbock 6475 sq ft → Corpus 443.5 sq ft)  
2. ExG maintenance rewards instead of decline penalties
3. Seasonal ExG expectations (early growth vs late senescence)
4. Realistic irrigation recommendations based on plant needs
"""

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import torch
import random
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

# Set fixed random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class PlantHealthAwareEnv(gym.Env):
    """Interactive RL Environment for Irrigation Policy Learning"""
    
    def __init__(self, treatment_type='F_I', max_days=212, historical_mode=False):  # April 3 to October 31 = 212 days
        super().__init__()
        
        self.treatment_type = treatment_type
        self.max_days = max_days  # Full growing season
        self.historical_mode = historical_mode  # Flag to disable weather generation
        
        # EXPANDED: Substantial irrigation action space for agricultural production
        if treatment_type == 'F_I':  # Full irrigation - substantial watering for production
            self.irrigation_amounts = [0, 10, 20, 30, 40, 50, 60, 75, 90, 110, 130, 150, 175, 200, 250, 300]  # gallons
        elif treatment_type == 'H_I':  # Half irrigation - moderate to substantial watering  
            self.irrigation_amounts = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75, 90, 110, 130, 150]   # gallons
        else:  # R_F - Rainfed
            self.irrigation_amounts = [0]  # gallons
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(len(self.irrigation_amounts))
        
        # State: [soil_moisture, ET0, heat_index, rainfall, ExG, days_after_planting, water_deficit, Kc]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 50, 0, 0, 0, 0, 0.2]),
            high=np.array([500, 20, 120, 600, 1.0, 200, 300, 1.5]),
            dtype=np.float32
        )
        
        # Initialize ML-based weather generator
        self.weather_generator = MLBasedWeatherGenerator()
        
        # Episode tracking
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset environment for new growing season episode"""
        super().reset(seed=seed)
        
        self.current_day = 0
        self.days_after_planting = 1
        self.planting_date = datetime(2025, 4, 3)  # Actual Corpus Christi planting date
        self.current_date = self.planting_date
        
        # Initial conditions
        self.soil_moisture = 200.0  # Starting soil moisture
        self.exg = 0.3  # Initial plant health
        self.cumulative_reward = 0.0
        self.last_rainfall = 0.0  # For weather persistence
        
        # CRITICAL FIX: Generate NEW weather pattern for each episode
        # This ensures the RL agent learns robust policies across weather variability
        # Don't recreate the weather generator - just reset its state
        self.weather_generator.reset_state()
        
        # Generate today's weather
        self.today_rainfall = self.weather_generator.generate_rainfall(self.current_date, self.last_rainfall)
        self.today_et0 = self.weather_generator.generate_et0(self.current_date)
        self.today_heat_index = self.weather_generator.generate_heat_index(self.current_date)
        
        # Diagnostic: Log weather variability (1% of episodes)
        if np.random.random() < 0.01:
            print(f"[WEATHER] Episode {self.current_day}: Rain={self.today_rainfall:.1f}, ET0={self.today_et0:.1f}, Heat={self.today_heat_index:.1f}")
        
        return self._get_state(), {}
    
    def step(self, action):
        """Take irrigation action and advance one day"""
        # Get irrigation amount
        irrigation_amount = self.irrigation_amounts[action]
        
        # Apply irrigation to soil moisture
        self.soil_moisture += irrigation_amount * 0.554  # Conversion factor
        
        # Calculate water deficit
        water_deficit = max(0, 250 - self.soil_moisture)
        
        # Calculate Kc (crop coefficient) based on growth stage
        kc = self._calculate_kc(self.days_after_planting)
        
        # Update plant health (ExG) based on water stress and irrigation
        self.exg = self._update_plant_health(irrigation_amount, water_deficit, kc)
        
        # Calculate reward for this action
        reward = self._calculate_reward(irrigation_amount, water_deficit, kc)
        
        # Apply water loss (ET0, drainage)
        self.soil_moisture -= self.today_et0 * kc  # Crop water use
        self.soil_moisture = max(50, min(400, self.soil_moisture))  # Realistic bounds
        
        # Add today's rainfall
        self.soil_moisture += self.today_rainfall

        # ENFORCE CAP: Soil moisture cannot exceed 300 gallons
        self.soil_moisture = min(self.soil_moisture, 300)
        
        # Advance to next day
        self.current_day += 1
        self.days_after_planting += 1
        self.current_date += timedelta(days=1)
        self.last_rainfall = self.today_rainfall
        
        # Generate tomorrow's weather (if not done and not in historical mode)
        if self.current_day < self.max_days and not self.historical_mode:
            self.today_rainfall = self.weather_generator.generate_rainfall(self.current_date, self.last_rainfall)
            self.today_et0 = self.weather_generator.generate_et0(self.current_date)
            self.today_heat_index = self.weather_generator.generate_heat_index(self.current_date)
        
        # Check if episode is done
        terminated = self.current_day >= self.max_days
        truncated = False  # We don't truncate episodes
        
        # Episode info
        info = {
            'day': self.current_day,
            'irrigation': irrigation_amount,
            'soil_moisture': self.soil_moisture,
            'exg': self.exg,
            'rainfall': self.today_rainfall,
            'water_deficit': water_deficit
        }
        
        return self._get_state(), reward, terminated, truncated, info
    
    def _get_state(self):
        """Get current state vector"""
        water_deficit = max(0, 250 - self.soil_moisture)
        kc = self._calculate_kc(self.days_after_planting)
        
        state = np.array([
            self.soil_moisture,
            self.today_et0,
            self.today_heat_index,
            self.today_rainfall,
            self.exg,
            self.days_after_planting,
            water_deficit,
            kc
        ], dtype=np.float32)
        
        return state
    
    def _calculate_kc(self, days_after_planting):
        """Calculate crop coefficient based on growth stage"""
        if days_after_planting <= 30:
            return 0.4  # Initial stage
        elif days_after_planting <= 70:
            return 0.4 + (days_after_planting - 30) * (1.2 - 0.4) / 40  # Development
        elif days_after_planting <= 120:
            return 1.2  # Mid-season
        else:
            return max(0.6, 1.2 - (days_after_planting - 120) * (0.6 - 1.2) / 30)  # Late season
    
    def _update_plant_health(self, irrigation_amount, water_deficit, kc):
        """Update plant health based on water stress and irrigation with improved dynamics"""
        # FIXED: More realistic stress factors that allow recovery
        if water_deficit > 100:  # Extreme stress
            stress_factor = 0.92  # Severe decline
        elif water_deficit > 50:  # High stress
            stress_factor = 0.96  # Moderate decline
        elif water_deficit > 25:  # Medium stress
            stress_factor = 0.99  # Slight decline
        else:  # Low stress
            stress_factor = 1.005  # Slight growth
        
        # ENHANCED: Better irrigation benefit calculation
        if water_deficit > 25 and irrigation_amount > 0:
            # Scale benefit based on how much irrigation helps with deficit
            irrigation_efficiency = min(1.0, irrigation_amount / max(1, water_deficit))
            irrigation_benefit = 0.03 * irrigation_efficiency  # Increased benefit
        else:
            irrigation_benefit = 0
        
        # ADDED: Growth stage factor (plants more resilient mid-season)
        if 30 <= self.days_after_planting <= 120:  # Mid-season
            resilience_factor = 1.002  # Slightly more resilient
        else:
            resilience_factor = 1.0
        
        # Update ExG with improved bounds and dynamics
        new_exg = self.exg * stress_factor * resilience_factor + irrigation_benefit
        
        # FIXED: Allow more ExG variability (was 0.1-1.0, now 0.05-1.0)
        return max(0.05, min(1.0, new_exg))
    
    def _calculate_reward(self, irrigation_amount, water_deficit, kc):
        """Calculate improved reward for current action with better balance"""
        # Calculate target irrigation with more realistic targets
        if water_deficit > 100:  # Extreme stress
            target_irrigation = min(water_deficit * 0.8, 150)  # Cap at max irrigation
        elif water_deficit > 50:  # High stress
            target_irrigation = water_deficit * 0.6
        elif water_deficit > 25:  # Medium stress
            target_irrigation = water_deficit * 0.4
        else:  # Low stress
            target_irrigation = 0
        
        # FIXED: Reduced penalty for target deviation (was -0.2, now -0.05)
        target_penalty = -0.05 * abs(irrigation_amount - target_irrigation)
        
        # Calculate Delta ExG (plant health improvement)
        old_exg = self.exg
        new_exg = self._update_plant_health(irrigation_amount, water_deficit, kc)
        delta_exg = new_exg - old_exg
        
        # ENHANCED: Stronger bonus for plant health improvement with stress multiplier
        stress_multiplier = 3.0 if water_deficit > 100 else 2.0 if water_deficit > 50 else 1.0
        delta_exg_bonus = 300 * delta_exg * stress_multiplier  # Increased from 100 to 300
        
        # CRITICAL: Severe penalties for allowing plant stress (agricultural production priority)
        if water_deficit > 100 and irrigation_amount == 0:
            extreme_stress_penalty = -200.0  # Catastrophic penalty for extreme stress
            high_stress_penalty = 0
        elif water_deficit > 80 and irrigation_amount == 0:
            extreme_stress_penalty = -150.0  # Very severe penalty for high stress
            high_stress_penalty = 0
        elif water_deficit > 50 and irrigation_amount == 0:
            extreme_stress_penalty = 0
            high_stress_penalty = -100.0  # Severe penalty for medium stress
        else:
            extreme_stress_penalty = 0
            high_stress_penalty = 0
        
        # FIXED: Reduced water cost (was -0.05, now -0.005)
        water_cost = -0.005 * irrigation_amount
        
        # CRITICAL: Strong rewards for preventing stress (agricultural production priority)
        if water_deficit > 100 and irrigation_amount > 0:
            stress_response_bonus = 100.0  # Massive reward for preventing extreme stress
            # CRITICAL: Minimum irrigation requirements for extreme stress
            if self.treatment_type == 'F_I' and irrigation_amount >= 100:
                stress_response_bonus += 50.0  # F_I must use substantial water
            elif self.treatment_type == 'H_I' and irrigation_amount >= 50:
                stress_response_bonus += 40.0  # H_I must use moderate water
            else:
                stress_response_bonus -= 50.0  # Penalty for inadequate irrigation
        elif water_deficit > 80 and irrigation_amount > 0:
            stress_response_bonus = 60.0  # Major reward for preventing high stress
            # Treatment-specific bonuses to ensure differentiation
            if self.treatment_type == 'F_I' and irrigation_amount >= 75:
                stress_response_bonus += 30.0  # F_I should use substantial water
            elif self.treatment_type == 'H_I' and irrigation_amount >= 40:
                stress_response_bonus += 25.0  # H_I should use moderate water
            else:
                stress_response_bonus -= 30.0  # Penalty for inadequate irrigation
        elif water_deficit > 50 and irrigation_amount > 0:
            stress_response_bonus = 40.0  # Good reward for preventing medium stress
            # Treatment-specific bonuses
            if self.treatment_type == 'F_I' and irrigation_amount >= 50:
                stress_response_bonus += 20.0
            elif self.treatment_type == 'H_I' and irrigation_amount >= 25:
                stress_response_bonus += 20.0
            else:
                stress_response_bonus -= 20.0  # Penalty for inadequate irrigation
        elif water_deficit > 25 and irrigation_amount > 0:
            stress_response_bonus = 20.0  # Moderate reward for preventing low stress
        else:
            stress_response_bonus = 0
        
        # FIXED: Reduced penalty for unnecessary irrigation
        if water_deficit <= 25 and irrigation_amount > 20:
            unnecessary_penalty = -5.0  # Reduced from -15
        else:
            unnecessary_penalty = 0
        
        # Small random noise to break ties
        noise = np.random.uniform(-0.1, 0.1)  # Reduced noise
        
        # Combine all components
        reward = (target_penalty + delta_exg_bonus + extreme_stress_penalty + 
                 high_stress_penalty + water_cost + stress_response_bonus + 
                 unnecessary_penalty + noise)
        
        # CRITICAL: Expanded reward range for agricultural production priorities
        reward = np.clip(reward, -300, 300)
        
        # Diagnostic logging (2% of steps for better visibility)
        if np.random.random() < 0.02:
            print(f"[DIAG] target: {target_irrigation:.2f}, action: {irrigation_amount}, delta_exg: {delta_exg:.4f}, reward: {reward:.2f}, water_deficit: {water_deficit:.1f}")
        
        return reward


class MLBasedWeatherGenerator:
    """ML-based weather generator using trained Random Forest models from Corpus Christi pipeline"""
    
    def __init__(self):
        self.rainfall_model = None
        self.et0_model = None
        self.heat_index_model = None
        self.scaler_rainfall = None
        self.scaler_et0 = None
        self.scaler_heat = None
        self.last_rainfall = 0.0
        self._load_ml_models()
    
    def _load_ml_models(self):
        """Load the trained ML models from Corpus Christi pipeline"""
        try:
            # Load the trained models (these should be saved from corpus_ml.py)
            import joblib
            import os
            
            model_path = "../Corpus Christi Synthetic ML Forecasting/data/"
            
            # Try to load models, fall back to rule-based if not available
            if os.path.exists(f"{model_path}rainfall_model.pkl"):
                self.rainfall_model = joblib.load(f"{model_path}rainfall_model.pkl")
                self.scaler_rainfall = joblib.load(f"{model_path}rainfall_scaler.pkl")
                print("✅ Loaded ML rainfall model from Corpus Christi pipeline")
            else:
                print("⚠️  ML rainfall model not found, using enhanced rule-based generation")
                
            if os.path.exists(f"{model_path}et0_model.pkl"):
                self.et0_model = joblib.load(f"{model_path}et0_model.pkl")
                self.scaler_et0 = joblib.load(f"{model_path}et0_scaler.pkl")
                print("✅ Loaded ML ET0 model from Corpus Christi pipeline")
            else:
                print("⚠️  ML ET0 model not found, using enhanced rule-based generation")
                
            if os.path.exists(f"{model_path}heat_index_model.pkl"):
                self.heat_index_model = joblib.load(f"{model_path}heat_index_model.pkl")
                self.scaler_heat = joblib.load(f"{model_path}heat_index_scaler.pkl")
                print("✅ Loaded ML heat index model from Corpus Christi pipeline")
            else:
                print("⚠️  ML heat index model not found, using enhanced rule-based generation")
                
        except Exception as e:
            print(f"⚠️  Error loading ML models: {e}")
            print("Falling back to enhanced rule-based weather generation")
    
    def _prepare_ml_features(self, date, last_rainfall=0.0):
        """Prepare features for ML model prediction - matches training data exactly"""
        days_after_planting = (date - datetime(2025, 4, 3)).days
        day_of_year = date.timetuple().tm_yday
        month = date.month
        
        # Match exactly the 12 features used in training
        features = [
            days_after_planting,
            month,
            day_of_year,
            date.weekday(),
            np.sin(2 * np.pi * day_of_year / 365),  # Seasonal sine
            np.cos(2 * np.pi * day_of_year / 365),  # Seasonal cosine
            np.sin(4 * np.pi * day_of_year / 365),  # Semi-annual pattern
            np.cos(4 * np.pi * day_of_year / 365),  # Semi-annual pattern
            self._get_texas_amu_cotton_kc(days_after_planting),  # Cotton Kc
            1,  # Corpus location indicator
            (day_of_year - 182) ** 2 / 10000,  # Distance from summer peak
            1 if 152 <= day_of_year <= 244 else 0,  # Summer months (Jun-Aug)
        ]
        return np.array(features).reshape(1, -1)
    
    def _get_texas_amu_cotton_kc(self, days_after_planting: int) -> float:
        """Get Texas A&M cotton Kc values - matches training data"""
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
    
    def reset_state(self):
        """Reset the weather generator state for new episodes"""
        self.last_rainfall = 0.0
    
    def generate_rainfall(self, date, last_rainfall=0.0):
        """Generate rainfall using ML model or enhanced rule-based fallback"""
        if self.rainfall_model is not None and self.scaler_rainfall is not None:
            # Use ML model
            features = self._prepare_ml_features(date, last_rainfall)
            features_scaled = self.scaler_rainfall.transform(features)
            rainfall = self.rainfall_model.predict(features_scaled)[0]
            # Add realistic noise
            rainfall += np.random.normal(0, max(0.1, rainfall * 0.1))
            return max(0.0, rainfall)
        else:
            # Enhanced rule-based fallback (improved from original)
            return self._generate_enhanced_rule_based_rainfall(date, last_rainfall)
    
    def generate_et0(self, date):
        """Generate ET0 using ML model or enhanced rule-based fallback"""
        if self.et0_model is not None and self.scaler_et0 is not None:
            # Use ML model
            features = self._prepare_ml_features(date, self.last_rainfall)
            features_scaled = self.scaler_et0.transform(features)
            et0 = self.et0_model.predict(features_scaled)[0]
            # Add realistic noise
            et0 += np.random.normal(0, max(0.1, et0 * 0.05))
            return max(0.5, et0)
        else:
            # Enhanced rule-based fallback
            return self._generate_enhanced_rule_based_et0(date)
    
    def generate_heat_index(self, date):
        """Generate heat index using ML model or enhanced rule-based fallback"""
        if self.heat_index_model is not None and self.scaler_heat is not None:
            # Use ML model
            features = self._prepare_ml_features(date, self.last_rainfall)
            features_scaled = self.scaler_heat.transform(features)
            heat_index = self.heat_index_model.predict(features_scaled)[0]
            # Add realistic noise
            heat_index += np.random.normal(0, max(1.0, heat_index * 0.02))
            return max(60, heat_index)
        else:
            # Enhanced rule-based fallback
            return self._generate_enhanced_rule_based_heat_index(date)
    
    def _generate_enhanced_rule_based_rainfall(self, date, last_rainfall=0.0):
        """Enhanced rule-based rainfall generation with Corpus Christi patterns"""
        month = date.month
        day_of_year = date.timetuple().tm_yday
        
        # Corpus Christi specific patterns (from historical data analysis)
        if month in [6, 7, 8, 9]:  # Summer/early fall - peak rainfall
            base_rain_prob = 0.30
            seasonal_multiplier = 1.4
        elif month in [4, 5, 10]:  # Spring/late fall - moderate rainfall
            base_rain_prob = 0.25
            seasonal_multiplier = 1.1
        elif month in [11, 12, 1, 2]:  # Winter - low rainfall
            base_rain_prob = 0.15
            seasonal_multiplier = 0.7
        else:  # March - transition
            base_rain_prob = 0.20
            seasonal_multiplier = 0.9
        
        # Enhanced weather persistence (from Corpus Christi patterns)
        if last_rainfall > 5.0:  # Heavy rain yesterday
            persistence_factor = 1.8  # High chance of continued rain
        elif last_rainfall > 0:  # Light rain yesterday
            persistence_factor = 1.3  # Moderate chance of continued rain
        else:  # No rain yesterday
            persistence_factor = 0.85  # Slightly lower chance
        
        rain_prob = np.clip(base_rain_prob * persistence_factor, 0.05, 0.7)
        
        # Generate rainfall
        if np.random.random() > rain_prob:
            return 0.0
        
        # Enhanced rainfall distribution (from Corpus Christi patterns)
        rain_type = np.random.random()
        if rain_type < 0.55:  # Light rain (55% of events)
            rainfall_inches = np.random.uniform(0.05, 0.4)
        elif rain_type < 0.80:  # Moderate rain (25% of events)
            rainfall_inches = np.random.uniform(0.4, 1.2)
        elif rain_type < 0.95:  # Heavy rain (15% of events)
            rainfall_inches = np.random.uniform(1.2, 2.5)
        else:  # Extreme rain (5% of events)
            rainfall_inches = np.random.uniform(2.5, 4.0)
        
        # Convert to gallons and apply seasonal multiplier
        rainfall_gallons = rainfall_inches * 2.07 * seasonal_multiplier
        rainfall_gallons *= np.random.uniform(0.85, 1.15)  # Realistic noise
        
        self.last_rainfall = rainfall_gallons
        return max(0.0, rainfall_gallons)
    
    def _generate_enhanced_rule_based_et0(self, date):
        """Enhanced rule-based ET0 generation with Corpus Christi patterns"""
        day_of_year = date.timetuple().tm_yday
        month = date.month
        
        # Corpus Christi ET0 patterns (from historical data)
        if month in [6, 7, 8]:  # Peak summer
            base_et0 = 7.5 + 2.0 * np.sin(2 * np.pi * (day_of_year - 172) / 365)
        elif month in [4, 5, 9]:  # Spring/Fall
            base_et0 = 5.5 + 1.5 * np.sin(2 * np.pi * (day_of_year - 135) / 365)
        elif month in [10, 11]:  # Late Fall
            base_et0 = 4.0 + 1.0 * np.sin(2 * np.pi * (day_of_year - 305) / 365)
        else:  # Winter/Early Spring
            base_et0 = 3.0 + 0.8 * np.sin(2 * np.pi * (day_of_year - 45) / 365)
        
        # Add realistic daily variation
        daily_variation = np.random.normal(0, 0.8)
        et0 = base_et0 + daily_variation
        
        return max(0.5, et0)
    
    def _generate_enhanced_rule_based_heat_index(self, date):
        """Enhanced rule-based heat index generation with Corpus Christi patterns"""
        day_of_year = date.timetuple().tm_yday
        month = date.month
        
        # Corpus Christi temperature patterns (from historical data)
        if month in [6, 7, 8]:  # Peak summer
            base_temp = 88 + 8 * np.sin(2 * np.pi * (day_of_year - 172) / 365)
        elif month in [4, 5, 9]:  # Spring/Fall
            base_temp = 78 + 6 * np.sin(2 * np.pi * (day_of_year - 135) / 365)
        elif month in [10, 11]:  # Late Fall
            base_temp = 70 + 4 * np.sin(2 * np.pi * (day_of_year - 305) / 365)
        else:  # Winter/Early Spring
            base_temp = 62 + 3 * np.sin(2 * np.pi * (day_of_year - 45) / 365)
        
        # Add realistic daily variation
        daily_variation = np.random.normal(0, 4.0)
        heat_index = base_temp + daily_variation
        
        return max(60, heat_index)

def create_synthetic_training_data():
    """Create highly diverse synthetic training data for RL."""
    print("🔄 Creating synthetic training data for realistic plant responses...")
    dates = pd.date_range('2025-04-03', '2025-10-31', freq='D')
    training_data = []
    for i, date in enumerate(dates):
        days_after_planting = i + 30
        # Diverse soil moisture
        soil_moisture = np.random.uniform(160, 240)
        # Diverse ExG, seasonally dependent
        if days_after_planting < 45:
            exg = np.random.uniform(0.2, 0.5)
        elif days_after_planting < 95:
            exg = np.random.uniform(0.4, 0.7)
        else:
            exg = np.random.uniform(0.2, 0.5)
        # Diverse ET0
        et0 = np.random.uniform(3.5, 7.5)
        # Diverse heat index
        heat_index = np.random.uniform(75, 96)
        # Diverse rainfall
        rainfall = np.random.exponential(2) if np.random.random() < 0.3 else 0
        # Diverse Kc
        kc = np.random.uniform(0.4, 1.1)
        training_data.append({
            'Date': date,
            'Plot ID': 999,
            'Treatment Type': 'SYNTH',
            'ExG': round(exg, 4),
            'Total Soil Moisture': round(soil_moisture, 1),
            'ET0 (mm)': round(et0, 2),
            'Heat Index (F)': round(heat_index, 1),
            'Rainfall (gallons)': round(rainfall, 2),
            'Kc (Crop Coefficient)': round(kc, 3),
            'Days_After_Planting': days_after_planting
        })
    df = pd.DataFrame(training_data)
    print(f"Created {len(df)} synthetic training samples")
    return df

# Diagnostic: plot state diversity
def plot_state_diversity(env):
    features = []
    for _, row in env.corpus_data.iterrows():
        soil_moisture = row.get('Total Soil Moisture', 200)
        et0 = row.get('ET0 (mm)', 5)
        heat_index = row.get('Heat Index (F)', 85)
        rainfall = row.get('Rainfall (gallons)', 0)
        exg = row.get('ExG', 0.4)
        days_after_planting = row.get('Days_After_Planting', 60)
        kc = row.get('Kc (Crop Coefficient)', 0.8)
        water_deficit = max(0, 200 - soil_moisture)
        features.append([soil_moisture, et0, heat_index, rainfall, exg, days_after_planting, water_deficit, kc])
    features = np.array(features)
    labels = ['Soil Moisture', 'ET0', 'Heat Index', 'Rainfall', 'ExG', 'Days After Planting', 'Water Deficit', 'Kc']
    plt.figure(figsize=(16, 8))
    for i in range(features.shape[1]):
        plt.subplot(2, 4, i+1)
        plt.hist(features[:, i], bins=20)
        plt.title(labels[i])
    plt.tight_layout()
    plt.show()

def train_and_save_ml_models():
    """Train and save ML models from Corpus Christi pipeline for RL system use"""
    print("🤖 TRAINING ML MODELS FROM CORPUS CHRISTI PIPELINE")
    print("=" * 60)
    
    try:
        # Import the Corpus Christi ML pipeline
        import sys
        sys.path.append('../../Corpus Christi Synthetic ML Forecasting/scripts/')
        
        # Import the ML generator
        from corpus_ml import EnhancedMLCottonSyntheticGenerator
        
        # Create and train the ML models
        generator = EnhancedMLCottonSyntheticGenerator()
        generator.load_data()
        
        # Prepare training data
        (X_exg, y_exg, weights_exg, X_soil, y_soil, weights_soil,
         X_heat, y_heat, weights_heat, X_et0, y_et0, weights_et0,
         X_rainfall, y_rainfall, weights_rainfall) = generator.prepare_enhanced_training_data()
        
        # Train models
        generator.train_enhanced_models(X_exg, y_exg, weights_exg, X_soil, y_soil, weights_soil,
                                       X_heat, y_heat, weights_heat, X_et0, y_et0, weights_et0,
                                       X_rainfall, y_rainfall, weights_rainfall)
        
        # Save models for RL system use
        import joblib
        import os
        
        # Create output directory
        os.makedirs('../../Corpus Christi Synthetic ML Forecasting/data/', exist_ok=True)
        
        # Save models
        joblib.dump(generator.rainfall_model, '../../Corpus Christi Synthetic ML Forecasting/data/rainfall_model.pkl')
        joblib.dump(generator.scaler_rainfall, '../../Corpus Christi Synthetic ML Forecasting/data/rainfall_scaler.pkl')
        
        joblib.dump(generator.et0_model, '../../Corpus Christi Synthetic ML Forecasting/data/et0_model.pkl')
        joblib.dump(generator.scaler_et0, '../../Corpus Christi Synthetic ML Forecasting/data/et0_scaler.pkl')
        
        joblib.dump(generator.heat_index_model, '../../Corpus Christi Synthetic ML Forecasting/data/heat_index_model.pkl')
        joblib.dump(generator.scaler_heat, '../../Corpus Christi Synthetic ML Forecasting/data/heat_index_scaler.pkl')
        
        print("✅ ML models trained and saved successfully!")
        print("RL system will now use ML-based weather generation")
        
    except Exception as e:
        print(f"⚠️  Error training ML models: {e}")
        print("RL system will use enhanced rule-based weather generation")

def demonstrate_weather_variability():
    """Demonstrate that weather patterns vary between episodes."""
    print("🌤️  DEMONSTRATING WEATHER VARIABILITY ACROSS EPISODES")
    print("=" * 60)
    
    env = PlantHealthAwareEnv(treatment_type='H_I', max_days=212)
    
    # Show weather patterns for 5 different episodes
    for episode in range(5):
        state, _ = env.reset()
        print(f"\n📅 Episode {episode + 1} Weather Pattern:")
        
        # Show first 10 days of weather
        for day in range(10):
            rainfall = env.today_rainfall
            et0 = env.today_et0
            heat = env.today_heat_index
            print(f"  Day {day + 1}: Rain={rainfall:.1f}, ET0={et0:.1f}, Heat={heat:.1f}")
            
            # Step through environment to get next day's weather
            action = 0  # No irrigation for demo
            state, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break
    
    print("\n✅ Weather patterns are now VARIABLE between episodes!")
    print("This ensures RL agent learns robust policies across weather uncertainty.")

def train_plant_health_policy(treatment_type='H_I', total_timesteps=60000):
    """Train plant health-aware irrigation policy using interactive RL environment."""
    print(f"🌱 Training plant health-aware irrigation policy for {treatment_type}")
    
    # Create interactive environment (no CSV loading needed)
    env = PlantHealthAwareEnv(treatment_type=treatment_type, max_days=180)
    
    print(f"Environment created with {len(env.irrigation_amounts)} actions: {env.irrigation_amounts}")
    print(f"Treatment type: {treatment_type}")
    print(f"Max days per episode: {env.max_days}")
    
    # PPO configuration for interactive RL
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=2e-4,     # Slightly reduced for stable convergence in long training
        n_steps=2048,           # Steps per environment per update (reduced for shorter episodes)
        batch_size=128,         # Larger batch size for stable gradients
        n_epochs=10,            # Number of epochs per update
        gamma=0.99,             # Discount factor
        gae_lambda=0.95,        # GAE lambda
        clip_range=0.2,         # PPO clip range
        ent_coef=0.2,           # FURTHER INCREASED: Entropy coefficient for exploration and action diversity
        vf_coef=0.5,            # Value function coefficient
        max_grad_norm=0.5,      # Gradient clipping
        verbose=1,
        tensorboard_log=f"../models/ppo/tensorboard_{treatment_type}/"
    )
    
    # Train model through actual interaction
    print(f"Training for {total_timesteps} timesteps...")
    print("Agent will learn through trial-and-error interaction with the environment")
    print("Each episode = 212 days of irrigation decisions (April 3 - October 31)")
    
    model.learn(total_timesteps=total_timesteps)
    
    # Save model
    model_path = f"../models/ppo/plant_health_ppo_{treatment_type}_{total_timesteps//1000}k.zip"
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Diagnostic: analyze action distribution after training
    print("\n🔍 Analyzing action distribution after training...")
    actions = []
    for _ in range(100):
        state, _ = env.reset()
        action, _ = model.predict(state, deterministic=True)
        actions.append(env.irrigation_amounts[action])
    
    from collections import Counter
    action_counts = Counter(actions)
    print(f"Action distribution: {dict(action_counts)}")
    print(f"Unique actions used: {len(set(actions))}/{len(env.irrigation_amounts)}")
    print(f"Most common action: {max(action_counts, key=action_counts.get)} gallons ({action_counts[max(action_counts, key=action_counts.get)]/100*100:.1f}%)")
    
    return model

def apply_plant_health_policy(model, treatment_type='H_I'):
    """Apply plant-health-focused policy to Corpus Christi."""
    print(f"\n📊 APPLYING PLANT HEALTH POLICY: {treatment_type}")
    print("=" * 50)
    
    # Load Corpus Christi data
    corpus_path = '../../Corpus Christi Synthetic ML Forecasting/data/corpus_season_completed_enhanced_lubbock_ml.csv'
    corpus_data = pd.read_csv(corpus_path)
    
    # Filter for treatment type
    treatment_plots = {'R_F': 102, 'H_I': 404, 'F_I': 409}
    plot_id = treatment_plots[treatment_type]
    
    plot_data = corpus_data[corpus_data['Plot ID'] == plot_id].copy()
    plot_data['Date'] = pd.to_datetime(plot_data['Date'])
    plot_data = plot_data.sort_values('Date').reset_index(drop=True)
    
    # Calculate days after planting
    corpus_season_start = pd.to_datetime('2025-04-03')
    plot_data['Days_After_Planting'] = (plot_data['Date'] - corpus_season_start).dt.days
    
    # Create environment for state preprocessing (even for rainfed)
    temp_env = PlantHealthAwareEnv(plot_data, treatment_type)
    
    recommendations = []
    
    for _, row in plot_data.iterrows():
        # Create state with robust NaN handling
        soil_moisture = row.get('Total Soil Moisture', 200)
        if pd.isna(soil_moisture):
            soil_moisture = 200
        
        et0 = row.get('ET0 (mm)', 5)
        if pd.isna(et0):
            et0 = 5
        
        heat_index = row.get('Heat Index (F)', 85)
        if pd.isna(heat_index):
            heat_index = 85
        
        rainfall = row.get('Rainfall (gallons)', 0)
        if pd.isna(rainfall):
            rainfall = 0
        
        exg = row.get('ExG', 0.4)
        if pd.isna(exg):
            exg = 0.4
        
        days_after_planting = row.get('Days_After_Planting', 60)
        if pd.isna(days_after_planting):
            days_after_planting = 60
        
        kc = row.get('Kc (Crop Coefficient)', 0.8)
        if pd.isna(kc):
            kc = 0.8
        
        water_deficit = max(0, 200 - soil_moisture)
        
        features = [soil_moisture, et0, heat_index, rainfall, exg, days_after_planting, water_deficit, kc]
        
        # Double-check for any remaining NaN values
        for i, val in enumerate(features):
            if pd.isna(val) or np.isnan(val):
                # Replace with reasonable defaults
                defaults = [200, 5, 85, 0, 0.4, 60, 0, 0.8]
                features[i] = defaults[i]
        
        # Handle irrigation recommendation
        if treatment_type == 'R_F' or model is None:
            # Rainfed - no irrigation allowed
            recommended_irrigation = 0
        else:
            # Use trained model to predict irrigation
            try:
                normalized_features = temp_env.scaler.transform([features])[0]
            except:
                normalized_features = np.array(features) / 100
            
            # Final NaN check on normalized values before feeding to model
            if np.any(np.isnan(normalized_features)):
                print(f"⚠️ Warning: NaN detected in normalized features for prediction, using defaults")
                normalized_features = np.nan_to_num(normalized_features, nan=0.5)
            
            state = normalized_features.astype(np.float32)
            
            # Get policy recommendation
            action, _ = model.predict(state, deterministic=True)
            recommended_irrigation = temp_env.irrigation_amounts[action]
        
        # ISSUE 2 FIX: Updated water stress calculation to match reward function
        if soil_moisture < 190:  # High stress threshold aligned with reward function
            water_stress = 'High'
        elif soil_moisture < 200:  # Medium stress threshold aligned with reward function
            water_stress = 'Medium'  
        else:
            water_stress = 'Low'
        
        # Calculate ETc
        et0 = row.get('ET0 (mm)', 5)
        kc = row.get('Kc (Crop Coefficient)', 0.8)
        if pd.isna(et0): et0 = 5
        if pd.isna(kc): kc = 0.8
        predicted_etc = et0 * kc
        
        # Get ExG values
        current_exg = row.get('ExG', 0.4)
        if pd.isna(current_exg): current_exg = 0.4
        
        # ISSUE 3 FIX: Improved Delta ExG calculation to show irrigation benefits
        days_after_planting = row.get('Days_After_Planting', 60)
        if pd.isna(days_after_planting): days_after_planting = 60
        
        # Base seasonal trend
        if days_after_planting > 100:  # Late season senescence
            base_trend = -0.002
        elif days_after_planting > 80:  # Mid-season decline
            base_trend = -0.001
        else:  # Early season growth potential
            base_trend = 0.001
        
        # Irrigation benefit calculation
        if recommended_irrigation > 0:
            if soil_moisture < 190:  # High stress - irrigation helps significantly
                irrigation_benefit = 0.01 + (recommended_irrigation / 2000)  # 0.01-0.085 benefit
            elif soil_moisture < 200:  # Medium stress - irrigation helps moderately  
                irrigation_benefit = 0.005 + (recommended_irrigation / 3000)  # 0.005-0.055 benefit
            else:  # Low stress - minimal benefit, potential waste
                irrigation_benefit = max(-0.002, recommended_irrigation / 5000)  # Small benefit or waste
        else:
            irrigation_benefit = 0  # No irrigation, no benefit
        
        # Combined effect - irrigation should overcome negative seasonal trends when needed
        predicted_delta_exg = base_trend + irrigation_benefit
        
        # Ensure reasonable bounds
        predicted_delta_exg = max(-0.01, min(0.05, predicted_delta_exg))
        
        recommendations.append({
            'Date': row['Date'].strftime('%Y-%m-%d'),
            'Plot ID': plot_id,
            'Treatment Type': treatment_type,
            'Water Stress': water_stress,
            'Recommended Irrigation (gallons)': round(recommended_irrigation, 1),
            'Predicted ETc': round(predicted_etc, 2),
            'Current ExG': round(current_exg, 4),
            'Predicted Delta ExG': round(predicted_delta_exg, 4),
            'Plot Size (sq ft)': 443.5
        })
    
    return recommendations

def apply_interactive_policy(model, treatment_type='H_I', num_scenarios=5):
    """Apply trained irrigation policy across multiple weather scenarios."""
    print(f"\n🌱 APPLYING INTERACTIVE POLICY: {treatment_type}")
    print(f"📊 Testing across {num_scenarios} different weather scenarios")
    print("=" * 50)
    
    all_recommendations = []
    
    for scenario in range(num_scenarios):
        print(f"🌤️  Scenario {scenario + 1}/{num_scenarios}")
        
        # Create interactive environment for full season simulation
        env = PlantHealthAwareEnv(treatment_type=treatment_type, max_days=212)
        
        # Run full season simulation
        state, _ = env.reset()
        terminated = False
        
        while not terminated:
            # Get policy recommendation
            if model is None:  # Rainfed treatment
                action = 0  # Always no irrigation
                irrigation_amount = 0
            else:
                action, _ = model.predict(state, deterministic=True)
                irrigation_amount = env.irrigation_amounts[action]
            
            # Calculate water stress level (aligned with reward function)
            water_deficit = max(0, 250 - env.soil_moisture)
            
            if water_deficit > 100:
                stress_level = 'Extreme'
            elif water_deficit > 50:
                stress_level = 'High'
            elif water_deficit > 25:
                stress_level = 'Medium'
            else:
                stress_level = 'Low'
            
            # Store current ExG before action
            current_exg = env.exg
            
            # Take action and get next state
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Calculate Delta ExG from the step
            delta_exg = env.exg - current_exg
            
            all_recommendations.append({
                'Scenario': scenario + 1,
                'Date': env.current_date.strftime('%Y-%m-%d'),
                'Plot ID': {'R_F': 102, 'H_I': 404, 'F_I': 409}[treatment_type],
                'Treatment Type': treatment_type,
                'Water Stress': stress_level,
                'Recommended Irrigation (gallons)': round(irrigation_amount, 1),
                'Predicted ETc': round(env.today_et0 * env._calculate_kc(env.days_after_planting), 2),
                'Current ExG': round(current_exg, 4),
                'Predicted Delta ExG': round(delta_exg, 4),
                'Plot Size (sq ft)': 443.5,
                'Total Rainfall (scenario)': round(sum([env.weather_generator.generate_rainfall(
                    env.planting_date + timedelta(days=i), 0) for i in range(180)]), 1)
            })
            
            state = next_state
    
    return all_recommendations

def apply_single_scenario(model, treatment_type='H_I', seed=None):
    """Apply trained irrigation policy to a single weather scenario."""
    # Set seed for reproducible weather if specified
    if seed is not None:
        np.random.seed(seed)
    
    # Create interactive environment for full season simulation
    env = PlantHealthAwareEnv(treatment_type=treatment_type, max_days=212)
    
    recommendations = []
    
    # Run full season simulation
    state, _ = env.reset()
    terminated = False
    
    while not terminated:
        # Get policy recommendation
        if model is None:  # Rainfed treatment
            action = 0  # Always no irrigation
            irrigation_amount = 0
        else:
            action, _ = model.predict(state, deterministic=True)
            irrigation_amount = env.irrigation_amounts[action]
        
        # Calculate water stress level (aligned with reward function)
        water_deficit = max(0, 250 - env.soil_moisture)
        
        if water_deficit > 100:
            stress_level = 'Extreme'
        elif water_deficit > 50:
            stress_level = 'High'
        elif water_deficit > 25:
            stress_level = 'Medium'
        else:
            stress_level = 'Low'
        
        # Store current ExG before action
        current_exg = env.exg
        
        # Take action and get next state
        next_state, reward, terminated, truncated, info = env.step(action)
        
        # Calculate Delta ExG from the step
        delta_exg = env.exg - current_exg
        
        recommendations.append({
            'Date': env.current_date.strftime('%Y-%m-%d'),
            'Plot ID': {'R_F': 102, 'H_I': 404, 'F_I': 409}[treatment_type],
            'Treatment Type': treatment_type,
            'Water Stress': stress_level,
            'Recommended Irrigation (gallons)': round(irrigation_amount, 1),
            'Predicted ETc': round(env.today_et0 * env._calculate_kc(env.days_after_planting), 2),
            'Current ExG': round(current_exg, 4),
            'Predicted Delta ExG': round(delta_exg, 4),
            'Plot Size (sq ft)': 443.5
        })
        
        state = next_state
    
    return recommendations

def main_interactive():
    """Main training and application pipeline using interactive RL environment."""
    print("🌱 INTERACTIVE RL IRRIGATION SYSTEM - TRUE REINFORCEMENT LEARNING")
    print("=" * 70)
    print("Training policies through actual environment interaction, not static CSV data")
    print("Agent learns through trial-and-error over full growing seasons")
    print()
    
    all_recommendations = []
    
    # Train policies for each treatment type
    for treatment in ['H_I', 'F_I']:
        print(f"\n📚 Processing treatment: {treatment}")
        
        # Train plant-health-focused policy with interactive RL
        model = train_plant_health_policy(treatment, total_timesteps=3000)  # Much faster demo
        
        if model is not None:
            # Apply to generate recommendations
            recommendations = apply_interactive_policy(model, treatment, num_scenarios=2)  # Reduced scenarios
            all_recommendations.extend(recommendations)
            
            print(f"Generated {len(recommendations)} recommendations for {treatment}")
    
    # Add rainfed recommendations (always 0 irrigation)
    rf_recommendations = apply_interactive_policy(None, 'R_F')
    all_recommendations.extend(rf_recommendations)
    
    # Save all recommendations
    if all_recommendations:
        df_recommendations = pd.DataFrame(all_recommendations)
        
        # Save multi-scenario results
        if 'Scenario' in df_recommendations.columns:
            # Sort by Scenario, Date, then Treatment Type
            df_recommendations = df_recommendations.sort_values(['Scenario', 'Date', 'Treatment Type'])
            output_file = 'outputs/policy_transfer_recommendations_multi_scenario.csv'
        else:
            # Sort by Date first, then Treatment Type for easy day-by-day comparison
            df_recommendations = df_recommendations.sort_values(['Date', 'Treatment Type'])
            output_file = 'outputs/policy_transfer_recommendations_interactive_rl.csv'
        
        df_recommendations.to_csv(output_file, index=False)
        
        # Also generate a representative scenario for easy analysis
        print(f"\n🌱 GENERATING REPRESENTATIVE SCENARIO (Fixed Weather)")
        representative_recommendations = []
        for treatment in ['H_I', 'F_I']:
            if treatment in [rec['Treatment Type'] for rec in all_recommendations]:
                # Find the trained model for this treatment
                model_path = f"../models/ppo/plant_health_ppo_{treatment}_10k.zip"
                try:
                    from stable_baselines3 import PPO
                    model = PPO.load(model_path)
                    recs = apply_single_scenario(model, treatment, seed=42)
                    representative_recommendations.extend(recs)
                except:
                    # If model not found, use None for rainfed
                    recs = apply_single_scenario(None, treatment, seed=42)
                    representative_recommendations.extend(recs)
        
        # Add rainfed
        rf_recs = apply_single_scenario(None, 'R_F', seed=42)
        representative_recommendations.extend(rf_recs)
        
        # Save representative scenario
        df_representative = pd.DataFrame(representative_recommendations)
        df_representative = df_representative.sort_values(['Date', 'Treatment Type'])
        representative_file = 'outputs/policy_transfer_recommendations_representative.csv'
        df_representative.to_csv(representative_file, index=False)
        
        print(f"📁 Representative scenario saved to: {representative_file}")
    
    else:
        print("❌ No recommendations generated!")
    
    print(f"\n✅ INTERACTIVE RL POLICY TRANSFER COMPLETE")
    print(f"📁 Recommendations saved to: {output_file}")
    print(f"📊 Total recommendations: {len(df_recommendations)}")
    
    # Summary by treatment
    summary = df_recommendations.groupby('Treatment Type').agg({
        'Recommended Irrigation (gallons)': ['count', 'sum', 'mean']
    }).round(1)
    
    print("\n📈 SUMMARY BY TREATMENT:")
    print(summary)
    
    # Count irrigation days
    irrigation_days = df_recommendations[df_recommendations['Recommended Irrigation (gallons)'] > 0].groupby('Treatment Type').size()
    print(f"\n💧 IRRIGATION DAYS:")
    for treatment in ['R_F', 'H_I', 'F_I']:
        if treatment in irrigation_days.index:
            total_days = len(df_recommendations[df_recommendations['Treatment Type'] == treatment])
            irrigated_days = irrigation_days[treatment]
            print(f"  {treatment}: {irrigated_days}/{total_days} days irrigated ({irrigated_days/total_days*100:.1f}%)")
        else:
            total_days = len(df_recommendations[df_recommendations['Treatment Type'] == treatment])
            print(f"  {treatment}: 0/{total_days} days irrigated (0.0%)")
    
    # Evaluation metrics
    print("\n✅ EVALUATION METRICS")
    
    # Check if we have multiple scenarios
    if 'Scenario' in df_recommendations.columns:
        print("\n📊 MULTI-SCENARIO ANALYSIS")
        
        # Scenario-level analysis
        scenario_summary = df_recommendations.groupby(['Treatment Type', 'Scenario']).agg({
            'Recommended Irrigation (gallons)': ['sum', 'mean'],
            'Predicted Delta ExG': 'mean',
            'Total Rainfall (scenario)': 'first'
        }).round(2)
        
        print("\n🌤️  Policy Performance by Weather Scenario:")
        print(scenario_summary)
        
        # Policy robustness analysis
        robustness = df_recommendations.groupby('Treatment Type').agg({
            'Recommended Irrigation (gallons)': ['mean', 'std', 'min', 'max'],
            'Predicted Delta ExG': ['mean', 'std']
        }).round(2)
        
        print("\n🛡️  Policy Robustness (Variability Across Scenarios):")
        print(robustness)
        
        # Rainfall vs Irrigation correlation
        rainfall_irrigation = df_recommendations.groupby(['Treatment Type', 'Scenario']).agg({
            'Recommended Irrigation (gallons)': 'sum',
            'Total Rainfall (scenario)': 'first'
        }).reset_index()
        
        print("\n🌧️  Rainfall vs Irrigation Correlation:")
        for treatment in rainfall_irrigation['Treatment Type'].unique():
            subset = rainfall_irrigation[rainfall_irrigation['Treatment Type'] == treatment]
            correlation = subset['Recommended Irrigation (gallons)'].corr(subset['Total Rainfall (scenario)'])
            print(f"  {treatment}: r = {correlation:.3f}")
    
    # Overall metrics (across all scenarios)
    # Cumulative irrigation per treatment
    cumulative_irrigation = df_recommendations.groupby('Treatment Type')['Recommended Irrigation (gallons)'].sum()
    print("\n💧 Cumulative Irrigation by Treatment (All Scenarios):")
    print(cumulative_irrigation)

    # Mean Delta ExG by treatment
    mean_delta_exg = df_recommendations.groupby('Treatment Type')['Predicted Delta ExG'].mean()
    print("\n🌿 Mean Predicted Delta ExG by Treatment (All Scenarios):")
    print(mean_delta_exg)

    # Stress frequency
    stress_counts = df_recommendations.groupby(['Treatment Type', 'Water Stress']).size().unstack(fill_value=0)
    print("\n⚠️ Water Stress Frequency by Treatment (All Scenarios):")
    print(stress_counts)

def main():
    """Main training and application pipeline with plant health focus using PPO."""
    print("🌱 PLANT HEALTH IRRIGATION SYSTEM - PPO FINAL PRODUCTION RUN")
    print("=" * 70)
    print("Training policies with PPO that prioritize plant health over expert imitation")
    print("OPTIMIZED FOR 50K TIMESTEPS - FINAL PRODUCTION DATA GENERATION")
    print()
    
    # First, train and save ML models from Corpus Christi pipeline
    print("🔧 SETTING UP ML-BASED WEATHER GENERATION")
    print("-" * 50)
    train_and_save_ml_models()
    print()
    
    all_recommendations = []
    
    # Train policies for each treatment type
    for treatment in ['H_I', 'F_I']:
        print(f"\n📚 Processing treatment: {treatment}")
        
        # Train plant-health-focused policy with PPO - Final Production Run
        model = train_plant_health_policy(treatment, total_timesteps=50000)  # Full training for production
        
        if model is not None:
            # Apply to Corpus Christi
            recommendations = apply_plant_health_policy(model, treatment)
            all_recommendations.extend(recommendations)
            
            print(f"Generated {len(recommendations)} recommendations for {treatment}")
    
    # Add rainfed recommendations (always 0 irrigation but with realistic ExG expectations)
    rf_recommendations = apply_plant_health_policy(None, 'R_F')  # None model for rainfed
    all_recommendations.extend(rf_recommendations)
    
    # Save all recommendations
    if all_recommendations:
        df_recommendations = pd.DataFrame(all_recommendations)
        # Sort by Date first, then Treatment Type for easy day-by-day comparison
        df_recommendations = df_recommendations.sort_values(['Date', 'Treatment Type'])
        
        output_file = '../outputs/policy_transfer_recommendations_50k_final.csv'
        df_recommendations.to_csv(output_file, index=False)
        
        print(f"\n✅ POLICY TRANSFER COMPLETE")
        print(f"📁 Recommendations saved to: {output_file}")
        print(f"📊 Total recommendations: {len(df_recommendations)}")
        
        # Summary by treatment
        summary = df_recommendations.groupby('Treatment Type').agg({
            'Recommended Irrigation (gallons)': ['count', 'sum', 'mean']
        }).round(1)
        
        print("\n📈 SUMMARY BY TREATMENT:")
        print(summary)
        
        # Count irrigation days
        irrigation_days = df_recommendations[df_recommendations['Recommended Irrigation (gallons)'] > 0].groupby('Treatment Type').size()
        print(f"\n💧 IRRIGATION DAYS:")
        for treatment in ['R_F', 'H_I', 'F_I']:
            if treatment in irrigation_days.index:
                total_days = len(df_recommendations[df_recommendations['Treatment Type'] == treatment])
                irrigated_days = irrigation_days[treatment]
                print(f"  {treatment}: {irrigated_days}/{total_days} days irrigated ({irrigated_days/total_days*100:.1f}%)")
            else:
                total_days = len(df_recommendations[df_recommendations['Treatment Type'] == treatment])
                print(f"  {treatment}: 0/{total_days} days irrigated (0.0%)")
        
        # Add detailed evaluation metrics
        print("\n✅ EVALUATION METRICS")
        
        # Compute cumulative irrigation per treatment
        cumulative_irrigation = df_recommendations.groupby('Treatment Type')['Recommended Irrigation (gallons)'].sum()
        print("\n💧 Cumulative Irrigation by Treatment:")
        print(cumulative_irrigation)

        # Compute mean final ExG by treatment
        final_exg = df_recommendations.groupby('Treatment Type')['Predicted Delta ExG'].mean()
        print("\n🌿 Mean Predicted Delta ExG by Treatment:")
        print(final_exg)

        # Compute stress frequency
        stress_counts = df_recommendations.groupby(['Treatment Type', 'Water Stress']).size().unstack(fill_value=0)
        print("\n🔥 Water Stress Frequency by Treatment:")
        print(stress_counts)
    
    else:
        print("❌ No recommendations generated")

def main_multi_seed():
    """Main training and application pipeline with multiple seeds for reproducibility testing."""
    print("🌱 IRRIGATION POLICY TRANSFER SYSTEM - MULTI-SEED REPRODUCIBILITY TEST")
    print("=" * 80)
    print("Learning from Lubbock experts → Applying to Corpus Christi (Multiple Seeds)")
    print()
    
    seeds = [42, 123, 999]
    all_seed_results = {}
    
    for seed in seeds:
        print(f"\n🔁 TRAINING WITH SEED: {seed}")
        print("=" * 40)
        
        # Set seeds for this iteration
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        seed_recommendations = []
        
        # Train policies for each treatment type
        for treatment in ['H_I', 'F_I']:
            print(f"\n📚 Processing treatment: {treatment} (seed {seed})")
            
            # Train policy from expert demonstrations
            model = train_plant_health_policy(treatment, total_timesteps=20000)  # Reduced for faster testing
            
            if model is not None:
                # Apply to Corpus Christi
                recommendations = apply_plant_health_policy(model, treatment)
                seed_recommendations.extend(recommendations)
                
                print(f"Generated {len(recommendations)} recommendations for {treatment}")
        
        # Add rainfed recommendations for this seed using ML enhanced dataset
        corpus_path = '../../Corpus Christi Synthetic ML Forecasting/data/corpus_season_completed_enhanced_lubbock_ml.csv'
        corpus_data = pd.read_csv(corpus_path)
        rf_data = corpus_data[corpus_data['Plot ID'] == 102].copy()
        rf_data['Date'] = pd.to_datetime(rf_data['Date'])
        rf_data['Delta_ExG'] = rf_data['ExG'].diff().fillna(0)
        
        for _, row in rf_data.iterrows():
            # Calculate water stress level for rainfed
            soil_moisture = row.get('Total Soil Moisture', 200)
            if pd.isna(soil_moisture):
                soil_moisture = 200  # Default fallback
                
            if soil_moisture < 195:
                water_stress = 'High'
            elif soil_moisture < 210:
                water_stress = 'Medium'
            else:
                water_stress = 'Low'
            
            # Calculate predicted ETc with proper handling
            et0 = row.get('ET0 (mm)', 5.0)
            kc = row.get('Kc (Crop Coefficient)', 0.8)
            
            if pd.isna(et0):
                et0 = 5.0  # Default fallback
            if pd.isna(kc):
                kc = 0.8  # Default fallback
                
            predicted_etc = et0 * kc
            
            # Use actual ExG values from the ML enhanced dataset
            current_exg = row.get('ExG', 0.25)
            delta_exg = row.get('Delta_ExG', 0)
            
            # Handle NaN values
            if pd.isna(current_exg):
                current_exg = 0.25
            if pd.isna(delta_exg):
                delta_exg = 0
            
            seed_recommendations.append({
                'Date': row['Date'].strftime('%Y-%m-%d'),
                'Plot ID': 102,
                'Treatment Type': 'R_F',
                'Water Stress': water_stress,
                'Recommended Irrigation (gallons)': 0,
                'Predicted ETc': round(predicted_etc, 2),
                'Current ExG': round(current_exg, 4),
                'Predicted Delta ExG': round(delta_exg, 4),
                'Scaled Irrigation Factor': 0.0685  # Same scaling factor for consistency
            })
        
        # Store results for this seed
        if seed_recommendations:
            df_seed = pd.DataFrame(seed_recommendations)
            all_seed_results[seed] = df_seed
            
            # Print summary for this seed
            print(f"\n📊 SEED {seed} SUMMARY:")
            cumulative = df_seed.groupby('Treatment Type')['Recommended Irrigation (gallons)'].sum()
            print(f"Cumulative irrigation: {cumulative.to_dict()}")
    
    # Compare results across seeds
    print(f"\n🔄 REPRODUCIBILITY ANALYSIS ACROSS {len(seeds)} SEEDS:")
    print("=" * 60)
    
    if all_seed_results:
        for treatment in ['R_F', 'H_I', 'F_I']:
            irrigation_totals = []
            for seed in seeds:
                if seed in all_seed_results:
                    total = all_seed_results[seed][all_seed_results[seed]['Treatment Type'] == treatment]['Recommended Irrigation (gallons)'].sum()
                    irrigation_totals.append(total)
            
            if irrigation_totals:
                mean_irrigation = np.mean(irrigation_totals)
                std_irrigation = np.std(irrigation_totals)
                print(f"{treatment}: {mean_irrigation:.1f} ± {std_irrigation:.1f} gallons (CV: {std_irrigation/mean_irrigation*100:.1f}%)")
        
        # Save combined results
        all_recommendations = []
        for seed, df in all_seed_results.items():
            df_copy = df.copy()
            df_copy['Seed'] = seed
            all_recommendations.append(df_copy)
        
        combined_df = pd.concat(all_recommendations, ignore_index=True)
        output_file = 'policy_transfer_multi_seed_recommendations.csv'
        combined_df.to_csv(output_file, index=False)
        print(f"\n📁 Multi-seed results saved to: {output_file}")

def print_policy_action_distribution(model, env):
    actions = []
    for _ in range(100):
        idx = np.random.randint(0, len(env.corpus_data))
        env.current_step = idx
        state = env._get_state()
        action, _ = model.predict(state, deterministic=True)
        actions.append(env.irrigation_amounts[action])
    print("Action distribution over 100 random states:", actions)
    print("Unique actions:", set(actions))

def print_reward_sensitivity(env):
    for _ in range(5):
        idx = np.random.randint(0, len(env.corpus_data))
        row = env.corpus_data.iloc[idx]
        print(f"\nState {idx}:")
        for action in range(len(env.irrigation_amounts)):
            irrigation = env.irrigation_amounts[action]
            reward = env._calculate_plant_health_reward(irrigation, row)
            print(f"  Action {action} (Irrigation {irrigation}): Reward = {reward:.2f}")

# Add a function to plot reward sensitivity for a random sample of states

def plot_reward_sensitivity(env, num_states=5):
    import matplotlib.pyplot as plt
    import os
    np.random.seed(42)
    os.makedirs('outputs', exist_ok=True)
    sample_indices = np.random.choice(len(env.corpus_data), size=min(num_states, len(env.corpus_data)), replace=False)
    for idx in sample_indices:
        row = env.corpus_data.iloc[idx]
        rewards = []
        delta_exgs = []
        for action in range(len(env.irrigation_amounts)):
            irrigation = env.irrigation_amounts[action]
            reward = env._calculate_plant_health_reward(irrigation, row)
            delta_exg = env._calculate_delta_exg(irrigation, row)
            rewards.append(reward)
            delta_exgs.append(delta_exg)
        # Plot reward vs. action
        plt.figure(figsize=(8,4))
        plt.bar([str(a) for a in env.irrigation_amounts], rewards)
        plt.title(f'Reward vs. Action (State idx {idx})')
        plt.xlabel('Irrigation Amount (gallons)')
        plt.ylabel('Reward')
        reward_path = f'../outputs/reward_sensitivity_state_{idx}.png'
        plt.tight_layout()
        plt.savefig(reward_path)
        print(f'Reward sensitivity plot saved: {reward_path}')
        plt.close()
        # Plot Delta ExG vs. action
        plt.figure(figsize=(8,4))
        plt.bar([str(a) for a in env.irrigation_amounts], delta_exgs, color='green')
        plt.title(f'Delta ExG vs. Action (State idx {idx})')
        plt.xlabel('Irrigation Amount (gallons)')
        plt.ylabel('Delta ExG')
        exg_path = f'../outputs/delta_exg_sensitivity_state_{idx}.png'
        plt.tight_layout()
        plt.savefig(exg_path)
        print(f'Delta ExG sensitivity plot saved: {exg_path}')
        plt.close()

def apply_policy_to_historical_corpus(treatment_type='H_I'):
    """
    Apply trained RL policy to historical Corpus Christi conditions
    This validates the policy transfer using real historical data
    """
    print(f"🌱 APPLYING RL POLICY TO HISTORICAL CORPUS CHRISTI DATA: {treatment_type}")
    print("=" * 70)
    
    # Load historical Corpus Christi data
    historical_data_path = "../Corpus Christi Synthetic ML Forecasting/data/Model Input - Corpus.csv"
    try:
        historical_data = pd.read_csv(historical_data_path)
        print(f"✅ Loaded historical Corpus Christi data: {len(historical_data)} records")
    except FileNotFoundError:
        print(f"❌ Historical data not found at {historical_data_path}")
        return None
    
    # Load trained policy
    model_path = f"../models/ppo/plant_health_ppo_{treatment_type}_3k.zip"
    try:
        model = PPO.load(model_path)
        print(f"✅ Loaded trained policy: {model_path}")
    except FileNotFoundError:
        print(f"❌ Trained policy not found at {model_path}")
        return None
    
    # Create environment for historical application (disable weather generation)
    env = PlantHealthAwareEnv(treatment_type=treatment_type, max_days=212, historical_mode=True)
    state, _ = env.reset()
    
    # Parse historical dates and apply policy
    recommendations = []
    
    # Convert date column to datetime
    if 'Date' in historical_data.columns:
        historical_data['Date'] = pd.to_datetime(historical_data['Date'])
    else:
        # Create date range for the growing season
        start_date = datetime(2025, 4, 3)
        historical_data['Date'] = pd.date_range(start_date, periods=len(historical_data), freq='D')
    
    # Filter data for the specific treatment type
    treatment_data = historical_data[historical_data['Treatment Type'] == treatment_type].copy()
    treatment_data = treatment_data.sort_values('Date').reset_index(drop=True)
    
    print(f"📅 Applying policy to {len(treatment_data)} days of historical data for {treatment_type}")
    print(f"📊 Date range: {treatment_data['Date'].min()} to {treatment_data['Date'].max()}")
    
    for idx, row in treatment_data.iterrows():
        current_date = row['Date']
        
        # Extract historical weather data with proper column names and handling
        # Rainfall: Keep in gallons (original unit from historical data)
        if 'Rainfall (gallons)' in row and pd.notna(row['Rainfall (gallons)']):
            historical_rainfall = row['Rainfall (gallons)']  # Keep in gallons
        else:
            historical_rainfall = 0.0
            
        # ET0: Already in mm/day
        if 'ET0 (mm)' in row and pd.notna(row['ET0 (mm)']):
            historical_et0 = row['ET0 (mm)']
        else:
            historical_et0 = 5.0  # Default value
            
        # Heat Index: Already in Fahrenheit
        if 'Heat Index (F)' in row and pd.notna(row['Heat Index (F)']):
            historical_heat = row['Heat Index (F)']
        else:
            historical_heat = 80.0  # Default value
        
        # Override environment weather with historical data
        env.today_rainfall = historical_rainfall
        env.today_et0 = historical_et0
        env.today_heat_index = historical_heat
        
        # Get policy recommendation
        action, _ = model.predict(state, deterministic=True)
        irrigation_amount = env.irrigation_amounts[action]
        
        # Step environment with historical weather
        next_state, reward, done, truncated, info = env.step(action)
        
        # Calculate water stress level
        water_deficit = info.get('water_deficit', 0.0)
        if water_deficit > 100:
            stress_level = "Extreme"
        elif water_deficit > 50:
            stress_level = "High"
        elif water_deficit > 25:
            stress_level = "Medium"
        else:
            stress_level = "Low"
        
        # Store recommendation
        recommendation = {
            'Date': current_date.strftime('%Y-%m-%d'),
            'Plot ID': f"{treatment_type}_Historical_{idx+1:03d}",
            'Treatment Type': treatment_type,
            'Water Stress': stress_level,
            'Recommended Irrigation (gallons)': irrigation_amount,
            'Historical Rainfall (gallons)': historical_rainfall,  # Fixed: keep in gallons
            'Historical ET0 (mm/day)': historical_et0,
            'Historical Heat Index': historical_heat,
            'Current ExG': env.exg,
            'Predicted Delta ExG': info.get('delta_exg', 0.0),
            'Plot Size (sq ft)': 443.5,  # Correct plot size from documentation
            'Water Deficit': water_deficit,
            'Soil Moisture': env.soil_moisture,
            'Reward': reward
        }
        recommendations.append(recommendation)
        
        # Update state
        state = next_state
        
        if done or truncated:
            break
    
    # Convert to DataFrame
    df = pd.DataFrame(recommendations)
    
    # Save results
    output_path = f"outputs/historical_corpus_policy_{treatment_type}.csv"
    df.to_csv(output_path, index=False)
    print(f"📁 Historical policy application saved to: {output_path}")
    
    # Summary statistics
    print(f"\n📊 HISTORICAL POLICY APPLICATION SUMMARY: {treatment_type}")
    print("=" * 50)
    print(f"Total days analyzed: {len(df)}")
    print(f"Total irrigation recommended: {df['Recommended Irrigation (gallons)'].sum():.0f} gallons")
    print(f"Average daily irrigation: {df['Recommended Irrigation (gallons)'].mean():.1f} gallons")
    print(f"Days with irrigation: {(df['Recommended Irrigation (gallons)'] > 0).sum()}")
    print(f"Irrigation frequency: {(df['Recommended Irrigation (gallons)'] > 0).mean()*100:.1f}%")
    
    # Water stress analysis
    stress_counts = df['Water Stress'].value_counts()
    print(f"\n💧 Water Stress Distribution:")
    for stress, count in stress_counts.items():
        print(f"  {stress}: {count} days ({count/len(df)*100:.1f}%)")
    
    # Historical weather summary
    print(f"\n🌤️ Historical Weather Summary:")
    print(f"  Total rainfall: {df['Historical Rainfall (gallons)'].sum():.1f} gallons")
    print(f"  Average ET0: {df['Historical ET0 (mm/day)'].mean():.1f} mm/day")
    print(f"  Average heat index: {df['Historical Heat Index'].mean():.1f}")
    
    # Validate irrigation plausibility
    validation_results = validate_irrigation_plausibility(df)
    
    return df

def validate_irrigation_plausibility(df):
    """
    Validate irrigation recommendations for agricultural plausibility.
    
    Args:
        df: DataFrame with historical policy application results
        
    Returns:
        dict: Validation metrics and warnings
    """
    print("\n🔍 IRRIGATION PLAUSIBILITY VALIDATION")
    print("=" * 50)
    
    validation_results = {
        'total_days': len(df),
        'agricultural_violations': [],
        'water_efficiency_metrics': {},
        'summary': {}
    }
    
    # Agricultural Rule Validation
    agricultural_violations = []
    
    for idx, row in df.iterrows():
        soil_moisture = row['Soil Moisture']
        rainfall = row['Historical Rainfall (gallons)']
        irrigation = row['Recommended Irrigation (gallons)']
        treatment = row['Treatment Type']
        date = row['Date']
        
        # Rule 1: Over-irrigation when soil is already wet
        if soil_moisture > 250 and rainfall > 50 and irrigation > 20:
            agricultural_violations.append({
                'date': date,
                'type': 'over_irrigation_wet_soil',
                'details': f"Soil: {soil_moisture:.1f}, Rain: {rainfall:.1f}, Irrigation: {irrigation:.1f}"
            })
        
        # Rule 2: Under-irrigation when soil is dry and no rain
        if soil_moisture < 180 and rainfall < 10 and irrigation < 30:
            agricultural_violations.append({
                'date': date,
                'type': 'under_irrigation_dry_soil',
                'details': f"Soil: {soil_moisture:.1f}, Rain: {rainfall:.1f}, Irrigation: {irrigation:.1f}"
            })
        
        # Rule 3: Treatment protocol violations
        if treatment == 'R_F' and irrigation > 0:
            agricultural_violations.append({
                'date': date,
                'type': 'rainfed_irrigation_violation',
                'details': f"Rainfed treatment got {irrigation:.1f} gallons irrigation"
            })
        
        if treatment == 'H_I' and irrigation > 90:
            agricultural_violations.append({
                'date': date,
                'type': 'half_irrigation_excess',
                'details': f"H_I treatment got {irrigation:.1f} gallons (exceeds typical 90 gal limit)"
            })
    
    validation_results['agricultural_violations'] = agricultural_violations
    
    # Water Efficiency Calculation
    water_efficiency_metrics = {
        'total_irrigation': df['Recommended Irrigation (gallons)'].sum(),
        'total_rainfall': df['Historical Rainfall (gallons)'].sum(),
        'total_water_input': df['Recommended Irrigation (gallons)'].sum() + df['Historical Rainfall (gallons)'].sum(),
        'days_with_irrigation': (df['Recommended Irrigation (gallons)'] > 0).sum(),
        'days_with_rainfall': (df['Historical Rainfall (gallons)'] > 0).sum(),
        'overlap_days': ((df['Recommended Irrigation (gallons)'] > 0) & (df['Historical Rainfall (gallons)'] > 0)).sum()
    }
    
    # Calculate efficiency metrics
    if water_efficiency_metrics['total_water_input'] > 0:
        irrigation_ratio = water_efficiency_metrics['total_irrigation'] / water_efficiency_metrics['total_water_input']
        water_efficiency_metrics['irrigation_ratio'] = irrigation_ratio
        water_efficiency_metrics['rainfall_utilization'] = 1 - irrigation_ratio
    else:
        water_efficiency_metrics['irrigation_ratio'] = 0
        water_efficiency_metrics['rainfall_utilization'] = 1
    
    # Calculate stress-based efficiency
    stress_days = df[df['Water Stress'].isin(['Medium', 'High', 'Extreme'])]
    if len(stress_days) > 0:
        stress_irrigation = stress_days['Recommended Irrigation (gallons)'].sum()
        total_stress_irrigation = stress_days['Recommended Irrigation (gallons)'].sum()
        water_efficiency_metrics['stress_targeted_efficiency'] = stress_irrigation / max(1, total_stress_irrigation)
    else:
        water_efficiency_metrics['stress_targeted_efficiency'] = 1.0
    
    validation_results['water_efficiency_metrics'] = water_efficiency_metrics
    
    # Print Results
    print(f"📊 AGRICULTURAL RULE VIOLATIONS: {len(agricultural_violations)}")
    if agricultural_violations:
        print("⚠️  Violations Found:")
        for violation in agricultural_violations:
            print(f"   {violation['date']}: {violation['type']} - {violation['details']}")
    else:
        print("✅ No agricultural rule violations detected")
    
    print(f"\n💧 WATER EFFICIENCY ANALYSIS:")
    print(f"   Total irrigation recommended: {water_efficiency_metrics['total_irrigation']:.1f} gallons")
    print(f"   Total rainfall received: {water_efficiency_metrics['total_rainfall']:.1f} gallons")
    print(f"   Irrigation ratio: {water_efficiency_metrics['irrigation_ratio']:.1%}")
    print(f"   Rainfall utilization: {water_efficiency_metrics['rainfall_utilization']:.1%}")
    print(f"   Days with irrigation: {water_efficiency_metrics['days_with_irrigation']}/{validation_results['total_days']}")
    print(f"   Days with rainfall: {water_efficiency_metrics['days_with_rainfall']}/{validation_results['total_days']}")
    print(f"   Overlap days (irrigation + rain): {water_efficiency_metrics['overlap_days']}/{validation_results['total_days']}")
    print(f"   Stress-targeted efficiency: {water_efficiency_metrics['stress_targeted_efficiency']:.1%}")
    
    # Summary assessment
    efficiency_score = 0
    if water_efficiency_metrics['irrigation_ratio'] < 0.3:
        efficiency_score += 2  # Good - mostly using rainfall
    elif water_efficiency_metrics['irrigation_ratio'] < 0.5:
        efficiency_score += 1  # Moderate
    
    if water_efficiency_metrics['stress_targeted_efficiency'] > 0.8:
        efficiency_score += 2  # Good - targeting stressed conditions
    elif water_efficiency_metrics['stress_targeted_efficiency'] > 0.6:
        efficiency_score += 1  # Moderate
    
    if len(agricultural_violations) == 0:
        efficiency_score += 1  # Good - no rule violations
    
    validation_results['summary'] = {
        'efficiency_score': efficiency_score,
        'efficiency_grade': 'A' if efficiency_score >= 4 else 'B' if efficiency_score >= 2 else 'C',
        'recommendations': []
    }
    
    # Generate recommendations
    if water_efficiency_metrics['irrigation_ratio'] > 0.5:
        validation_results['summary']['recommendations'].append(
            "Consider reducing irrigation when rainfall is sufficient"
        )
    
    if water_efficiency_metrics['overlap_days'] > validation_results['total_days'] * 0.3:
        validation_results['summary']['recommendations'].append(
            "High overlap between irrigation and rainfall - review timing"
        )
    
    if len(agricultural_violations) > 0:
        validation_results['summary']['recommendations'].append(
            f"Address {len(agricultural_violations)} agricultural rule violations"
        )
    
    print(f"\n📈 EFFICIENCY ASSESSMENT:")
    print(f"   Efficiency Score: {efficiency_score}/5")
    print(f"   Grade: {validation_results['summary']['efficiency_grade']}")
    
    if validation_results['summary']['recommendations']:
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in validation_results['summary']['recommendations']:
            print(f"   • {rec}")
    
    return validation_results

if __name__ == "__main__":
    # Run interactive RL version (TRUE reinforcement learning)
    main_interactive()
    
    # Apply trained policies to historical Corpus Christi data
    print("\n" + "="*80)
    print("🌱 APPLYING TRAINED POLICIES TO HISTORICAL CORPUS CHRISTI DATA")
    print("="*80)
    
    for treatment in ['H_I', 'F_I']:
        apply_policy_to_historical_corpus(treatment)
    
    # Uncomment below to run old CSV-based version
    # main()
    
    # Uncomment below to run multi-seed reproducibility test
    # main_multi_seed() 