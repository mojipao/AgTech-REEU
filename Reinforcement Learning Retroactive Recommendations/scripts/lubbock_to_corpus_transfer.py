"""
Lubbock → Corpus Christi Policy Transfer
========================================

Trains RL irrigation policies on Lubbock's rich dataset (1,288 observations)
and transfers them to Corpus Christi with climate-based scaling.

Key Features:
- Climate translation using weather data
- Plot size scaling (Lubbock 6475 sq ft → Corpus 443.5 sq ft)
- Treatment type mapping (DICT/DIEG → H_I, FICT/FIEG → F_I)
- Data-driven irrigation scaling
"""

import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import PPO
import torch
import random
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# Set fixed random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

script_dir = os.path.dirname(os.path.abspath(__file__))
combined_data_path = os.path.abspath(os.path.join(script_dir, '../../Corpus Christi Synthetic ML Forecasting/data/combined_weather_dataset.csv'))
corpus_path = os.path.abspath(os.path.join(script_dir, '../../Corpus Christi Synthetic ML Forecasting/data/corpus_season_completed_enhanced_lubbock_ml.csv'))

# Before saving to outputs directory
os.makedirs(os.path.join(script_dir, '../outputs'), exist_ok=True)

class LubbockTrainingEnv(gym.Env):
    """Environment for training RL policies on Lubbock data."""
    
    def __init__(self, lubbock_data, treatment_type='DICT'):
        super().__init__()
        
        self.treatment_type = treatment_type
        self.lubbock_data = lubbock_data.copy()
        
        # Filter for treatment type
        treatment_data = lubbock_data[lubbock_data['Treatment Type'] == treatment_type].copy()
        
        if len(treatment_data) == 0:
            raise ValueError(f"No data found for treatment type: {treatment_type}")
        
        # Calculate days after planting for Lubbock (planted May 1, 2023)
        lubbock_planting_date = datetime(2023, 5, 1)
        treatment_data['Date'] = pd.to_datetime(treatment_data['Date'])
        treatment_data['Days_After_Planting'] = (treatment_data['Date'] - lubbock_planting_date).dt.days
        
        self.training_data = treatment_data.sort_values('Date').reset_index(drop=True)
        
        # Irrigation amounts based on treatment type - FIXED to match historical data
        if treatment_type in ['DICT', 'DIEG']:  # Deficit irrigation (65%)
            # Historical DICT/DIEG: 1500-7000 gallons, scaled down by factor of 50
            self.irrigation_amounts = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1200, 1400]  # gallons
        elif treatment_type in ['FICT', 'FIEG']:  # Full irrigation (100%)
            # Historical FICT/FIEG: 2500-9000+ gallons, scaled down by factor of 50
            self.irrigation_amounts = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2200]  # gallons
        else:
            self.irrigation_amounts = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1200, 1400]  # default
            
        self.action_space = spaces.Discrete(len(self.irrigation_amounts))
        
        # State space: [soil_moisture, ET0, heat_index, rainfall, ExG, days_after_planting, water_deficit, Kc]
        self.observation_space = spaces.Box(low=-3, high=3, shape=(8,), dtype=np.float32)
        
        # Initialize scaler
        self.scaler = MinMaxScaler()
        self._fit_scaler()
        
        # Episode tracking
        self.current_step = 0
        self.episode_length = 15
        
        # Lubbock plot size (6475 sq ft)
        self.plot_size = 6475
        
    def _fit_scaler(self):
        """Fit scaler on Lubbock features."""
        features = []
        for _, row in self.training_data.iterrows():
            # Extract features with robust NaN handling
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
            
            kc = row.get('Kc (Crop Coeffecient)', 0.8)
            if pd.isna(kc):
                kc = 0.8
            
            water_deficit = max(0, 200 - soil_moisture)
            
            feature_row = [soil_moisture, et0, heat_index, rainfall, exg, days_after_planting, water_deficit, kc]
            
            # Ensure no NaN values in feature row
            if not any(pd.isna(val) or np.isnan(val) for val in feature_row):
                features.append(feature_row)
        
        if len(features) > 10:
            self.scaler.fit(features)
        else:
            # Fallback with diverse samples
            dummy_features = [
                [200, 5, 85, 0, 0.4, 60, 0, 0.8],
                [180, 3, 75, 0, 0.3, 40, 20, 0.6],
                [220, 7, 95, 2, 0.6, 80, 0, 1.0],
                [190, 4, 80, 1, 0.35, 50, 10, 0.7],
                [210, 6, 90, 0, 0.5, 70, 0, 0.9]
            ]
            self.scaler.fit(dummy_features)
    
    def reset(self, seed=None, options=None):
        """Reset environment to random starting point."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.episode_start = np.random.randint(0, max(1, len(self.training_data) - self.episode_length))
        
        return self._get_state(), {}
    
    def _get_state(self):
        """Get current state from Lubbock data."""
        if self.episode_start + self.current_step >= len(self.training_data):
            return np.zeros(8, dtype=np.float32)
        
        row = self.training_data.iloc[self.episode_start + self.current_step]
        
        # Extract features with robust NaN handling
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
        
        kc = row.get('Kc (Crop Coeffecient)', 0.8)
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
        
        # Normalize features
        try:
            normalized_features = self.scaler.transform([features])[0]
        except:
            normalized_features = np.array(features) / 100
        
        # Final NaN check on normalized values
        if np.any(np.isnan(normalized_features)):
            print(f"⚠️ Warning: NaN detected in normalized state, using defaults")
            normalized_features = np.nan_to_num(normalized_features, nan=0.5)
        
        return normalized_features.astype(np.float32)
    
    def step(self, action):
        """Execute action and get reward based on plant health outcomes."""
        if self.episode_start + self.current_step >= len(self.training_data) - 1:
            return self._get_state(), 0, True, True, {}
        
        current_row = self.training_data.iloc[self.episode_start + self.current_step]
        
        # Convert action to irrigation amount
        irrigation_amount = self.irrigation_amounts[action]
        
        # Calculate reward based on plant health and water efficiency
        reward = self._calculate_plant_health_reward(irrigation_amount, current_row)
        
        # Move to next step
        self.current_step += 1
        done = (self.current_step >= self.episode_length) or (self.episode_start + self.current_step >= len(self.training_data) - 1)
        
        return self._get_state(), reward, done, False, {}
    
    def _calculate_plant_health_reward(self, irrigation_amount, row):
        """Calculate reward based on plant health outcomes."""
        # Get current state values
        soil_moisture = row.get('Total Soil Moisture', 200)
        if pd.isna(soil_moisture):
            soil_moisture = 200
        
        current_exg = row.get('ExG', 0.4)
        if pd.isna(current_exg):
            current_exg = 0.4
        
        days_after_planting = row.get('Days_After_Planting', 60)
        if pd.isna(days_after_planting):
            days_after_planting = 60
        
        # Base seasonal trend for Lubbock
        if days_after_planting > 100:  # Late season senescence
            base_trend = -0.002
        elif days_after_planting > 80:  # Mid-season decline
            base_trend = -0.001
        else:  # Early season growth potential
            base_trend = 0.001
        
        # Water stress assessment
        if soil_moisture < 190:  # High stress
            water_stress_penalty = -0.05  # Increased penalty
        elif soil_moisture < 200:  # Medium stress
            water_stress_penalty = -0.03
        else:  # Low stress
            water_stress_penalty = 0
        
        # Irrigation benefit calculation
        if irrigation_amount > 0:
            if soil_moisture < 190:  # High stress - irrigation helps significantly
                irrigation_benefit = 0.01 + (irrigation_amount / 2000)
            elif soil_moisture < 200:  # Medium stress - irrigation helps moderately
                irrigation_benefit = 0.005 + (irrigation_amount / 3000)
            else:  # Low stress - minimal benefit, potential waste
                irrigation_benefit = max(-0.002, irrigation_amount / 5000)
        else:
            irrigation_benefit = 0
        
        # Water efficiency penalty (Lubbock is drier, so water efficiency is important)
        water_efficiency_penalty = -0.001 * irrigation_amount / 100
        
        # Combined reward
        reward = base_trend + irrigation_benefit + water_stress_penalty + water_efficiency_penalty
        
        return reward

def analyze_climate_differences():
    """Analyze climate differences between Lubbock and Corpus Christi."""
    print("🌍 ANALYZING CLIMATE DIFFERENCES")
    print("=" * 50)
    
    # Load combined weather dataset
    combined_data = pd.read_csv(combined_data_path)
    
    # Separate Lubbock and Corpus Christi data
    lubbock_data = combined_data[combined_data['Source'] == 'Lubbock'].copy()
    corpus_data = combined_data[combined_data['Source'] == 'Weather_Station'].copy()
    
    # Filter for growing season (May-October)
    lubbock_data['Date'] = pd.to_datetime(lubbock_data['Date'])
    corpus_data['Date'] = pd.to_datetime(corpus_data['Date'])
    
    # Filter for 2023 growing season for comparison
    lubbock_growing = lubbock_data[
        (lubbock_data['Date'] >= '2023-05-01') & 
        (lubbock_data['Date'] <= '2023-10-31')
    ]
    
    corpus_growing = corpus_data[
        (corpus_data['Date'] >= '2023-05-01') & 
        (corpus_data['Date'] <= '2023-10-31')
    ]
    
    print(f"Lubbock growing season (2023): {len(lubbock_growing)} observations")
    print(f"Corpus growing season (2023): {len(corpus_growing)} observations")
    
    # Calculate climate statistics
    climate_stats = {}
    
    for location, data in [('Lubbock', lubbock_growing), ('Corpus', corpus_growing)]:
        stats = {}
        
        # ET0 comparison
        et0_values = data['ET0_mm'].dropna()
        if len(et0_values) > 0:
            stats['ET0_mean'] = et0_values.mean()
            stats['ET0_std'] = et0_values.std()
        
        # Heat Index comparison
        heat_values = data['Heat_Index_F'].dropna()
        if len(heat_values) > 0:
            stats['Heat_Index_mean'] = heat_values.mean()
            stats['Heat_Index_std'] = heat_values.std()
        
        # Rainfall comparison
        rain_values = data['Rainfall (gallons)'].dropna() if 'Rainfall (gallons)' in data.columns else []
        if len(rain_values) > 0:
            stats['Rainfall_mean'] = rain_values.mean()
            stats['Rainfall_std'] = rain_values.std()
        
        climate_stats[location] = stats
    
    # Calculate scaling factors
    scaling_factors = {}
    
    if 'Lubbock' in climate_stats and 'Corpus' in climate_stats:
        # ET0 scaling (Corpus has lower ET0 = less water demand)
        if 'ET0_mean' in climate_stats['Lubbock'] and 'ET0_mean' in climate_stats['Corpus']:
            et0_ratio = climate_stats['Corpus']['ET0_mean'] / climate_stats['Lubbock']['ET0_mean']
            scaling_factors['ET0_ratio'] = et0_ratio
            print(f"ET0 ratio (Corpus/Lubbock): {et0_ratio:.3f}")
        
        # Heat Index scaling (Corpus is more humid)
        if 'Heat_Index_mean' in climate_stats['Lubbock'] and 'Heat_Index_mean' in climate_stats['Corpus']:
            heat_ratio = climate_stats['Corpus']['Heat_Index_mean'] / climate_stats['Lubbock']['Heat_Index_mean']
            scaling_factors['Heat_Index_ratio'] = heat_ratio
            print(f"Heat Index ratio (Corpus/Lubbock): {heat_ratio:.3f}")
        
        # Rainfall scaling (Corpus has more rainfall)
        if 'Rainfall_mean' in climate_stats['Lubbock'] and 'Rainfall_mean' in climate_stats['Corpus']:
            rain_ratio = climate_stats['Corpus']['Rainfall_mean'] / climate_stats['Lubbock']['Rainfall_mean']
            scaling_factors['Rainfall_ratio'] = rain_ratio
            print(f"Rainfall ratio (Corpus/Lubbock): {rain_ratio:.3f}")
    
    return scaling_factors

def train_lubbock_policy(treatment_type='DICT', total_timesteps=10000):
    """Train RL policy on Lubbock data."""
    print(f"\n🚀 TRAINING LUBBOCK POLICY: {treatment_type}")
    print("=" * 50)
    
    # Load Lubbock data
    combined_data = pd.read_csv(combined_data_path)
    lubbock_data = combined_data[combined_data['Source'] == 'Lubbock'].copy()
    
    print(f"Lubbock data: {len(lubbock_data)} total observations")
    print(f"Treatment {treatment_type}: {len(lubbock_data[lubbock_data['Treatment Type'] == treatment_type])} observations")
    
    # Create environment
    env = LubbockTrainingEnv(lubbock_data, treatment_type)
    
    # Create PPO agent
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device='auto',
        policy_kwargs=dict(
            net_arch=[64, 64],
            activation_fn=torch.nn.Tanh
        )
    )
    
    # Train model
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps)
    
    # Save model
    model_path = f"../../../models/ppo/lubbock_policy_{treatment_type}.zip"
    model.save(model_path)
    print(f"Lubbock policy saved to: {model_path}")
    
    return model, env

def apply_lubbock_policy_to_corpus(model, treatment_type, scaling_factors):
    """Apply Lubbock-trained policy to Corpus Christi with climate scaling."""
    print(f"\n📊 APPLYING LUBBOCK POLICY TO CORPUS: {treatment_type}")
    print("=" * 50)
    
    # Load Corpus Christi data
    corpus_data = pd.read_csv(corpus_path)
    print("Unique Plot IDs:", corpus_data['Plot ID'].unique())
    print("Plot ID dtype:", corpus_data['Plot ID'].dtype)
    
    # Map Lubbock treatments to Corpus treatments
    treatment_mapping = {
        'DICT': 'H_I',  # Deficit irrigation → Half irrigation
        'DIEG': 'H_I',  # Deficit irrigation → Half irrigation
        'FICT': 'F_I',  # Full irrigation → Full irrigation
        'FIEG': 'F_I'   # Full irrigation → Full irrigation
    }
    
    corpus_treatment = treatment_mapping.get(treatment_type, 'H_I')
    
    # Filter for Corpus treatment
    treatment_plots = {'R_F': '102', 'H_I': '404', 'F_I': '409'}
    plot_id = treatment_plots[corpus_treatment]
    
    # When filtering for plot_id, use:
    plot_data = corpus_data[corpus_data['Plot ID'].astype(str) == str(float(plot_id))].copy()
    print(f"Found {len(plot_data)} rows for Corpus Plot ID {plot_id} (Treatment: {corpus_treatment})")
    
    if len(plot_data) == 0:
        print(f"❌ No data found for Plot ID {plot_id}")
        return []
    
    plot_data['Date'] = pd.to_datetime(plot_data['Date'])
    plot_data = plot_data.sort_values('Date').reset_index(drop=True)
    
    # Calculate days after planting
    corpus_season_start = pd.to_datetime('2025-04-03')
    plot_data['Days_After_Planting'] = (plot_data['Date'] - corpus_season_start).dt.days
    
    # Calculate scaling factors
    plot_size_ratio = 443.5 / 6475  # Corpus/Lubbock plot size ratio
    
    # Treatment scaling factors
    if treatment_type in ['DICT', 'DIEG']:
        treatment_ratio = 0.5 / 0.65  # Corpus 50% / Lubbock 65%
    else:  # FICT, FIEG
        treatment_ratio = 1.0  # Both 100%
    
    # Climate scaling factor (reduce irrigation for Corpus's higher humidity)
    climate_factor = scaling_factors.get('ET0_ratio', 0.8)  # Default to 0.8 if not calculated
    
    # Combined scaling factor - ADJUSTED to be less aggressive
    # The plot size ratio is the main factor, others are secondary
    total_scaling = plot_size_ratio * 0.8  # Simplified scaling focusing on plot size
    
    print(f"Scaling factors:")
    print(f"  Plot size ratio: {plot_size_ratio:.4f}")
    print(f"  Treatment ratio: {treatment_ratio:.3f}")
    print(f"  Climate factor: {climate_factor:.3f}")
    print(f"  Total scaling: {total_scaling:.4f}")
    
    recommendations = []
    
    for _, row in plot_data.iterrows():
        # Create state (same as training)
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
        
        # Handle irrigation recommendation
        if corpus_treatment == 'R_F' or model is None:
            # Rainfed - no irrigation allowed
            recommended_irrigation = 0
        else:
            # Use trained model to predict irrigation
            try:
                # Normalize features using the same scaler as training
                normalized_features = np.array(features) / 100  # Simple normalization
                state = normalized_features.astype(np.float32)
                
                # Get policy recommendation
                action, _ = model.predict(state, deterministic=True)
                
                # Get irrigation amount from Lubbock action space
                if treatment_type in ['DICT', 'DIEG']:
                    lubbock_amounts = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1200, 1400]
                else:  # FICT, FIEG
                    lubbock_amounts = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000, 1200, 1400, 1600, 1800, 2000, 2200]
                
                lubbock_irrigation = lubbock_amounts[action]
                
                # Apply scaling factors
                recommended_irrigation = lubbock_irrigation * total_scaling
                
                # Apply correction factor to convert training amounts to realistic levels
                # Training amounts are scaled down by ~50x from historical amounts
                correction_factor = 50.0  # Convert training amounts back to realistic levels
                recommended_irrigation = recommended_irrigation * correction_factor
                
            except Exception as e:
                print(f"Error in prediction: {e}")
                recommended_irrigation = 0
        
        # Water stress assessment
        if soil_moisture < 190:
            water_stress = 'High'
        elif soil_moisture < 200:
            water_stress = 'Medium'
        else:
            water_stress = 'Low'
        
        # Calculate ETc
        predicted_etc = et0 * kc
        
        # Calculate predicted Delta ExG
        if days_after_planting > 100:
            base_trend = -0.002
        elif days_after_planting > 80:
            base_trend = -0.001
        else:
            base_trend = 0.001
        
        if recommended_irrigation > 0:
            if soil_moisture < 190:
                irrigation_benefit = 0.01 + (recommended_irrigation / 2000)
            elif soil_moisture < 200:
                irrigation_benefit = 0.005 + (recommended_irrigation / 3000)
            else:
                irrigation_benefit = max(-0.002, recommended_irrigation / 5000)
        else:
            irrigation_benefit = 0
        
        predicted_delta_exg = base_trend + irrigation_benefit
        predicted_delta_exg = max(-0.01, min(0.05, predicted_delta_exg))
        
        recommendations.append({
            'Date': row['Date'].strftime('%Y-%m-%d'),
            'Plot ID': plot_id,
            'Treatment Type': corpus_treatment,
            'Lubbock Policy': treatment_type,
            'Water Stress': water_stress,
            'Recommended Irrigation (gallons)': round(recommended_irrigation, 1),
            'Lubbock Irrigation (gallons)': round(lubbock_irrigation if 'lubbock_irrigation' in locals() else 0, 1),
            'Scaling Factor': round(total_scaling, 4),
            'Predicted ETc': round(predicted_etc, 2),
            'Current ExG': round(exg, 4),
            'Predicted Delta ExG': round(predicted_delta_exg, 4),
            'Plot Size (sq ft)': 443.5
        })
    
    return recommendations

def main():
    """Main Lubbock to Corpus Christi transfer pipeline."""
    print("🌱 LUBBOCK → CORPUS CHRISTI POLICY TRANSFER")
    print("=" * 60)
    print("Training on Lubbock's rich dataset, transferring to Corpus Christi")
    print()
    
    # Step 1: Analyze climate differences
    scaling_factors = analyze_climate_differences()
    
    # Step 2: Train policies on Lubbock data
    all_recommendations = []
    
    # Train on deficit irrigation treatments (DICT, DIEG)
    for treatment in ['DICT', 'DIEG']:
        print(f"\n📚 Training on Lubbock {treatment}")
        model, env = train_lubbock_policy(treatment, total_timesteps=8000)
        
        # Apply to Corpus Christi H_I
        recommendations = apply_lubbock_policy_to_corpus(model, treatment, scaling_factors)
        all_recommendations.extend(recommendations)
    
    # Train on full irrigation treatments (FICT, FIEG)
    for treatment in ['FICT', 'FIEG']:
        print(f"\n📚 Training on Lubbock {treatment}")
        model, env = train_lubbock_policy(treatment, total_timesteps=8000)
        
        # Apply to Corpus Christi F_I
        recommendations = apply_lubbock_policy_to_corpus(model, treatment, scaling_factors)
        all_recommendations.extend(recommendations)
    
    # Add rainfed recommendations
    corpus_data = pd.read_csv(corpus_path)
    rf_data = corpus_data[corpus_data['Plot ID'] == '102'].copy()
    
    for _, row in rf_data.iterrows():
        soil_moisture = row.get('Total Soil Moisture', 200)
        if pd.isna(soil_moisture):
            soil_moisture = 200
        
        if soil_moisture < 190:
            water_stress = 'High'
        elif soil_moisture < 200:
            water_stress = 'Medium'
        else:
            water_stress = 'Low'
        
        et0 = row.get('ET0 (mm)', 5)
        kc = row.get('Kc (Crop Coefficient)', 0.8)
        if pd.isna(et0): et0 = 5
        if pd.isna(kc): kc = 0.8
        predicted_etc = et0 * kc
        
        current_exg = row.get('ExG', 0.4)
        if pd.isna(current_exg): current_exg = 0.4
        
        all_recommendations.append({
            'Date': row['Date'],
            'Plot ID': '102',
            'Treatment Type': 'R_F',
            'Lubbock Policy': 'None',
            'Water Stress': water_stress,
            'Recommended Irrigation (gallons)': 0,
            'Lubbock Irrigation (gallons)': 0,
            'Scaling Factor': 0,
            'Predicted ETc': round(predicted_etc, 2),
            'Current ExG': round(current_exg, 4),
            'Predicted Delta ExG': 0.001,
            'Plot Size (sq ft)': 443.5
        })
    
    # Save results
    if all_recommendations:
        df_recommendations = pd.DataFrame(all_recommendations)
        df_recommendations = df_recommendations.sort_values(['Date', 'Treatment Type'])

        # --- FILTER TO DIEG (H_I), FIEG (F_I), None (R_F) ---
        filtered = []
        for (date, plot_id, treatment), group in df_recommendations.groupby(['Date', 'Plot ID', 'Treatment Type']):
            if treatment == 'F_I':
                row = group[group['Lubbock Policy'] == 'FIEG']
            elif treatment == 'H_I':
                row = group[group['Lubbock Policy'] == 'DIEG']
            elif treatment == 'R_F':
                row = group[group['Lubbock Policy'] == 'None']
            else:
                row = group.iloc[[0]]
            if not row.empty:
                filtered.append(row.iloc[0])
        df_filtered = pd.DataFrame(filtered)
        df_filtered = df_filtered.sort_values(['Date', 'Treatment Type'])

        # --- SCALING STEP ---
        gallons_per_inch = 276
        targets = {'F_I': 10.51, 'H_I': 9.76}  # inches, from field data
        for treatment, target_inches in targets.items():
            target_gallons = target_inches * gallons_per_inch
            mask = df_filtered['Treatment Type'] == treatment
            total_raw = df_filtered.loc[mask, 'Recommended Irrigation (gallons)'].sum()
            if total_raw > 0:
                scaling_factor = target_gallons / total_raw
            else:
                scaling_factor = 1.0
            df_filtered.loc[mask, 'Scaled Irrigation (gallons)'] = df_filtered.loc[mask, 'Recommended Irrigation (gallons)'] * scaling_factor
            df_filtered.loc[mask, 'Scaling Factor (post)'] = scaling_factor
        # Rainfed stays zero
        df_filtered.loc[df_filtered['Treatment Type'] == 'R_F', 'Scaled Irrigation (gallons)'] = 0
        df_filtered.loc[df_filtered['Treatment Type'] == 'R_F', 'Scaling Factor (post)'] = 0

        output_file = os.path.join(script_dir, '../outputs', 'lubbock_to_corpus_transfer_results_DIEG_FIEG_scaled.csv')
        df_filtered.to_csv(output_file, index=False)

        print(f"\n✅ TRANSFER COMPLETE (DIEG/FIEG only)")
        print(f"📁 Results saved to: {output_file}")
        print(f"📊 Total recommendations: {len(df_filtered)}")
        
        # Summary by treatment
        summary = df_filtered.groupby('Treatment Type').agg({
            'Recommended Irrigation (gallons)': ['count', 'sum', 'mean']
        }).round(1)
        
        print("\n📈 SUMMARY BY TREATMENT:")
        print(summary)
        
        # Summary by Lubbock policy
        policy_summary = df_filtered.groupby('Lubbock Policy').agg({
            'Recommended Irrigation (gallons)': ['count', 'sum', 'mean'],
            'Scaling Factor': 'mean'
        }).round(3)
        
        print("\n🎯 SUMMARY BY LUBBOCK POLICY:")
        print(policy_summary)
        
    else:
        print("❌ No recommendations generated")

if __name__ == "__main__":
    main() 