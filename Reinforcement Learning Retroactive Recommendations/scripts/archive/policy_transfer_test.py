"""
Policy Transfer Test: Corpus Christi → Lubbock
==============================================

Tests how well RL irrigation policies trained on Corpus Christi data
generalize to Lubbock, Texas - a different geographic location with
different climate patterns and soil conditions.

This validates the robustness and generalizability of the learned policies.
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

class PolicyTransferEnv(gym.Env):
    """Environment for testing policy transfer to Lubbock data."""
    
    def __init__(self, lubbock_data, treatment_type='DICT'):
        super().__init__()
        
        self.treatment_type = treatment_type
        self.lubbock_data = lubbock_data.copy()
        
        # Map Lubbock treatment types to Corpus Christi equivalents
        self.treatment_mapping = {
            'DICT': 'H_I',  # Deficit irrigation control → Half irrigation
            'DIEG': 'H_I',  # Deficit irrigation experimental → Half irrigation  
            'FIEG': 'F_I',  # Full irrigation experimental → Full irrigation
            'FICT': 'F_I'   # Full irrigation control → Full irrigation
        }
        
        mapped_treatment = self.treatment_mapping.get(treatment_type, 'H_I')
        
        # Irrigation amounts based on mapped treatment
        if mapped_treatment == 'F_I':  # Full irrigation
            self.irrigation_amounts = [0, 25, 50, 75, 100, 125, 150]  # gallons
        elif mapped_treatment == 'H_I':  # Half irrigation
            self.irrigation_amounts = [0, 15, 30, 45, 60, 75, 90]   # gallons
        else:  # Rainfed
            self.irrigation_amounts = [0]  # No irrigation allowed
            
        self.action_space = spaces.Discrete(len(self.irrigation_amounts))
        
        # State space: [soil_moisture, ET0, heat_index, rainfall, ExG, days_after_planting, water_deficit, Kc]
        self.observation_space = spaces.Box(low=-3, high=3, shape=(8,), dtype=np.float32)
        
        # Initialize scaler (will be fitted on Corpus Christi data)
        self.scaler = MinMaxScaler()
        
        # Episode tracking
        self.current_step = 0
        self.episode_length = 15
        
        # Lubbock plot size (6475 sq ft vs Corpus 443.5 sq ft)
        self.plot_size_ratio = 6475 / 443.5  # ~14.6x larger
        
    def fit_scaler_on_corpus(self, corpus_data):
        """Fit scaler on Corpus Christi data to maintain consistent normalization."""
        features = []
        for _, row in corpus_data.iterrows():
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
            
            kc = row.get('Kc (Crop Coefficient)', 0.8)
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
        self.episode_start = np.random.randint(0, max(1, len(self.lubbock_data) - self.episode_length))
        
        return self._get_state(), {}
    
    def _get_state(self):
        """Get current state from Lubbock data with robust NaN handling."""
        if self.episode_start + self.current_step >= len(self.lubbock_data):
            return np.zeros(8, dtype=np.float32)
        
        row = self.lubbock_data.iloc[self.episode_start + self.current_step]
        
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
        
        # Calculate days after planting for Lubbock (planted May 1, 2023)
        lubbock_planting_date = datetime(2023, 5, 1)
        date = pd.to_datetime(row['Date'])
        days_after_planting = (date - lubbock_planting_date).days
        
        kc = row.get('Kc (Crop Coeffecient)', 0.8)  # Note: Lubbock has different column name
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
        
        # Normalize features using Corpus Christi scaler
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
        if self.episode_start + self.current_step >= len(self.lubbock_data) - 1:
            return self._get_state(), 0, True, True, {}
        
        current_row = self.lubbock_data.iloc[self.episode_start + self.current_step]
        
        # Convert action to irrigation amount
        irrigation_amount = self.irrigation_amounts[action]
        
        # Calculate reward based on plant health and water efficiency
        reward = self._calculate_plant_health_reward(irrigation_amount, current_row)
        
        # Move to next step
        self.current_step += 1
        done = (self.current_step >= self.episode_length) or (self.episode_start + self.current_step >= len(self.lubbock_data) - 1)
        
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
        
        # Calculate days after planting
        lubbock_planting_date = datetime(2023, 5, 1)
        date = pd.to_datetime(row['Date'])
        days_after_planting = (date - lubbock_planting_date).days
        
        # Base seasonal trend for Lubbock (similar to Corpus but adjusted for climate)
        if days_after_planting > 100:  # Late season senescence
            base_trend = -0.002
        elif days_after_planting > 80:  # Mid-season decline
            base_trend = -0.001
        else:  # Early season growth potential
            base_trend = 0.001
        
        # Water stress assessment
        if soil_moisture < 190:  # High stress
            water_stress_penalty = -0.02
        elif soil_moisture < 200:  # Medium stress
            water_stress_penalty = -0.01
        else:  # Low stress
            water_stress_penalty = 0
        
        # Irrigation benefit calculation (adjusted for Lubbock conditions)
        if irrigation_amount > 0:
            if soil_moisture < 190:  # High stress - irrigation helps significantly
                irrigation_benefit = 0.01 + (irrigation_amount / 2000)
            elif soil_moisture < 200:  # Medium stress - irrigation helps moderately
                irrigation_benefit = 0.005 + (irrigation_amount / 3000)
            else:  # Low stress - minimal benefit, potential waste
                irrigation_benefit = max(-0.002, irrigation_amount / 5000)
        else:
            irrigation_benefit = 0
        
        # Water efficiency penalty (Lubbock is drier, so water efficiency is more important)
        water_efficiency_penalty = -0.001 * irrigation_amount / 100  # Small penalty for water use
        
        # Combined reward
        reward = base_trend + irrigation_benefit + water_stress_penalty + water_efficiency_penalty
        
        return reward

def load_corpus_data():
    """Load Corpus Christi data for scaler fitting."""
    corpus_path = '../../Corpus Christi Synthetic ML Forecasting/../../data/corpus_season_completed_enhanced_lubbock_ml.csv'
    corpus_data = pd.read_csv(corpus_path)
    return corpus_data

def load_lubbock_data():
    """Load Lubbock data for policy transfer testing."""
    lubbock_path = '../../Corpus Christi Synthetic ML Forecasting/../../data/Model Input - Lubbock-3.csv'
    lubbock_data = pd.read_csv(lubbock_path)
    lubbock_data['Date'] = pd.to_datetime(lubbock_data['Date'])
    return lubbock_data

def load_trained_policies():
    """Load trained RL policies from Corpus Christi."""
    policies = {}
    
    # Load Half Irrigation policy
    try:
        hi_policy_path = '../../../models/ppo/plant_health_ppo_H_I.zip'
        if os.path.exists(hi_policy_path):
            policies['H_I'] = PPO.load(hi_policy_path)
            print(f"✅ Loaded H_I policy from {hi_policy_path}")
        else:
            print(f"❌ H_I policy not found at {hi_policy_path}")
    except Exception as e:
        print(f"❌ Error loading H_I policy: {e}")
    
    # Load Full Irrigation policy
    try:
        fi_policy_path = '../../../models/ppo/plant_health_ppo_F_I.zip'
        if os.path.exists(fi_policy_path):
            policies['F_I'] = PPO.load(fi_policy_path)
            print(f"✅ Loaded F_I policy from {fi_policy_path}")
        else:
            print(f"❌ F_I policy not found at {fi_policy_path}")
    except Exception as e:
        print(f"❌ Error loading F_I policy: {e}")
    
    return policies

def test_policy_transfer(policy, lubbock_data, treatment_type, corpus_data):
    """Test policy transfer to Lubbock data."""
    print(f"\n🧪 Testing Policy Transfer: {treatment_type}")
    print("=" * 50)
    
    # Create environment
    env = PolicyTransferEnv(lubbock_data, treatment_type)
    env.fit_scaler_on_corpus(corpus_data)
    
    # Filter Lubbock data for this treatment type
    treatment_data = lubbock_data[lubbock_data['Treatment Type'] == treatment_type].copy()
    
    if len(treatment_data) == 0:
        print(f"❌ No data found for treatment type: {treatment_type}")
        return None
    
    print(f"📊 Lubbock data: {len(treatment_data)} observations for {treatment_type}")
    
    # Test policy on multiple episodes
    results = []
    num_episodes = min(10, len(treatment_data) // 15)  # Test up to 10 episodes
    
    for episode in range(num_episodes):
        # Set episode start to different parts of the data
        episode_start = episode * (len(treatment_data) // num_episodes)
        env.episode_start = episode_start
        
        obs, _ = env.reset()
        episode_reward = 0
        episode_actions = []
        episode_states = []
        
        for step in range(env.episode_length):
            if policy is None:
                action = 0  # No irrigation for rainfed
            else:
                action, _ = policy.predict(obs, deterministic=True)
            
            episode_actions.append(env.irrigation_amounts[action])
            episode_states.append(obs.copy())
            
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            
            if done:
                break
        
        results.append({
            'episode': episode,
            'treatment_type': treatment_type,
            'total_reward': episode_reward,
            'avg_irrigation': np.mean(episode_actions),
            'total_irrigation': np.sum(episode_actions),
            'episode_length': len(episode_actions)
        })
    
    return results

def analyze_transfer_results(all_results):
    """Analyze policy transfer results."""
    if not all_results:
        print("❌ No results to analyze")
        return
    
    # Combine all results
    df_results = pd.DataFrame(all_results)
    
    print("\n📈 POLICY TRANSFER ANALYSIS")
    print("=" * 50)
    
    # Summary statistics by treatment type
    summary = df_results.groupby('treatment_type').agg({
        'total_reward': ['count', 'mean', 'std'],
        'avg_irrigation': ['mean', 'std'],
        'total_irrigation': ['mean', 'std']
    }).round(3)
    
    print("\n📊 Summary by Treatment Type:")
    print(summary)
    
    # Compare with expected performance
    print("\n🔍 Performance Analysis:")
    for treatment in df_results['treatment_type'].unique():
        treatment_data = df_results[df_results['treatment_type'] == treatment]
        
        avg_reward = treatment_data['total_reward'].mean()
        avg_irrigation = treatment_data['avg_irrigation'].mean()
        
        print(f"\n{treatment}:")
        print(f"  Average Reward: {avg_reward:.3f}")
        print(f"  Average Irrigation: {avg_irrigation:.1f} gallons")
        
        # Performance assessment
        if avg_reward > 0:
            print(f"  ✅ Policy performing well (positive reward)")
        elif avg_reward > -0.1:
            print(f"  ⚠️ Policy performing moderately (slight negative reward)")
        else:
            print(f"  ❌ Policy struggling (significant negative reward)")
    
    # Create visualization
    create_transfer_visualizations(df_results)
    
    return df_results

def create_transfer_visualizations(df_results):
    """Create visualizations for policy transfer results."""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Policy Transfer Analysis: Corpus Christi → Lubbock', fontsize=16, fontweight='bold')
    
    # 1. Reward distribution by treatment
    ax1 = axes[0, 0]
    for treatment in df_results['treatment_type'].unique():
        treatment_data = df_results[df_results['treatment_type'] == treatment]
        ax1.hist(treatment_data['total_reward'], alpha=0.7, label=treatment, bins=10)
    ax1.set_xlabel('Total Episode Reward')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Reward Distribution by Treatment')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Average irrigation by treatment
    ax2 = axes[0, 1]
    irrigation_means = df_results.groupby('treatment_type')['avg_irrigation'].mean()
    irrigation_stds = df_results.groupby('treatment_type')['avg_irrigation'].std()
    
    treatments = irrigation_means.index
    means = irrigation_means.values
    stds = irrigation_stds.values
    
    bars = ax2.bar(treatments, means, yerr=stds, capsize=5, alpha=0.7)
    ax2.set_xlabel('Treatment Type')
    ax2.set_ylabel('Average Irrigation (gallons)')
    ax2.set_title('Average Irrigation by Treatment')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{mean:.1f}', ha='center', va='bottom')
    
    # 3. Reward vs Irrigation scatter
    ax3 = axes[1, 0]
    for treatment in df_results['treatment_type'].unique():
        treatment_data = df_results[df_results['treatment_type'] == treatment]
        ax3.scatter(treatment_data['avg_irrigation'], treatment_data['total_reward'], 
                   alpha=0.7, label=treatment, s=50)
    ax3.set_xlabel('Average Irrigation (gallons)')
    ax3.set_ylabel('Total Episode Reward')
    ax3.set_title('Reward vs Irrigation Relationship')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Treatment comparison boxplot
    ax4 = axes[1, 1]
    df_results.boxplot(column='total_reward', by='treatment_type', ax=ax4)
    ax4.set_xlabel('Treatment Type')
    ax4.set_ylabel('Total Episode Reward')
    ax4.set_title('Reward Distribution Comparison')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = '../outputs'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/policy_transfer_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {output_dir}/policy_transfer_analysis.png")
    
    plt.show()

def main():
    """Main policy transfer testing function."""
    print("🌱 POLICY TRANSFER TEST: Corpus Christi → Lubbock")
    print("=" * 60)
    print("Testing how well RL irrigation policies generalize to different locations")
    print()
    
    # Load data
    print("📂 Loading data...")
    corpus_data = load_corpus_data()
    lubbock_data = load_lubbock_data()
    
    print(f"✅ Corpus Christi data: {len(corpus_data)} observations")
    print(f"✅ Lubbock data: {len(lubbock_data)} observations")
    
    # Load trained policies
    print("\n🤖 Loading trained policies...")
    policies = load_trained_policies()
    
    if not policies:
        print("❌ No policies loaded. Please train policies first.")
        return
    
    # Define Lubbock treatment types to test
    lubbock_treatments = ['DICT', 'DIEG', 'FIEG', 'FICT']
    
    # Test policy transfer
    all_results = []
    
    for treatment in lubbock_treatments:
        # Map Lubbock treatment to Corpus policy
        if treatment in ['DICT', 'DIEG']:
            policy = policies.get('H_I')
        elif treatment in ['FIEG', 'FICT']:
            policy = policies.get('F_I')
        else:
            policy = None
        
        results = test_policy_transfer(policy, lubbock_data, treatment, corpus_data)
        if results:
            all_results.extend(results)
    
    # Analyze results
    if all_results:
        df_results = analyze_transfer_results(all_results)
        
        # Save results
        output_file = '../outputs/policy_transfer_results.csv'
        df_results.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        
        # Final assessment
        print("\n🎯 POLICY TRANSFER ASSESSMENT:")
        print("=" * 40)
        
        avg_reward = df_results['total_reward'].mean()
        if avg_reward > 0:
            print("✅ Overall: Policies transfer well to Lubbock")
        elif avg_reward > -0.05:
            print("⚠️ Overall: Policies transfer moderately to Lubbock")
        else:
            print("❌ Overall: Policies struggle to transfer to Lubbock")
        
        print(f"📊 Average reward across all treatments: {avg_reward:.3f}")
        
    else:
        print("❌ No results generated. Check data and policies.")

if __name__ == "__main__":
    main() 