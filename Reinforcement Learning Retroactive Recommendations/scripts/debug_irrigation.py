#!/usr/bin/env python3
"""
Debug script to understand why the irrigation training is failing
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))

# Test the reward function directly
def test_reward_function():
    """Test the reward function with different scenarios."""
    print("🔍 TESTING REWARD FUNCTION")
    print("=" * 40)
    
    # Import the environment
    from irrigation_policy_transfer import PlantHealthAwareEnv
    
    # Create dummy data for testing
    test_data = pd.DataFrame({
        'Date': pd.date_range('2025-06-01', periods=10),
        'Plot ID': [404] * 10,
        'Treatment Type': ['H_I'] * 10,
        'Total Soil Moisture': [180, 190, 200, 210, 220, 180, 190, 200, 210, 220],  # Mix of stressed/unstressed
        'ET0 (mm)': [5.0] * 10,
        'Heat Index (F)': [85] * 10,
        'Rainfall (gallons)': [0] * 10,
        'ExG': [0.4, 0.45, 0.5, 0.55, 0.6, 0.4, 0.45, 0.5, 0.55, 0.6],
        'Days_After_Planting': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
        'Kc (Crop Coefficient)': [0.8] * 10
    })
    
    # Test H_I treatment
    env = PlantHealthAwareEnv(test_data, 'H_I')
    
    print(f"H_I Irrigation amounts: {env.irrigation_amounts}")
    print()
    
    # Test different scenarios
    scenarios = [
        {"name": "High Stress + No Irrigation", "soil": 180, "irrigation": 0, "exg": 0.4, "days": 60},
        {"name": "High Stress + Good Irrigation", "soil": 180, "irrigation": 45, "exg": 0.4, "days": 60},
        {"name": "High Stress + Too Much Irrigation", "soil": 180, "irrigation": 75, "exg": 0.4, "days": 60},
        {"name": "Well Watered + No Irrigation", "soil": 220, "irrigation": 0, "exg": 0.5, "days": 60},
        {"name": "Well Watered + Unnecessary Irrigation", "soil": 220, "irrigation": 45, "exg": 0.5, "days": 60},
        {"name": "Moderate Stress + Moderate Irrigation", "soil": 195, "irrigation": 30, "exg": 0.45, "days": 70},
    ]
    
    for scenario in scenarios:
        row = pd.Series({
            'Total Soil Moisture': scenario["soil"],
            'ET0 (mm)': 5.0,
            'Heat Index (F)': 85,
            'Rainfall (gallons)': 0,
            'ExG': scenario["exg"],
            'Days_After_Planting': scenario["days"],
            'Kc (Crop Coefficient)': 0.8
        })
        
        reward = env._calculate_plant_health_reward(scenario["irrigation"], row)
        water_deficit = max(0, 200 - scenario["soil"])
        
        print(f"{scenario['name']:30} | Soil: {scenario['soil']:3} | Deficit: {water_deficit:2.0f} | Irrigation: {scenario['irrigation']:2} | Reward: {reward:6.1f}")
    
    print()

def test_action_space():
    """Test if the action space is working correctly."""
    print("🎯 TESTING ACTION SPACE")
    print("=" * 40)
    
    from irrigation_policy_transfer import PlantHealthAwareEnv
    
    test_data = pd.DataFrame({
        'Date': pd.date_range('2025-06-01', periods=5),
        'Plot ID': [404] * 5,
        'Treatment Type': ['H_I'] * 5,
        'Total Soil Moisture': [200] * 5,
        'ET0 (mm)': [5.0] * 5,
        'Heat Index (F)': [85] * 5,
        'Rainfall (gallons)': [0] * 5,
        'ExG': [0.4] * 5,
        'Days_After_Planting': [60] * 5,
        'Kc (Crop Coefficient)': [0.8] * 5
    })
    
    for treatment in ['H_I', 'F_I', 'R_F']:
        env = PlantHealthAwareEnv(test_data, treatment)
        print(f"{treatment}: {len(env.irrigation_amounts)} actions = {env.irrigation_amounts}")
    
    print()

def test_state_normalization():
    """Test if state normalization is working."""
    print("🔧 TESTING STATE NORMALIZATION") 
    print("=" * 40)
    
    from irrigation_policy_transfer import PlantHealthAwareEnv
    
    test_data = pd.DataFrame({
        'Date': pd.date_range('2025-06-01', periods=5),
        'Plot ID': [404] * 5,
        'Treatment Type': ['H_I'] * 5,
        'Total Soil Moisture': [180, 190, 200, 210, 220],
        'ET0 (mm)': [3.0, 5.0, 7.0, 9.0, 11.0],
        'Heat Index (F)': [75, 80, 85, 90, 95],
        'Rainfall (gallons)': [0, 1, 2, 3, 4],
        'ExG': [0.2, 0.3, 0.4, 0.5, 0.6],
        'Days_After_Planting': [30, 45, 60, 75, 90],
        'Kc (Crop Coefficient)': [0.4, 0.6, 0.8, 1.0, 1.2]
    })
    
    env = PlantHealthAwareEnv(test_data, 'H_I')
    env.reset()
    
    for i in range(5):
        env.current_step = i
        state = env._get_state()
        row = test_data.iloc[i]
        print(f"Row {i}: Soil={row['Total Soil Moisture']:3.0f}, ET0={row['ET0 (mm)']:3.1f}, ExG={row['ExG']:.1f} → State={state}")
    
    print()

def test_simple_training():
    """Test a simple training scenario to see if learning works at all."""
    print("🎓 TESTING SIMPLE TRAINING")
    print("=" * 40)
    
    from irrigation_policy_transfer import PlantHealthAwareEnv
    from stable_baselines3 import PPO
    
    # Create simple test data where irrigation should clearly help
    test_data = pd.DataFrame({
        'Date': pd.date_range('2025-06-01', periods=50),
        'Plot ID': [404] * 50,
        'Treatment Type': ['H_I'] * 50,
        'Total Soil Moisture': [180] * 25 + [220] * 25,  # Half stressed, half good
        'ET0 (mm)': [5.0] * 50,
        'Heat Index (F)': [85] * 50,
        'Rainfall (gallons)': [0] * 50,
        'ExG': [0.4] * 25 + [0.6] * 25,  # Low when stressed, high when not
        'Days_After_Planting': list(range(30, 80)),
        'Kc (Crop Coefficient)': [0.8] * 50
    })
    
    env = PlantHealthAwareEnv(test_data, 'H_I')
    
    # Test manual actions
    print("Testing manual actions:")
    obs, _ = env.reset()
    
    # Try different actions on stressed soil
    for action in range(len(env.irrigation_amounts)):
        env.reset()
        _, reward, _, _, info = env.step(action)
        irrigation = env.irrigation_amounts[action]
        print(f"Action {action} (Irrigation {irrigation:2}): Reward = {reward:6.1f}")
    
    print("\nTraining mini-model for 1000 steps...")
    model = PPO('MlpPolicy', env, learning_rate=1e-3, n_steps=128, verbose=0)
    model.learn(total_timesteps=1000)
    
    # Test predictions
    print("\nTesting trained model predictions:")
    obs, _ = env.reset()
    for i in range(5):
        action, _ = model.predict(obs, deterministic=True)
        irrigation = env.irrigation_amounts[action]
        obs, reward, done, _, info = env.step(action)
        print(f"Step {i}: Action {action} (Irrigation {irrigation:2}), Reward {reward:6.1f}, Soil {info.get('soil_moisture', 'N/A')}")
        if done:
            obs, _ = env.reset()

if __name__ == "__main__":
    print("🚨 IRRIGATION SYSTEM DIAGNOSTIC")
    print("=" * 60)
    print()
    
    test_reward_function()
    test_action_space()
    test_state_normalization()
    test_simple_training()
    
    print("✅ Diagnostic complete!") 