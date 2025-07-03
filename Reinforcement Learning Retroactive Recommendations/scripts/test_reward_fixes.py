#!/usr/bin/env python3
"""
Validation script to test that all three key fixes are working:
1. Treatment differentiation (F_I should irrigate more than H_I)
2. Irrigation amounts properly calibrated to reduce water stress
3. Reward function promotes plant health
"""

import pandas as pd
import numpy as np
from irrigation_policy_transfer import PlantHealthAwareEnv

def test_treatment_differentiation():
    """Test that treatments behave appropriately for their agricultural purpose."""
    print("🧪 TESTING TREATMENT DIFFERENTIATION")
    print("=" * 50)
    
    # Test 1: High stress scenario - F_I should excel at high irrigation
    test_data_high_stress = pd.DataFrame([{
        'Date': '2025-07-15',
        'Plot ID': 404,
        'Total Soil Moisture': 180,  # Very high stress
        'ET0 (mm)': 8.0,
        'Heat Index (F)': 95,
        'Rainfall (gallons)': 0,
        'ExG': 0.3,  # Stressed plant
        'Kc (Crop Coefficient)': 0.9,
        'Days_After_Planting': 70
    }])
    
    # Test 2: Low stress scenario - H_I should avoid overwatering
    test_data_low_stress = pd.DataFrame([{
        'Date': '2025-07-15',
        'Plot ID': 404,
        'Total Soil Moisture': 210,  # Low stress
        'ET0 (mm)': 4.0,
        'Heat Index (F)': 80,
        'Rainfall (gallons)': 5,
        'ExG': 0.6,  # Healthy plant
        'Kc (Crop Coefficient)': 0.7,
        'Days_After_Planting': 70
    }])
    
    h_i_env = PlantHealthAwareEnv(test_data_high_stress, 'H_I')
    f_i_env = PlantHealthAwareEnv(test_data_high_stress, 'F_I')
    
    print("HIGH STRESS SCENARIO (soil=180, very dry):")
    print(f"{'Irrigation':<12} {'H_I Reward':<12} {'F_I Reward':<12} {'F_I Better?'}")
    print("-" * 55)
    
    # In high stress, F_I should excel at high irrigation amounts
    high_stress_f_i_advantage = 0
    high_irrigation_amounts = [75, 100, 125]
    
    for amount in [25, 50, 75, 100, 125]:
        h_i_reward = h_i_env._calculate_plant_health_reward(amount, test_data_high_stress.iloc[0])
        f_i_reward = f_i_env._calculate_plant_health_reward(amount, test_data_high_stress.iloc[0])
        f_i_better = f_i_reward > h_i_reward
        if amount in high_irrigation_amounts and f_i_better:
            high_stress_f_i_advantage += 1
        
        print(f"{amount:<12} {h_i_reward:<12.1f} {f_i_reward:<12.1f} {'✅' if f_i_better else '❌'}")
    
    # Update environments for low stress test
    h_i_env_low = PlantHealthAwareEnv(test_data_low_stress, 'H_I') 
    f_i_env_low = PlantHealthAwareEnv(test_data_low_stress, 'F_I')
    
    print(f"\nLOW STRESS SCENARIO (soil=210, well watered):")
    print(f"{'Irrigation':<12} {'H_I Reward':<12} {'F_I Reward':<12} {'H_I Better?'}")
    print("-" * 55)
    
    # In low stress, both should prefer minimal irrigation (avoid waste)
    low_stress_appropriate = 0
    for amount in [0, 25, 50]:
        h_i_reward = h_i_env_low._calculate_plant_health_reward(amount, test_data_low_stress.iloc[0])
        f_i_reward = f_i_env_low._calculate_plant_health_reward(amount, test_data_low_stress.iloc[0])
        # Both should get positive rewards for not overwatering
        both_positive = h_i_reward > 0 and f_i_reward > 0
        if both_positive:
            low_stress_appropriate += 1
        
        print(f"{amount:<12} {h_i_reward:<12.1f} {f_i_reward:<12.1f} {'✅' if both_positive else '❌'}")
    
    print(f"\n✅ Agricultural Logic Test:")
    print(f"   F_I excels in high stress + high irrigation: {high_stress_f_i_advantage}/3")
    print(f"   Both avoid overwatering in low stress: {low_stress_appropriate}/3")
    
    return high_stress_f_i_advantage >= 2 and low_stress_appropriate >= 2

def test_water_stress_calibration():
    """Test that irrigation amounts actually reduce water stress classification."""
    print("\n🧪 TESTING WATER STRESS CALIBRATION")
    print("=" * 50)
    
    # Test different soil moisture levels
    stress_scenarios = [
        (180, 'High stress (180)', 50),
        (195, 'Medium stress (195)', 30), 
        (205, 'Low stress (205)', 0)
    ]
    
    test_data = pd.DataFrame([{
        'Date': '2025-07-15',
        'Plot ID': 404,
        'Total Soil Moisture': 180,  # Will be updated
        'ET0 (mm)': 6.0,
        'Heat Index (F)': 90,
        'Rainfall (gallons)': 0,
        'ExG': 0.4,
        'Kc (Crop Coefficient)': 0.8,
        'Days_After_Planting': 70
    }])
    
    f_i_env = PlantHealthAwareEnv(test_data, 'F_I')
    
    print(f"{'Scenario':<20} {'No Irrigation':<15} {'With Irrigation':<15} {'Improvement?'}")
    print("-" * 65)
    
    improvements = 0
    for soil_moisture, description, irrigation_amount in stress_scenarios:
        test_data.iloc[0, test_data.columns.get_loc('Total Soil Moisture')] = soil_moisture
        
        no_irrigation_reward = f_i_env._calculate_plant_health_reward(0, test_data.iloc[0])
        with_irrigation_reward = f_i_env._calculate_plant_health_reward(irrigation_amount, test_data.iloc[0])
        
        improved = with_irrigation_reward > no_irrigation_reward
        if improved:
            improvements += 1
        
        print(f"{description:<20} {no_irrigation_reward:<15.1f} {with_irrigation_reward:<15.1f} {'✅' if improved else '❌'}")
    
    print(f"\n✅ Irrigation improved rewards in {improvements}/{len(stress_scenarios)} scenarios")
    return improvements >= 2  # Should improve in high and medium stress

def test_plant_health_promotion():
    """Test that the reward function promotes plant health outcomes."""
    print("\n🧪 TESTING PLANT HEALTH PROMOTION")
    print("=" * 50)
    
    # Test scenarios with different ExG levels
    health_scenarios = [
        (0.2, 'Poor health (0.2)'),
        (0.4, 'Moderate health (0.4)'),
        (0.6, 'Good health (0.6)')
    ]
    
    test_data = pd.DataFrame([{
        'Date': '2025-07-15',
        'Plot ID': 404,
        'Total Soil Moisture': 185,  # Stressed
        'ET0 (mm)': 6.0,
        'Heat Index (F)': 90,
        'Rainfall (gallons)': 0,
        'ExG': 0.4,  # Will be updated
        'Kc (Crop Coefficient)': 0.8,
        'Days_After_Planting': 70
    }])
    
    f_i_env = PlantHealthAwareEnv(test_data, 'F_I')
    irrigation_amount = 75  # Good irrigation amount
    
    print(f"{'Health Level':<20} {'Reward':<10} {'Expected'}")
    print("-" * 40)
    
    rewards = []
    for exg, description in health_scenarios:
        test_data.iloc[0, test_data.columns.get_loc('ExG')] = exg
        reward = f_i_env._calculate_plant_health_reward(irrigation_amount, test_data.iloc[0])
        rewards.append(reward)
        print(f"{description:<20} {reward:<10.1f} {'Higher is better'}")
    
    # Check if rewards increase with better plant health
    health_promotion = rewards[2] > rewards[1] > rewards[0]
    print(f"\n✅ Rewards increase with plant health: {health_promotion}")
    return health_promotion

def test_action_space_adequacy():
    """Test that new irrigation amounts are adequate for different treatments."""
    print("\n🧪 TESTING ACTION SPACE ADEQUACY")
    print("=" * 50)
    
    test_data = pd.DataFrame([{
        'Date': '2025-07-15',
        'Plot ID': 404,
        'Total Soil Moisture': 180,  # High stress
        'ET0 (mm)': 6.0,
        'Heat Index (F)': 90,
        'Rainfall (gallons)': 0,
        'ExG': 0.4,
        'Kc (Crop Coefficient)': 0.8,
        'Days_After_Planting': 70
    }])
    
    h_i_env = PlantHealthAwareEnv(test_data, 'H_I')
    f_i_env = PlantHealthAwareEnv(test_data, 'F_I')
    
    print(f"H_I action space: {h_i_env.irrigation_amounts}")
    print(f"F_I action space: {f_i_env.irrigation_amounts}")
    print()
    
    # Check that F_I has higher maximum
    f_i_max = max(f_i_env.irrigation_amounts)
    h_i_max = max(h_i_env.irrigation_amounts)
    
    print(f"F_I max irrigation: {f_i_max} gallons")
    print(f"H_I max irrigation: {h_i_max} gallons") 
    print(f"F_I max > H_I max: {'✅' if f_i_max > h_i_max else '❌'}")
    
    return f_i_max > h_i_max

def main():
    """Run all validation tests."""
    print("🔬 REWARD FUNCTION VALIDATION SUITE")
    print("=" * 60)
    print("Testing agricultural logic and fixes:")
    print("1. Treatment appropriateness (not just F_I > H_I)")
    print("2. Water stress calibration") 
    print("3. Plant health promotion")
    print()
    
    # Run all tests
    test1_pass = test_treatment_differentiation()
    test2_pass = test_water_stress_calibration()
    test3_pass = test_plant_health_promotion()
    test4_pass = test_action_space_adequacy()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"✅ Treatment appropriateness: {'PASS' if test1_pass else 'FAIL'}")
    print(f"✅ Water stress calibration: {'PASS' if test2_pass else 'FAIL'}")
    print(f"✅ Plant health promotion: {'PASS' if test3_pass else 'FAIL'}")
    print(f"✅ Action space adequacy: {'PASS' if test4_pass else 'FAIL'}")
    
    all_pass = all([test1_pass, test2_pass, test3_pass, test4_pass])
    print(f"\n🏆 OVERALL: {'ALL TESTS PASSED! 🎉' if all_pass else 'SOME TESTS FAILED ❌'}")
    
    if all_pass:
        print("\n✅ The reward function demonstrates proper agricultural logic!")
        print("✅ F_I excels under high stress conditions")
        print("✅ Both treatments avoid overwatering when appropriate")
        print("Ready to retrain the PPO models with improved reward function.")
    else:
        print("\n❌ Some agricultural logic needs refinement before retraining.")

if __name__ == "__main__":
    main() 