## 2. Data Generation Methodology

### 2.1 Weather Generation

#### 2.1.1 Stochastic Rainfall Generation
The system uses a **stochastic weather generator** to create realistic rainfall patterns for Corpus Christi, replacing the previous deterministic approach:

**Key Features:**
- **Seasonal Probability Models:** Different rainfall probabilities by season (summer thunderstorms: 25%, winter dry: 10%)
- **Weather Persistence:** If it rained yesterday, 50% higher chance of rain today
- **Realistic Rainfall Distributions:**
  - Light rain (60%): 0.1-0.5 inches
  - Moderate rain (25%): 0.5-1.5 inches  
  - Heavy rain (10%): 1.5-3.0 inches
  - Extreme rain (5%): 3.0-6.0 inches
- **Corpus Christi Climate:** Based on NOAA data (~32 inches annually, summer peak)

**Why This Approach:**
- **Eliminates deterministic copying** of single-year Lubbock patterns
- **Creates realistic weather variability** for robust RL training
- **Maintains climatological accuracy** while allowing day-to-day variation
- **Enables proper validation** against real weather scenarios

#### 2.1.2 Environmental Variable Integration
Other weather variables (ET0, heat index) are generated using trained ML models that incorporate:
- Seasonal patterns from date features
- Rainfall-dependent relationships
- Realistic noise and variability

### 2.2 Previous Approach (Deprecated)
The original system copied exact Lubbock rainfall patterns day-by-day, which created:
- **No weather variability** (same rainfall every year for each date)
- **Invalid comparisons** with real Corpus Christi conditions
- **Unrealistic deterministic weather** that doesn't exist in nature 