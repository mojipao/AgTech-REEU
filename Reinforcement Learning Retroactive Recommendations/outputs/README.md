# RL-Based Irrigation Policy Transfer: Rationale & Method

## Overview
This document describes the rationale and methodology for generating scientifically defensible, field-ready irrigation recommendations for Corpus Christi cotton using reinforcement learning (RL) policy transfer from Lubbock field data.

---

## Rationale
- **Challenge:** Traditional ML models for irrigation can overfit and produce unrealistic recommendations, especially when trained on limited or idealized data.
- **Goal:** Develop a robust, data-driven RL system that learns irrigation timing and amount in a realistic, uncertain environment, and produces recommendations that are both scientifically sound and practically actionable.
- **Approach:**
  - Use 27 years of real weather station data and field data from both Lubbock and Corpus Christi.
  - Train RL agents in a synthetic environment that mimics real-world uncertainty.
  - Transfer learned policies from Lubbock (where more data is available) to Corpus Christi, with climate and plot size adjustments.
  - Calibrate (scale) RL recommendations to match field-applied seasonal irrigation totals, ensuring agronomic realism.

---

## Data Sources
- **Weather:** 27 years (1998–2024) of weather station data for Corpus Christi and Lubbock.
- **Field Data:**
  - Lubbock: 1,288 observations, multiple irrigation treatments (DICT, DIEG, FICT, FIEG).
  - Corpus Christi: 28 days of growing season, 3 treatments (F_I, H_I, R_F).
- **Plant Health:** ExG (Excess Green) index used as a proxy for cotton health.

---

## RL Environment & Policy Transfer
- **State:** Soil moisture, ET₀, heat index, rainfall, ExG, days after planting, water deficit, crop coefficient (Kc).
- **Action Space:** Discrete irrigation amounts, mapped to Lubbock field practices.
- **Reward:** Based on plant health (ExG), water stress, and water use efficiency.
- **Policy Transfer:**
  - Lubbock DIEG → Corpus H_I (half irrigation, ExG-based)
  - Lubbock FIEG → Corpus F_I (full irrigation, ExG-based)
  - R_F (rainfed) always 0
- **Climate & Plot Scaling:** Adjust for differences in plot size and climate between Lubbock and Corpus Christi.

---

## Scaling & Calibration
- **Why Scale?** RL agents often learn relative patterns, not absolute magnitudes. To ensure recommendations are agronomically realistic, we scale the RL output so that the total seasonal irrigation matches field-applied totals:
  - F_I (100% ET): 10.51 inches/season (2,900 gallons for 443.5 sq ft)
  - H_I (50% ET): 9.76 inches/season (2,690 gallons for 443.5 sq ft)
- **How?**
  - For each treatment, sum the RL recommendations over the season.
  - Calculate the scaling factor: (target gallons) / (total RL gallons).
  - Multiply each RL recommendation by this factor to get the scaled, field-ready value.

---

## Exact Formulas Used

### 1. **Conversion from Inches to Gallons**
For a plot of area $A$ (sq ft):

- **Gallons per inch:**
  $$
  \text{Gallons per inch} = 0.623 \times A
  $$
  For this project: $A = 443.5$ sq ft, so:
  $$
  \text{Gallons per inch} = 0.623 \times 443.5 \approx 276
  $$

### 2. **Target Seasonal Irrigation (Gallons)**
- For each treatment:
  $$
  \text{Target gallons} = \text{Target inches (from field data)} \times \text{Gallons per inch}
  $$
  Example for F_I (100% ET):
  $$
  10.51 \text{ in} \times 276 \text{ gal/in} = 2,900 \text{ gal}
  $$

### 3. **Scaling Factor**
- For each treatment:
  $$
  \text{Scaling factor} = \frac{\text{Target gallons}}{\text{Total RL recommended gallons}}
  $$

### 4. **Scaled RL Recommendation**
- For each day and treatment:
  $$
  \text{Scaled Irrigation (gallons)} = \text{RL Recommended Irrigation (gallons)} \times \text{Scaling factor}
  $$

### 5. **Weekly Aggregation (for reporting)**
- For each week:
  $$
  \text{Weekly total} = \sum_{\text{days in week}} \text{Scaled Irrigation (gallons)}
  $$

---

## Practical Interpretation
- **Daily RL output is often “spiky”** (large events on a few days). For field application and reporting, we aggregate by week:
  - **Weekly total = sum of scaled irrigation for each week.**
  - This matches real-world practice and is easier to interpret.
- **The RL agent’s “decision logic”** is preserved: irrigation is concentrated in response to water stress, not applied uniformly.

---

## Limitations & Future Work
- RL output pattern depends on reward function and environment; further tuning could encourage more frequent, smaller irrigations.
- Scaling ensures agronomic realism, but the timing pattern is determined by the RL agent.
- Future work: test in field trials, adapt to other crops/regions, refine reward for smoother output.

---

## Output Files
- **Main RL transfer output:** `lubbock_to_corpus_transfer_results_DIEG_FIEG_scaled.csv` (saved in this outputs/ directory)
- All outputs are saved here using robust, script-relative paths, so you can run the script from any directory.

---

## Contact & Reproducibility
- All code and data sources are documented in this repository.
- For questions or collaboration, contact the project maintainer. 