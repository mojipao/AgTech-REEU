#!/usr/bin/env python3
"""
Weather Station Data Integration Script
Integrates 27 years of weather station data to fix overfitting issues
"""

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WeatherStationIntegrator:
    """Integrate weather station data with existing Corpus and Lubbock data"""
    
    def __init__(self):
        self.weather_data = None
        self.corpus_data = None
        self.lubbock_data = None
        self.combined_data = None
        
    def load_weather_station_data(self, data_dir="./Weather Data"):
        """Load all weather station CSV files from 1998-2024"""
        logger.info("Loading weather station data from 1998-2024...")
        
        # Find all CSV files (recursively)
        pattern = os.path.join(data_dir, "**", "686934_27.77_-97.42_*.csv")
        csv_files = glob.glob(pattern, recursive=True)
        
        if not csv_files:
            logger.error(f"No weather station files found in {data_dir}")
            return None
            
        logger.info(f"Found {len(csv_files)} weather station files")
        
        # Load and combine all files
        all_data = []
        
        for file_path in sorted(csv_files):
            try:
                # Extract year from filename
                year = int(file_path.split('_')[-1].replace('.csv', ''))
                
                # Read CSV, skip metadata rows
                df = pd.read_csv(file_path, skiprows=2)
                
                # Add year column
                df['Year'] = year
                
                # Create date column
                df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
                
                # Select relevant columns
                weather_df = df[['Date', 'Temperature', 'Relative Humidity', 'GHI', 'Wind Speed']].copy()
                
                # Convert GHI from W/m² to MJ/m²/day (multiply by 0.0864)
                weather_df['GHI_MJ'] = weather_df['GHI'] * 0.0864
                
                # Calculate daily averages (if hourly data)
                daily_weather = weather_df.groupby('Date').agg({
                    'Temperature': ['mean', 'min', 'max'],
                    'Relative Humidity': 'mean',
                    'GHI_MJ': 'sum',  # Daily total
                    'Wind Speed': 'mean'
                }).reset_index()
                
                # Flatten column names
                daily_weather.columns = ['Date', 'Temp_mean', 'Temp_min', 'Temp_max', 
                                       'RH_mean', 'GHI_daily', 'Wind_mean']
                
                all_data.append(daily_weather)
                
                logger.info(f"Loaded {year}: {len(daily_weather)} days")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                continue
        
        if all_data:
            self.weather_data = pd.concat(all_data, ignore_index=True)
            self.weather_data = self.weather_data.sort_values('Date')
            
            logger.info(f"Total weather station data: {len(self.weather_data)} days")
            logger.info(f"Date range: {self.weather_data['Date'].min()} to {self.weather_data['Date'].max()}")
            
            return self.weather_data
        else:
            logger.error("No weather data loaded")
            return None
    
    def calculate_et0_penman_monteith(self, temp_mean, rh_mean, wind_mean, ghi_daily, lat=27.77):
        """Calculate ET₀ using Penman-Monteith method"""
        # Constants
        lat_rad = np.radians(lat)
        elevation = 16  # meters above sea level
        
        # Atmospheric pressure (kPa)
        p = 101.3 * ((293 - 0.0065 * elevation) / 293)**5.26
        
        # Psychrometric constant (kPa/°C)
        gamma = 0.000665 * p
        
        # Saturation vapor pressure (kPa)
        es = 0.6108 * np.exp(17.27 * temp_mean / (temp_mean + 237.3))
        
        # Actual vapor pressure (kPa)
        ea = es * rh_mean / 100
        
        # Slope of saturation vapor pressure curve (kPa/°C)
        delta = 4098 * es / (temp_mean + 237.3)**2
        
        # Net radiation (MJ/m²/day) - simplified
        albedo = 0.23  # grass reference surface
        rn = (1 - albedo) * ghi_daily
        
        # Soil heat flux (negligible for daily calculations)
        g = 0
        
        # Wind function
        u2 = wind_mean  # wind speed at 2m height
        
        # Penman-Monteith equation
        et0 = (0.408 * delta * (rn - g) + gamma * 900 / (temp_mean + 273) * u2 * (es - ea)) / \
              (delta + gamma * (1 + 0.34 * u2))
        
        return et0
    
    def process_weather_data(self):
        """Process weather data and calculate ET₀ using only Penman-Monteith"""
        if self.weather_data is None:
            logger.error("No weather data loaded")
            return None
            
        logger.info("Processing weather data and calculating ET₀ (Penman-Monteith)...")
        
        self.weather_data['ET0_mm'] = self.calculate_et0_penman_monteith(
            temp_mean=self.weather_data['Temp_mean'],
            rh_mean=self.weather_data['RH_mean'],
            wind_mean=self.weather_data['Wind_mean'],
            ghi_daily=self.weather_data['GHI_daily']
        )
        
        # Add seasonal features
        self.weather_data['Month'] = self.weather_data['Date'].dt.month
        self.weather_data['Day_of_Year'] = self.weather_data['Date'].dt.dayofyear
        self.weather_data['Year'] = self.weather_data['Date'].dt.year
        
        # Calculate heat index (simplified)
        self.weather_data['Heat_Index'] = self.weather_data['Temp_mean'] * 1.8 + 32  # Convert to °F
        
        logger.info("Weather data processing completed")
        logger.info(f"ET₀ range: {self.weather_data['ET0_mm'].min():.2f} - {self.weather_data['ET0_mm'].max():.2f} mm/day")
        
        return self.weather_data
    
    def load_existing_data(self):
        """Load existing Corpus and Lubbock data"""
        logger.info("Loading existing Corpus and Lubbock data...")
        
        # Load Corpus data
        self.corpus_data = pd.read_csv('data/Model Input - Corpus.csv')
        self.corpus_data['Date'] = pd.to_datetime(self.corpus_data['Date'])
        self.corpus_data['Source'] = 'Corpus_Probe'
        
        # Load Lubbock data
        self.lubbock_data = pd.read_csv('data/Model Input - Lubbock-3.csv')
        self.lubbock_data['Date'] = pd.to_datetime(self.lubbock_data['Date'])
        self.lubbock_data['Source'] = 'Lubbock'
        
        logger.info(f"Corpus data: {len(self.corpus_data)} rows")
        logger.info(f"Lubbock data: {len(self.lubbock_data)} rows")
        
        return self.corpus_data, self.lubbock_data
    
    def combine_datasets(self):
        """Combine weather station, Corpus, and Lubbock data"""
        logger.info("Combining all datasets...")
        
        # Prepare weather station data for combination
        weather_combined = self.weather_data[['Date', 'Temp_mean', 'RH_mean', 'GHI_daily', 
                                            'Wind_mean', 'ET0_mm', 'Heat_Index', 'Month', 
                                            'Day_of_Year', 'Year']].copy()
        weather_combined['Source'] = 'Weather_Station'
        weather_combined['Location'] = 'Corpus_Christi'
        
        # Rename columns to match existing data
        weather_combined = weather_combined.rename(columns={
            'Temp_mean': 'Temperature_C',
            'RH_mean': 'Relative_Humidity',
            'GHI_daily': 'Solar_Radiation_MJ',
            'Wind_mean': 'Wind_Speed_ms',
            'ET0_mm': 'ET0_mm',
            'Heat_Index': 'Heat_Index_F'
        })
        
        # Combine all datasets
        self.combined_data = pd.concat([
            weather_combined,
            self.corpus_data,
            self.lubbock_data
        ], ignore_index=True)
        
        self.combined_data = self.combined_data.sort_values('Date')
        
        logger.info(f"Combined dataset: {len(self.combined_data)} total observations")
        logger.info(f"Sources: {self.combined_data['Source'].value_counts().to_dict()}")
        
        return self.combined_data
    
    def save_combined_data(self, output_file='data/combined_weather_dataset.csv'):
        """Save the combined dataset"""
        if self.combined_data is not None:
            self.combined_data.to_csv(output_file, index=False)
            logger.info(f"Combined dataset saved to {output_file}")
            return output_file
        else:
            logger.error("No combined data to save")
            return None

    def calculate_tamu_kc(self, days_after_planting):
        """Calculate Kc using Texas A&M cotton growth stage coefficients"""
        if days_after_planting <= 10:
            return 0.07  # Seeding
        elif 32 <= days_after_planting <= 40:
            return 0.22  # 1st Square
        elif 55 <= days_after_planting <= 60:
            return 0.44  # 1st Bloom
        elif 70 <= days_after_planting <= 90:
            return 1.10  # Max Bloom
        elif 105 <= days_after_planting <= 115:
            return 1.10  # 1st Open
        elif 115 <= days_after_planting <= 125:
            return 0.83  # 25% Open
        elif 135 <= days_after_planting <= 145:
            return 0.44  # 50% Open
        elif 140 <= days_after_planting <= 150:
            return 0.44  # 95% Open
        elif 140 <= days_after_planting <= 150:
            return 0.10  # Pick
        else:
            # Linear interpolation for gaps
            if 11 <= days_after_planting <= 31:
                return 0.07 + (0.22 - 0.07) * (days_after_planting - 10) / 21
            elif 41 <= days_after_planting <= 54:
                return 0.22 + (0.44 - 0.22) * (days_after_planting - 40) / 14
            elif 61 <= days_after_planting <= 69:
                return 0.44 + (1.10 - 0.44) * (days_after_planting - 60) / 9
            elif 91 <= days_after_planting <= 104:
                return 1.10 + (1.10 - 1.10) * (days_after_planting - 90) / 14  # Stay at 1.10
            elif 126 <= days_after_planting <= 134:
                return 0.83 + (0.44 - 0.83) * (days_after_planting - 125) / 9
            elif days_after_planting > 150:
                return 0.10  # After picking
            else:
                return 0.07  # Default to seeding

    def save_corpus_compatible_csv(self, output_file='data/Model Input - Corpus+Weather.csv'):
        """Save combined data in the same format as Model Input - Corpus.csv"""
        # Define required columns
        required_cols = [
            'Date',
            'Plot ID',
            'Treatment Type',
            'ExG',
            'Total Soil Moisture',
            'Irrigation Added (gallons)',
            'Rainfall (gallons)',
            'ET0 (mm)',
            'Heat Index (F)',
            'Kc (Crop Coefficient)'
        ]
        
        # Prepare weather station data
        ws = self.weather_data.copy()
        ws['Date'] = ws['Date'].dt.strftime('%Y-%m-%d')
        ws['Plot ID'] = 'WeatherStation'
        ws['Treatment Type'] = 'Synthetic'
        ws['ExG'] = np.nan
        ws['Total Soil Moisture'] = np.nan
        ws['Irrigation Added (gallons)'] = 0.0
        ws['Rainfall (gallons)'] = np.nan  # You may want to estimate this from GHI or set to 0
        ws['ET0 (mm)'] = ws['ET0_mm']
        ws['Heat Index (F)'] = ws['Heat_Index']
        
        # Calculate Kc for weather station data using Texas A&M coefficients
        # Assume planting date of April 3 for each year
        ws['Kc (Crop Coefficient)'] = ws.apply(
            lambda row: self.calculate_tamu_kc(
                (pd.to_datetime(row['Date']) - pd.to_datetime(f"{row['Year']}-04-03")).days
            ), axis=1
        )
        
        ws = ws[required_cols]
        
        # Prepare original Corpus data
        corpus = self.corpus_data.copy()
        corpus['Date'] = pd.to_datetime(corpus['Date']).dt.strftime('%Y-%m-%d')
        corpus = corpus[required_cols]
        
        # Combine
        combined = pd.concat([corpus, ws], ignore_index=True)
        combined = combined.sort_values('Date')
        
        combined.to_csv(output_file, index=False)
        logger.info(f"Corpus-compatible combined dataset saved to {output_file}")
        return output_file

def main():
    """Main execution function"""
    logger.info("Starting weather station data integration...")
    
    integrator = WeatherStationIntegrator()
    
    # Load weather station data
    weather_data = integrator.load_weather_station_data()
    if weather_data is None:
        logger.error("Failed to load weather station data")
        return
    
    # Process weather data
    processed_weather = integrator.process_weather_data()
    if processed_weather is None:
        logger.error("Failed to process weather data")
        return
    
    # Load existing data
    corpus_data, lubbock_data = integrator.load_existing_data()
    
    # Combine datasets
    combined_data = integrator.combine_datasets()
    
    # Save combined dataset
    output_file = integrator.save_combined_data()
    
    # Save corpus-compatible dataset
    integrator.save_corpus_compatible_csv()

    logger.info("Weather station integration completed successfully!")
    logger.info(f"Final dataset size: {len(combined_data)} observations")
    logger.info(f"Date range: {combined_data['Date'].min()} to {combined_data['Date'].max()}")

if __name__ == "__main__":
    main() 