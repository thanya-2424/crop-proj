"""
AI Agriculture Advisor - Data Generator
Creates realistic agricultural datasets for optimal ML performance
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_realistic_dataset():
    """Generate comprehensive, realistic agricultural dataset"""
    
    print("🌾 Generating realistic agricultural dataset...")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate 6 years of daily data (2019-2024)
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    print(f"📅 Generated {len(dates):,} potential dates from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Sample dates with seasonal bias (more data during growing seasons)
    n_samples = min(8000, len(dates))  # Ensure we don't exceed available dates
    print(f"📊 Will generate {n_samples:,} sample records")
    
    seasonal_weights = np.ones(len(dates))
    
    for i, date in enumerate(dates):
        month = date.month
        if month in [6, 7, 8, 9, 10]:  # Kharif season
            seasonal_weights[i] = 2.0
        elif month in [10, 11, 12, 1, 2, 3]:  # Rabi season
            seasonal_weights[i] = 1.8
        elif month in [3, 4, 5]:  # Zaid season
            seasonal_weights[i] = 1.5
    
    seasonal_weights = seasonal_weights / seasonal_weights.sum()
    sampled_indices = np.random.choice(len(dates), n_samples, p=seasonal_weights, replace=False)
    sampled_dates = sorted(dates[sampled_indices])
    
    print(f"✅ Sampled {len(sampled_dates):,} dates with seasonal bias")
    
    # Create dataset
    data = []
    
    for date in sampled_dates:
        # Generate multiple records per date for different crops and locations
        records_per_date = np.random.randint(1, 4)  # 1-3 records per date
        
        for _ in range(records_per_date):
            
            # Crop selection with realistic distribution
            crop = np.random.choice(['Rice', 'Wheat', 'Corn', 'Soybeans', 'Cotton'], 
                                  p=[0.25, 0.20, 0.20, 0.15, 0.20])
            
            # Location selection
            location = np.random.choice(['Punjab', 'Haryana', 'Uttar Pradesh', 'Madhya Pradesh', 
                                       'Rajasthan', 'Gujarat', 'Maharashtra', 'Karnataka'])
            
            # Generate realistic weather data
            month = date.month
            
            # Temperature with seasonal variation
            base_temp = 25 + 10 * np.sin(2 * np.pi * (month - 6) / 12)
            if location in ['Rajasthan', 'Gujarat']:
                base_temp += 3  # Hotter regions
            elif location in ['Karnataka', 'Maharashtra']:
                base_temp -= 2  # Cooler regions
            
            temperature = np.random.normal(base_temp, 3)
            temperature = np.clip(temperature, 10, 45)
            
            # Humidity with seasonal variation
            base_humidity = 60 + 20 * np.sin(2 * np.pi * (month - 6) / 12)
            if month in [6, 7, 8, 9]:  # Monsoon
                base_humidity += 15
            humidity = np.random.normal(base_humidity, 8)
            humidity = np.clip(humidity, 25, 90)
            
            # Rainfall
            if month in [6, 7, 8, 9]:  # Monsoon
                rainfall = np.random.exponential(8)
            elif month in [10, 11, 12, 1, 2, 3]:  # Winter
                rainfall = np.random.exponential(2)
            else:  # Summer
                rainfall = np.random.exponential(1)
            rainfall = np.clip(rainfall, 0, 40)
            
            # Soil parameters
            soil_moisture = 50 + (rainfall * 0.6) + np.random.normal(0, 8)
            soil_moisture = np.clip(soil_moisture, 20, 85)
            
            ph_level = np.random.normal(7.0, 0.5)
            ph_level = np.clip(ph_level, 6.0, 8.5)
            
            nitrogen = np.random.normal(200, 50)
            phosphorus = np.random.normal(100, 30)
            potassium = np.random.normal(150, 40)
            
            # Generate realistic price based on multiple factors
            base_prices = {'Rice': 2800, 'Wheat': 2200, 'Corn': 1800, 'Soybeans': 4500, 'Cotton': 6500}
            base_price = base_prices[crop]
            
            # Seasonal price variations
            seasonal_factor = 1.0
            if month in [9, 10, 11, 3, 4, 5]:  # Harvest months
                seasonal_factor = 0.85  # Lower prices during harvest
            elif month in [6, 7, 10, 11]:  # Planting months
                seasonal_factor = 1.15  # Higher prices during planting
            
            # Location factor
            location_factor = 1.0
            if location in ['Punjab', 'Haryana']:
                location_factor = 0.95  # Major producing states
            elif location in ['Rajasthan', 'Gujarat']:
                location_factor = 1.05  # Minor producing states
            
            # Weather impact
            weather_factor = 1.0
            if temperature > 35 or temperature < 15:
                weather_factor = 1.1  # Higher prices during stress
            
            # Final price calculation
            price = base_price * seasonal_factor * location_factor * weather_factor
            price *= np.random.normal(1, 0.15)  # Add volatility
            price = np.clip(price, base_price * 0.5, base_price * 2.0)
            
            # Yield prediction
            base_yield = 100
            if location in ['Punjab', 'Haryana']:
                base_yield *= 1.2
            elif location in ['Rajasthan']:
                base_yield *= 0.8
            
            # Weather impact on yield
            if 20 <= temperature <= 30 and 50 <= humidity <= 70:
                yield_factor = 1.1  # Optimal conditions
            elif temperature > 35 or temperature < 15:
                yield_factor = 0.8  # Stress conditions
            else:
                yield_factor = 1.0
            
            yield_value = base_yield * yield_factor * np.random.normal(1, 0.1)
            yield_value = np.clip(yield_value, 50, 150)
            
            # Create record
            record = {
                'date': date,
                'crop_type': crop,
                'location': location,
                'temperature': round(temperature, 1),
                'humidity': round(humidity, 1),
                'rainfall': round(rainfall, 1),
                'soil_moisture': round(soil_moisture, 1),
                'ph_level': round(ph_level, 2),
                'nitrogen': round(nitrogen, 0),
                'phosphorus': round(phosphorus, 0),
                'potassium': round(potassium, 0),
                'yield': round(yield_value, 1),
                'price': round(price, 2)
            }
            
            data.append(record)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    print(f"📊 Created {len(df):,} records with {len(df.columns)} features")
    
    # Add derived features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['season'] = df['month'].map(lambda x: (x % 12 + 3) // 3)
    df['quarter'] = df['date'].dt.quarter
    
    # Weather stress indicators
    df['temp_stress'] = np.where((df['temperature'] < 15) | (df['temperature'] > 35), 1, 0)
    df['humidity_stress'] = np.where((df['humidity'] < 40) | (df['humidity'] > 80), 1, 0)
    df['weather_stress_score'] = df['temp_stress'] + df['humidity_stress']
    
    # Soil health score
    df['soil_health_score'] = (
        (df['ph_level'] - 6.5) / 2.0 +
        (df['nitrogen'] - 150) / 100 +
        (df['phosphorus'] - 75) / 75 +
        (df['potassium'] - 125) / 75
    ) / 4.0
    df['soil_health_score'] = np.clip(df['soil_health_score'], 0, 1)
    
    # Market indicators
    df['market_demand'] = np.where(
        df['month'].isin([6, 7, 10, 11]), 'high',
        np.where(df['month'].isin([9, 10, 3, 4]), 'low', 'medium')
    )
    
    df['price_trend'] = np.where(
        df['price'] > df.groupby('crop_type')['price'].transform('mean'), 'rising', 'falling'
    )
    
    # Ensure data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')
        print("📁 Created data directory")
    
    # Save main dataset
    output_path = 'data/realistic_crop_data.csv'
    df.to_csv(output_path, index=False)
    
    print(f"✅ Dataset generated successfully!")
    print(f"📊 Final Shape: {df.shape}")
    print(f"📁 Saved to: {output_path}")
    
    # Display statistics
    display_dataset_stats(df)
    
    return df

def display_dataset_stats(df):
    """Display comprehensive dataset statistics"""
    print("\n" + "="*60)
    print("📊 DATASET STATISTICS")
    print("="*60)
    
    print(f"📈 Total Records: {len(df):,}")
    print(f"🔍 Total Features: {len(df.columns)}")
    print(f"📅 Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    
    print(f"\n🌾 Crop Distribution:")
    crop_counts = df['crop_type'].value_counts()
    for crop, count in crop_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {crop}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n📍 Top Locations:")
    location_counts = df['location'].value_counts().head(6)
    for location, count in location_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {location}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n💰 Price Statistics by Crop:")
    for crop in df['crop_type'].unique():
        crop_data = df[df['crop_type'] == crop]
        print(f"   {crop}: ₹{crop_data['price'].mean():.0f} ± ₹{crop_data['price'].std():.0f}")
    
    print(f"\n🌤️ Weather Statistics:")
    print(f"   Temperature: {df['temperature'].mean():.1f}°C ± {df['temperature'].std():.1f}°C")
    print(f"   Humidity: {df['humidity'].mean():.1f}% ± {df['humidity'].std():.1f}%")
    print(f"   Rainfall: {df['rainfall'].mean():.1f}mm ± {df['rainfall'].std():.1f}mm")
    
    print(f"\n🤖 ML Readiness:")
    print(f"   Feature Count: {len(df.columns)} - ✅ Excellent")
    print(f"   Sample Size: {len(df):,} - ✅ Excellent")
    print(f"   Price Variation: {df['price'].std() / df['price'].mean():.3f} - ✅ Good")
    
    print("="*60)

if __name__ == "__main__":
    # Generate the dataset
    df = generate_realistic_dataset()
    
    print("\n🎉 Data generation complete!")
    print("🚀 Next steps:")
    print("   1. Run the Streamlit app: python run_app.py")
    print("   2. The ML pipeline will automatically use this dataset")
    print("   3. Models will train on realistic data for accurate predictions") 