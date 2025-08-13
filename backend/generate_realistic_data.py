"""
AI Agriculture Advisor - Realistic Data Generator
This script generates comprehensive, realistic agricultural datasets that work optimally with the ML pipeline.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RealisticDataGenerator:
    """
    Generates realistic agricultural datasets with proper seasonal patterns and price variations
    """
    
    def __init__(self, data_dir="../data"):
        self.data_dir = data_dir
        self.ensure_data_directory()
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Define realistic crop parameters
        self.crop_parameters = {
            'Rice': {
                'base_price': 2800,
                'price_volatility': 0.15,
                'optimal_temp': (22, 32),
                'optimal_humidity': (60, 80),
                'water_requirement': 'high',
                'growing_season': 'kharif',
                'harvest_months': [9, 10, 11],
                'planting_months': [6, 7]
            },
            'Wheat': {
                'base_price': 2200,
                'price_volatility': 0.12,
                'optimal_temp': (15, 25),
                'optimal_humidity': (40, 60),
                'water_requirement': 'medium',
                'growing_season': 'rabi',
                'harvest_months': [3, 4, 5],
                'planting_months': [10, 11]
            },
            'Corn': {
                'base_price': 1800,
                'price_volatility': 0.18,
                'optimal_temp': (18, 30),
                'optimal_humidity': (50, 70),
                'water_requirement': 'high',
                'growing_season': 'kharif',
                'harvest_months': [9, 10],
                'planting_months': [6, 7]
            },
            'Soybeans': {
                'base_price': 4500,
                'price_volatility': 0.20,
                'optimal_temp': (20, 30),
                'optimal_humidity': (50, 70),
                'water_requirement': 'medium',
                'growing_season': 'kharif',
                'harvest_months': [9, 10],
                'planting_months': [6, 7]
            },
            'Cotton': {
                'base_price': 6500,
                'price_volatility': 0.25,
                'optimal_temp': (25, 35),
                'optimal_humidity': (40, 60),
                'water_requirement': 'medium_high',
                'growing_season': 'kharif',
                'harvest_months': [10, 11, 12],
                'planting_months': [5, 6]
            }
        }
        
        # Define location-specific parameters
        self.location_parameters = {
            'Punjab': {
                'climate': 'semi_arid',
                'soil_type': 'alluvial',
                'irrigation': 'high',
                'base_yield_multiplier': 1.2,
                'temperature_range': (15, 40),
                'humidity_range': (30, 75)
            },
            'Haryana': {
                'climate': 'semi_arid',
                'soil_type': 'alluvial',
                'irrigation': 'high',
                'base_yield_multiplier': 1.1,
                'temperature_range': (15, 42),
                'humidity_range': (25, 70)
            },
            'Uttar Pradesh': {
                'climate': 'humid_subtropical',
                'soil_type': 'alluvial',
                'irrigation': 'medium',
                'base_yield_multiplier': 1.0,
                'temperature_range': (12, 45),
                'humidity_range': (40, 85)
            },
            'Madhya Pradesh': {
                'climate': 'tropical_savanna',
                'soil_type': 'black',
                'irrigation': 'medium',
                'base_yield_multiplier': 0.9,
                'temperature_range': (18, 43),
                'humidity_range': (35, 80)
            },
            'Rajasthan': {
                'climate': 'arid',
                'soil_type': 'desert',
                'irrigation': 'low',
                'base_yield_multiplier': 0.7,
                'temperature_range': (20, 48),
                'humidity_range': (20, 60)
            },
            'Gujarat': {
                'climate': 'semi_arid',
                'soil_type': 'alluvial',
                'irrigation': 'medium',
                'base_yield_multiplier': 0.95,
                'temperature_range': (18, 42),
                'humidity_range': (30, 75)
            },
            'Maharashtra': {
                'climate': 'tropical_savanna',
                'soil_type': 'black',
                'irrigation': 'medium',
                'base_yield_multiplier': 0.9,
                'temperature_range': (20, 40),
                'humidity_range': (45, 85)
            },
            'Karnataka': {
                'climate': 'tropical_savanna',
                'soil_type': 'red',
                'irrigation': 'medium',
                'base_yield_multiplier': 0.85,
                'temperature_range': (18, 38),
                'humidity_range': (50, 80)
            },
            'Tamil Nadu': {
                'climate': 'tropical_savanna',
                'soil_type': 'red',
                'irrigation': 'medium',
                'base_yield_multiplier': 0.8,
                'temperature_range': (22, 38),
                'humidity_range': (55, 85)
            },
            'Andhra Pradesh': {
                'climate': 'tropical_savanna',
                'soil_type': 'red',
                'irrigation': 'medium',
                'base_yield_multiplier': 0.9,
                'temperature_range': (20, 40),
                'humidity_range': (50, 80)
            }
        }
    
    def ensure_data_directory(self):
        """Ensure data directory exists"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created data directory: {self.data_dir}")
    
    def generate_realistic_dataset(self, filename="realistic_crop_data.csv", n_samples=5000):
        """
        Generate a comprehensive, realistic dataset for optimal ML performance
        
        Args:
            filename: Name of the output file
            n_samples: Number of sample records to generate
        """
        print(f"🌾 Generating realistic agricultural dataset with {n_samples:,} records...")
        
        # Generate date range (6 years of data for better ML training)
        start_date = datetime(2018, 1, 1)
        end_date = datetime(2024, 12, 31)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Sample dates with seasonal bias
        seasonal_dates = self._generate_seasonal_dates(date_range, n_samples)
        
        # Generate comprehensive dataset
        dataset = self._generate_core_features(seasonal_dates, n_samples)
        
        # Add derived features
        dataset = self._add_derived_features(dataset)
        
        # Add realistic price variations
        dataset = self._add_realistic_prices(dataset)
        
        # Add yield predictions
        dataset = self._add_yield_predictions(dataset)
        
        # Add market factors
        dataset = self._add_market_factors(dataset)
        
        # Clean and validate data
        dataset = self._clean_dataset(dataset)
        
        # Save to file
        output_path = os.path.join(self.data_dir, filename)
        dataset.to_csv(output_path, index=False)
        
        print(f"✅ Realistic dataset generated: {output_path}")
        print(f"📊 Dataset shape: {dataset.shape}")
        print(f"🔍 Features: {list(dataset.columns)}")
        
        # Display comprehensive statistics
        self._display_comprehensive_stats(dataset)
        
        return dataset
    
    def _generate_seasonal_dates(self, date_range, n_samples):
        """Generate dates with seasonal bias for agricultural relevance"""
        # Weight dates by agricultural importance
        seasonal_weights = np.ones(len(date_range))
        
        for i, date in enumerate(date_range):
            month = date.month
            
            # Higher weight for growing seasons
            if month in [6, 7, 8, 9, 10]:  # Kharif season
                seasonal_weights[i] = 2.0
            elif month in [10, 11, 12, 1, 2, 3]:  # Rabi season
                seasonal_weights[i] = 1.8
            elif month in [3, 4, 5]:  # Zaid season
                seasonal_weights[i] = 1.5
            
            # Higher weight for harvest months
            if month in [9, 10, 11, 3, 4, 5]:
                seasonal_weights[i] *= 1.3
        
        # Normalize weights
        seasonal_weights = seasonal_weights / seasonal_weights.sum()
        
        # Sample dates with seasonal bias
        sampled_indices = np.random.choice(len(date_range), n_samples, p=seasonal_weights, replace=False)
        sampled_dates = date_range[sampled_indices]
        
        return sorted(sampled_dates)
    
    def _generate_core_features(self, dates, n_samples):
        """Generate core agricultural features"""
        dataset = pd.DataFrame()
        
        # Date features
        dataset['date'] = dates
        dataset['year'] = dataset['date'].dt.year
        dataset['month'] = dataset['date'].dt.month
        dataset['day'] = dataset['date'].dt.day
        dataset['day_of_year'] = dataset['date'].dt.dayofyear
        dataset['week_of_year'] = dataset['date'].dt.isocalendar().week
        
        # Crop type with realistic distribution
        crop_weights = [0.25, 0.20, 0.20, 0.15, 0.20]  # Rice, Wheat, Corn, Soybeans, Cotton
        dataset['crop_type'] = np.random.choice(
            list(self.crop_parameters.keys()), 
            n_samples, 
            p=crop_weights
        )
        
        # Location with realistic distribution
        location_weights = [0.20, 0.15, 0.18, 0.12, 0.10, 0.08, 0.07, 0.05, 0.03, 0.02]
        dataset['location'] = np.random.choice(
            list(self.location_parameters.keys()), 
            n_samples, 
            p=location_weights
        )
        
        # Generate realistic weather data
        dataset = self._generate_weather_data(dataset)
        
        # Generate soil and agricultural parameters
        dataset = self._generate_soil_data(dataset)
        
        return dataset
    
    def _generate_weather_data(self, dataset):
        """Generate realistic weather data based on location and season"""
        for idx, row in dataset.iterrows():
            location = row['location']
            month = row['month']
            crop = row['crop_type']
            
            # Get location parameters
            loc_params = self.location_parameters[location]
            crop_params = self.crop_parameters[crop]
            
            # Generate temperature with seasonal and location variations
            base_temp = 25 + 10 * np.sin(2 * np.pi * (month - 6) / 12)  # Seasonal variation
            location_temp_adjustment = (loc_params['temperature_range'][0] + loc_params['temperature_range'][1]) / 2 - 25
            temp_variation = np.random.normal(0, 3)
            
            dataset.loc[idx, 'temperature'] = base_temp + location_temp_adjustment + temp_variation
            
            # Generate humidity with realistic patterns
            base_humidity = 60 + 20 * np.sin(2 * np.pi * (month - 6) / 12)  # Seasonal variation
            if month in [6, 7, 8, 9]:  # Monsoon months
                base_humidity += 15
            humidity_variation = np.random.normal(0, 8)
            dataset.loc[idx, 'humidity'] = np.clip(base_humidity + humidity_variation, 20, 90)
            
            # Generate rainfall with realistic patterns
            if month in [6, 7, 8, 9]:  # Monsoon season
                rainfall = np.random.exponential(8)  # Higher rainfall
            elif month in [10, 11, 12, 1, 2, 3]:  # Winter
                rainfall = np.random.exponential(2)  # Lower rainfall
            else:  # Summer
                rainfall = np.random.exponential(1)  # Minimal rainfall
            
            dataset.loc[idx, 'rainfall'] = np.clip(rainfall, 0, 50)
            
            # Generate wind speed
            dataset.loc[idx, 'wind_speed'] = np.random.exponential(3)
            
            # Generate pressure
            dataset.loc[idx, 'pressure'] = np.random.normal(1013, 15)
        
        return dataset
    
    def _generate_soil_data(self, dataset):
        """Generate realistic soil and agricultural parameters"""
        for idx, row in dataset.iterrows():
            location = row['location']
            month = row['month']
            
            # Soil moisture (depends on rainfall and irrigation)
            base_moisture = 50 + (row['rainfall'] * 0.6)  # Rainfall increases moisture
            if location in ['Punjab', 'Haryana']:  # High irrigation areas
                base_moisture += 15
            moisture_variation = np.random.normal(0, 8)
            dataset.loc[idx, 'soil_moisture'] = np.clip(base_moisture + moisture_variation, 20, 85)
            
            # pH level (varies by location and soil type)
            if location in ['Punjab', 'Haryana', 'Uttar Pradesh']:  # Alluvial soil
                base_ph = 7.2
            elif location in ['Madhya Pradesh', 'Maharashtra']:  # Black soil
                base_ph = 7.8
            else:  # Red soil
                base_ph = 6.5
            
            ph_variation = np.random.normal(0, 0.3)
            dataset.loc[idx, 'ph_level'] = np.clip(base_ph + ph_variation, 5.5, 8.5)
            
            # Nutrient levels
            dataset.loc[idx, 'nitrogen'] = np.random.normal(200, 50)
            dataset.loc[idx, 'phosphorus'] = np.random.normal(100, 30)
            dataset.loc[idx, 'potassium'] = np.random.normal(150, 40)
            
            # Organic matter
            dataset.loc[idx, 'organic_matter'] = np.random.normal(1.5, 0.5)
        
        return dataset
    
    def _add_derived_features(self, dataset):
        """Add derived features for better ML performance"""
        # Season encoding
        dataset['season'] = dataset['month'].map(self._get_season)
        
        # Growing season indicator
        dataset['is_growing_season'] = dataset['month'].isin([6, 7, 8, 9, 10, 11, 12, 1, 2, 3])
        
        # Weather stress indicators
        dataset['temp_stress'] = np.where(
            (dataset['temperature'] < 15) | (dataset['temperature'] > 35), 1, 0
        )
        
        dataset['humidity_stress'] = np.where(
            (dataset['humidity'] < 40) | (dataset['humidity'] > 80), 1, 0
        )
        
        # Combined stress score
        dataset['weather_stress_score'] = dataset['temp_stress'] + dataset['humidity_stress']
        
        # Soil health indicators
        dataset['soil_health_score'] = (
            (dataset['ph_level'] - 6.5) / 2.0 +  # pH contribution
            (dataset['organic_matter'] - 1.0) / 1.0 +  # Organic matter contribution
            (dataset['nitrogen'] - 150) / 100 +  # Nitrogen contribution
            (dataset['phosphorus'] - 75) / 75 +  # Phosphorus contribution
            (dataset['potassium'] - 125) / 75  # Potassium contribution
        ) / 5.0
        
        # Normalize soil health score
        dataset['soil_health_score'] = np.clip(dataset['soil_health_score'], 0, 1)
        
        return dataset
    
    def _add_realistic_prices(self, dataset):
        """Generate realistic price variations based on multiple factors"""
        for idx, row in dataset.iterrows():
            crop = row['crop_type']
            month = row['month']
            location = row['location']
            
            crop_params = self.crop_parameters[crop]
            base_price = crop_params['base_price']
            
            # Seasonal price variations
            seasonal_factor = 1.0
            if month in crop_params['harvest_months']:
                seasonal_factor = 0.85  # Lower prices during harvest
            elif month in crop_params['planting_months']:
                seasonal_factor = 1.15  # Higher prices during planting
            
            # Location-based price variations
            location_factor = 1.0
            if location in ['Punjab', 'Haryana']:  # Major producing states
                location_factor = 0.95
            elif location in ['Rajasthan', 'Gujarat']:  # Minor producing states
                location_factor = 1.05
            
            # Weather impact on prices
            weather_factor = 1.0
            if row['weather_stress_score'] > 0:
                weather_factor = 1.1  # Higher prices during stress conditions
            
            # Market volatility
            volatility = crop_params['price_volatility']
            random_factor = np.random.normal(1, volatility)
            
            # Calculate final price
            final_price = base_price * seasonal_factor * location_factor * weather_factor * random_factor
            
            # Ensure reasonable price range
            final_price = np.clip(final_price, base_price * 0.5, base_price * 2.0)
            
            dataset.loc[idx, 'price'] = round(final_price, 2)
        
        return dataset
    
    def _add_yield_predictions(self, dataset):
        """Generate realistic yield predictions based on conditions"""
        for idx, row in dataset.iterrows():
            location = row['location']
            crop = row['crop_type']
            
            loc_params = self.location_parameters[location]
            base_yield = 100 * loc_params['base_yield_multiplier']
            
            # Weather impact on yield
            weather_impact = 1.0
            if row['weather_stress_score'] == 0:
                weather_impact = 1.1  # Good conditions
            elif row['weather_stress_score'] == 1:
                weather_impact = 0.95  # Mild stress
            else:
                weather_impact = 0.8  # High stress
            
            # Soil health impact
            soil_impact = 0.8 + (row['soil_health_score'] * 0.4)  # 0.8 to 1.2 range
            
            # Rainfall impact
            rainfall_impact = 1.0
            if 5 <= row['rainfall'] <= 15:
                rainfall_impact = 1.1  # Optimal rainfall
            elif row['rainfall'] > 20:
                rainfall_impact = 0.9  # Excessive rainfall
            
            # Calculate final yield
            final_yield = base_yield * weather_impact * soil_impact * rainfall_impact
            
            # Add some randomness
            yield_variation = np.random.normal(0, 5)
            final_yield += yield_variation
            
            # Ensure reasonable yield range
            final_yield = np.clip(final_yield, 50, 200)
            
            dataset.loc[idx, 'yield'] = round(final_yield, 1)
        
        return dataset
    
    def _add_market_factors(self, dataset):
        """Add market-related factors for comprehensive analysis"""
        # Market demand indicator (higher during planting, lower during harvest)
        dataset['market_demand'] = np.where(
            dataset['month'].isin([6, 7, 10, 11]), 'high',
            np.where(dataset['month'].isin([9, 10, 3, 4]), 'low', 'medium')
        )
        
        # Supply indicator (opposite of demand)
        dataset['market_supply'] = np.where(
            dataset['month'].isin([9, 10, 3, 4]), 'high',
            np.where(dataset['month'].isin([6, 7, 10, 11]), 'low', 'medium')
        )
        
        # Price trend indicator
        dataset['price_trend'] = np.where(
            dataset['price'] > dataset.groupby('crop_type')['price'].transform('mean'), 'rising', 'falling'
        )
        
        # Market volatility (higher for cash crops)
        dataset['market_volatility'] = dataset['crop_type'].map({
            'Rice': 'low',
            'Wheat': 'low',
            'Corn': 'medium',
            'Soybeans': 'high',
            'Cotton': 'high'
        })
        
        return dataset
    
    def _clean_dataset(self, dataset):
        """Clean and validate the final dataset"""
        # Remove any invalid values
        dataset = dataset.replace([np.inf, -np.inf], np.nan)
        dataset = dataset.dropna()
        
        # Ensure data types are correct
        dataset['date'] = pd.to_datetime(dataset['date'])
        dataset['crop_type'] = dataset['crop_type'].astype('category')
        dataset['location'] = dataset['location'].astype('category')
        dataset['market_demand'] = dataset['market_demand'].astype('category')
        dataset['market_supply'] = dataset['market_supply'].astype('category')
        dataset['price_trend'] = dataset['price_trend'].astype('category')
        dataset['market_volatility'] = dataset['market_volatility'].astype('category')
        
        # Round numerical columns
        numeric_columns = ['temperature', 'humidity', 'rainfall', 'wind_speed', 'pressure', 
                          'soil_moisture', 'ph_level', 'nitrogen', 'phosphorus', 'potassium', 
                          'organic_matter', 'price', 'yield']
        
        for col in numeric_columns:
            if col in dataset.columns:
                if col in ['temperature', 'humidity', 'rainfall', 'wind_speed', 'pressure']:
                    dataset[col] = dataset[col].round(1)
                elif col in ['ph_level', 'organic_matter']:
                    dataset[col] = dataset[col].round(2)
                elif col in ['nitrogen', 'phosphorus', 'potassium']:
                    dataset[col] = dataset[col].round(0)
                elif col == 'price':
                    dataset[col] = dataset[col].round(2)
                elif col == 'yield':
                    dataset[col] = dataset[col].round(1)
        
        return dataset
    
    def _get_season(self, month):
        """Convert month to season"""
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Autumn
    
    def _display_comprehensive_stats(self, dataset):
        """Display comprehensive dataset statistics"""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE DATASET STATISTICS")
        print("="*60)
        
        print(f"📈 Total Records: {len(dataset):,}")
        print(f"🔍 Total Features: {len(dataset.columns)}")
        print(f"💾 Memory Usage: {dataset.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Date range
        print(f"\n📅 Date Range: {dataset['date'].min().strftime('%Y-%m-%d')} to {dataset['date'].max().strftime('%Y-%m-%d')}")
        print(f"📅 Total Years: {dataset['year'].max() - dataset['year'].min() + 1}")
        
        # Crop distribution
        print(f"\n🌾 Crop Distribution:")
        crop_counts = dataset['crop_type'].value_counts()
        for crop, count in crop_counts.items():
            percentage = (count / len(dataset)) * 100
            print(f"   {crop}: {count:,} ({percentage:.1f}%)")
        
        # Location distribution
        print(f"\n📍 Top Locations:")
        location_counts = dataset['location'].value_counts().head(8)
        for location, count in location_counts.items():
            percentage = (count / len(dataset)) * 100
            print(f"   {location}: {count:,} ({percentage:.1f}%)")
        
        # Price statistics by crop
        print(f"\n💰 Price Statistics by Crop:")
        for crop in dataset['crop_type'].unique():
            crop_data = dataset[dataset['crop_type'] == crop]
            print(f"   {crop}:")
            print(f"     Mean: ₹{crop_data['price'].mean():.2f}")
            print(f"     Median: ₹{crop_data['price'].median():.2f}")
            print(f"     Std Dev: ₹{crop_data['price'].std():.2f}")
            print(f"     Range: ₹{crop_data['price'].min():.2f} - ₹{crop_data['price'].max():.2f}")
        
        # Weather statistics
        print(f"\n🌤️ Weather Statistics:")
        print(f"   Temperature: {dataset['temperature'].mean():.1f}°C ± {dataset['temperature'].std():.1f}°C")
        print(f"   Humidity: {dataset['humidity'].mean():.1f}% ± {dataset['humidity'].std():.1f}%")
        print(f"   Rainfall: {dataset['rainfall'].mean():.1f}mm ± {dataset['rainfall'].std():.1f}mm")
        
        # Data quality indicators
        print(f"\n✅ Data Quality Indicators:")
        print(f"   Missing Values: {dataset.isnull().sum().sum()}")
        print(f"   Duplicate Records: {dataset.duplicated().sum()}")
        print(f"   Weather Stress Records: {dataset['weather_stress_score'].sum()}")
        
        # ML readiness assessment
        print(f"\n🤖 ML Readiness Assessment:")
        print(f"   Feature Count: {len(dataset.columns)} - {'✅ Excellent' if len(dataset.columns) >= 20 else '⚠️ Good'}")
        print(f"   Sample Size: {len(dataset):,} - {'✅ Excellent' if len(dataset) >= 3000 else '⚠️ Good'}")
        print(f"   Price Variation: {dataset['price'].std() / dataset['price'].mean():.3f} - {'✅ Good' if (dataset['price'].std() / dataset['price'].mean()) > 0.1 else '⚠️ Low'}")
        
        print("="*60)
    
    def create_validation_dataset(self, filename="validation_crop_data.csv", n_samples=1000):
        """Create a smaller validation dataset for testing"""
        print(f"\n🔍 Creating validation dataset with {n_samples:,} records...")
        
        # Use the same generation logic but with fewer samples
        validation_dataset = self.generate_realistic_dataset(filename, n_samples)
        
        print(f"✅ Validation dataset created: {filename}")
        return validation_dataset

def main():
    """Main function for data generation"""
    print("🌾 AI Agriculture Advisor - Realistic Data Generator")
    print("="*60)
    
    # Initialize generator
    generator = RealisticDataGenerator()
    
    # Generate main dataset
    print("\n🚀 Generating main dataset...")
    main_dataset = generator.generate_realistic_dataset("realistic_crop_data.csv", 5000)
    
    # Generate validation dataset
    print("\n🔍 Generating validation dataset...")
    validation_dataset = generator.create_validation_dataset("validation_crop_data.csv", 1000)
    
    print("\n" + "="*60)
    print("🎉 DATA GENERATION COMPLETE!")
    print("="*60)
    print("\n📁 Generated Files:")
    print("   • realistic_crop_data.csv (5,000 records) - Main training dataset")
    print("   • validation_crop_data.csv (1,000 records) - Validation dataset")
    
    print("\n🚀 Next Steps:")
    print("   1. The ML pipeline will automatically use these datasets")
    print("   2. Run the Streamlit app: python run_app.py")
    print("   3. Models will train on the realistic data for accurate predictions")
    
    print("\n💡 Dataset Features:")
    print("   • Realistic seasonal patterns and price variations")
    print("   • Location-specific weather and soil conditions")
    print("   • Crop-specific growing parameters")
    print("   • Market factors and demand indicators")
    print("   • Balanced features to prevent overfitting/underfitting")

if __name__ == "__main__":
    main() 