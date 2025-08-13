"""
AI Agriculture Advisor - Data Loader Utility
This script helps users prepare and load crop datasets for the ML pipeline.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CropDataLoader:
    """
    Utility class for loading and preparing crop datasets
    """
    
    def __init__(self, data_dir="../data"):
        self.data_dir = data_dir
        self.ensure_data_directory()
    
    def ensure_data_directory(self):
        """Ensure data directory exists"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created data directory: {self.data_dir}")
    
    def create_sample_dataset(self, filename="sample_crop_data.csv", n_samples=1000):
        """
        Create a comprehensive sample dataset for testing
        
        Args:
            filename: Name of the output file
            n_samples: Number of sample records to generate
        """
        print(f"Creating sample dataset with {n_samples} records...")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate date range (5 years of data)
        start_date = datetime(2019, 1, 1)
        end_date = datetime(2024, 12, 31)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Sample dates
        sample_dates = np.random.choice(date_range, n_samples, replace=False)
        sample_dates.sort()
        
        # Generate sample data
        sample_data = {
            'date': sample_dates,
            'crop_type': np.random.choice(['Rice', 'Wheat', 'Corn', 'Soybeans', 'Cotton'], n_samples),
            'location': np.random.choice(['Punjab', 'Haryana', 'Uttar Pradesh', 'Madhya Pradesh', 'Rajasthan'], n_samples),
            'temperature': np.random.normal(25, 10, n_samples),
            'humidity': np.random.uniform(30, 90, n_samples),
            'rainfall': np.random.exponential(5, n_samples),
            'soil_moisture': np.random.uniform(20, 80, n_samples),
            'ph_level': np.random.uniform(5.5, 8.5, n_samples),
            'nitrogen': np.random.uniform(100, 300, n_samples),
            'phosphorus': np.random.uniform(50, 150, n_samples),
            'potassium': np.random.uniform(100, 250, n_samples),
            'yield': np.random.normal(100, 20, n_samples),
            'price': np.random.normal(2500, 500, n_samples)
        }
        
        # Create DataFrame
        df = pd.DataFrame(sample_data)
        
        # Add seasonal variations
        df['month'] = df['date'].dt.month
        df['season'] = df['month'].map(self._get_season)
        
        # Adjust prices based on season and crop type
        for crop in df['crop_type'].unique():
            crop_mask = df['crop_type'] == crop
            base_price = df.loc[crop_mask, 'price'].mean()
            
            # Seasonal price variations
            seasonal_factor = np.where(df.loc[crop_mask, 'season'] == 0, 1.1,  # Winter
                                    np.where(df.loc[crop_mask, 'season'] == 1, 1.05,  # Spring
                                    np.where(df.loc[crop_mask, 'season'] == 2, 0.95,  # Summer
                                    0.9)))  # Autumn
            
            df.loc[crop_mask, 'price'] = base_price * seasonal_factor * np.random.normal(1, 0.1, crop_mask.sum())
        
        # Ensure non-negative values
        df['price'] = np.maximum(df['price'], 100)
        df['yield'] = np.maximum(df['yield'], 10)
        df['temperature'] = np.clip(df['temperature'], -10, 50)
        
        # Save to file
        output_path = os.path.join(self.data_dir, filename)
        df.to_csv(output_path, index=False)
        
        print(f"Sample dataset created: {output_path}")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Display sample statistics
        self.display_dataset_stats(df)
        
        return df
    
    def _get_season(self, month):
        """Convert month to season (0: Winter, 1: Spring, 2: Summer, 3: Autumn)"""
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Autumn
    
    def load_dataset(self, filename):
        """
        Load dataset from file
        
        Args:
            filename: Name of the file to load
            
        Returns:
            pandas DataFrame or None if error
        """
        try:
            file_path = os.path.join(self.data_dir, filename)
            
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return None
            
            # Determine file type and load
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                print(f"Unsupported file format: {filename}")
                return None
            
            print(f"Dataset loaded successfully: {file_path}")
            print(f"Dataset shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Display basic statistics
            self.display_dataset_stats(df)
            
            return df
            
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return None
    
    def validate_dataset(self, df):
        """
        Validate dataset for ML pipeline compatibility
        
        Args:
            df: pandas DataFrame to validate
            
        Returns:
            dict: Validation results
        """
        if df is None:
            return {'valid': False, 'errors': ['Dataset is None']}
        
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check required columns
        required_columns = ['date', 'crop_type', 'location', 'price']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
        
        # Check data types
        if 'date' in df.columns:
            try:
                pd.to_datetime(df['date'])
            except:
                validation_results['warnings'].append("Date column format may need conversion")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        high_missing = missing_counts[missing_counts > len(df) * 0.1]
        
        if not high_missing.empty:
            validation_results['warnings'].append(f"High missing values in columns: {list(high_missing.index)}")
        
        # Check data quality
        if 'price' in df.columns:
            if (df['price'] <= 0).any():
                validation_results['warnings'].append("Some price values are non-positive")
            
            if df['price'].std() == 0:
                validation_results['warnings'].append("Price column has no variation")
        
        # Check data size
        if len(df) < 100:
            validation_results['warnings'].append("Dataset size is small for reliable ML training")
        
        # Generate recommendations
        if validation_results['valid']:
            validation_results['recommendations'].append("Dataset is ready for ML pipeline")
        
        if validation_results['warnings']:
            validation_results['recommendations'].append("Consider addressing warnings before training")
        
        return validation_results
    
    def display_dataset_stats(self, df):
        """Display basic dataset statistics"""
        if df is None:
            print("No dataset to display")
            return
        
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        
        print(f"Total Records: {len(df):,}")
        print(f"Total Columns: {len(df.columns)}")
        print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Date range
        if 'date' in df.columns:
            try:
                dates = pd.to_datetime(df['date'])
                print(f"Date Range: {dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")
            except:
                print("Date Range: Unable to parse")
        
        # Crop distribution
        if 'crop_type' in df.columns:
            print(f"\nCrop Distribution:")
            crop_counts = df['crop_type'].value_counts()
            for crop, count in crop_counts.items():
                print(f"  {crop}: {count:,} ({count/len(df)*100:.1f}%)")
        
        # Location distribution
        if 'location' in df.columns:
            print(f"\nTop Locations:")
            location_counts = df['location'].value_counts().head(5)
            for location, count in location_counts.items():
                print(f"  {location}: {count:,}")
        
        # Price statistics
        if 'price' in df.columns:
            print(f"\nPrice Statistics:")
            print(f"  Mean: ₹{df['price'].mean():.2f}")
            print(f"  Median: ₹{df['price'].median():.2f}")
            print(f"  Std Dev: ₹{df['price'].std():.2f}")
            print(f"  Min: ₹{df['price'].min():.2f}")
            print(f"  Max: ₹{df['price'].max():.2f}")
        
        # Missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"\nMissing Values:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"  {col}: {count:,} ({count/len(df)*100:.1f}%)")
        
        print("="*50)
    
    def prepare_for_ml(self, df):
        """
        Prepare dataset for ML pipeline
        
        Args:
            df: pandas DataFrame to prepare
            
        Returns:
            pandas DataFrame: Prepared dataset
        """
        if df is None:
            return None
        
        print("Preparing dataset for ML pipeline...")
        
        # Create a copy to avoid modifying original
        df_ml = df.copy()
        
        # Convert date to datetime if needed
        if 'date' in df_ml.columns:
            df_ml['date'] = pd.to_datetime(df_ml['date'])
            
            # Extract temporal features
            df_ml['year'] = df_ml['date'].dt.year
            df_ml['month'] = df_ml['date'].dt.month
            df_ml['day'] = df_ml['date'].dt.day
            df_ml['day_of_week'] = df_ml['date'].dt.dayofweek
            df_ml['quarter'] = df_ml['date'].dt.quarter
        
        # Handle categorical variables
        categorical_columns = df_ml.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != 'date':  # Don't encode date
                df_ml[col] = df_ml[col].astype('category').cat.codes
        
        # Fill missing values
        numeric_columns = df_ml.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df_ml[col].isnull().sum() > 0:
                df_ml[col].fillna(df_ml[col].median(), inplace=True)
        
        print("Dataset prepared for ML pipeline")
        return df_ml

def main():
    """Main function for testing and demonstration"""
    
    # Initialize data loader
    loader = CropDataLoader()
    
    print("🌾 AI Agriculture Advisor - Data Loader Utility")
    print("="*50)
    
    # Check existing datasets
    data_files = [f for f in os.listdir(loader.data_dir) if f.endswith(('.csv', '.xlsx'))]
    
    if data_files:
        print(f"Found existing datasets: {data_files}")
        
        # Load and validate first dataset
        first_file = data_files[0]
        print(f"\nLoading and validating: {first_file}")
        
        df = loader.load_dataset(first_file)
        if df is not None:
            validation = loader.validate_dataset(df)
            
            print(f"\nValidation Results:")
            print(f"Valid: {validation['valid']}")
            
            if validation['errors']:
                print(f"Errors: {validation['errors']}")
            
            if validation['warnings']:
                print(f"Warnings: {validation['warnings']}")
            
            if validation['recommendations']:
                print(f"Recommendations: {validation['recommendations']}")
            
            # Prepare for ML
            df_ml = loader.prepare_for_ml(df)
            
    else:
        print("No existing datasets found. Creating sample dataset...")
        
        # Create sample dataset
        df = loader.create_sample_dataset()
        
        # Validate sample dataset
        validation = loader.validate_dataset(df)
        print(f"\nSample dataset validation: {'✅ Valid' if validation['valid'] else '❌ Invalid'}")
        
        # Prepare for ML
        df_ml = loader.prepare_for_ml(df)
    
    print("\n" + "="*50)
    print("Data Loader Utility Ready!")
    print("="*50)
    print("\nNext steps:")
    print("1. Use the ML pipeline to train models")
    print("2. Run the Streamlit app to get predictions")
    print("3. Customize with your own datasets")

if __name__ == "__main__":
    main() 