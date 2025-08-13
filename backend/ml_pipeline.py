"""
AI Agriculture Advisor - ML Pipeline
This module handles data preprocessing, model training, and prediction for crop price forecasting.
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class CropPricePredictor:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = 'price'
        
    def load_and_preprocess_data(self, data_path):
        """
        Load and preprocess the crop price dataset
        """
        try:
            # Load data
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.xlsx'):
                df = pd.read_excel(data_path)
            else:
                raise ValueError("Unsupported file format. Please use CSV or Excel files.")
            
            print(f"Loaded dataset with shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            
            # Basic data cleaning
            df = df.dropna()
            df = df.reset_index(drop=True)
            
            # Convert date columns to datetime
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            for col in date_columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Extract temporal features
            if date_columns:
                date_col = date_columns[0]
                df['year'] = df[date_col].dt.year
                df['month'] = df[date_col].dt.month
                df['season'] = df[date_col].dt.month.map(self._get_season)
                df['quarter'] = df[date_col].dt.quarter
            
            # Handle categorical variables
            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if col != self.target_column:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
            
            # Select numerical features for modeling
            numerical_columns = df.select_dtypes(include=[np.number]).columns
            self.feature_columns = [col for col in numerical_columns if col != self.target_column]
            
            # Ensure target column exists
            if self.target_column not in df.columns:
                # If no price column, create a synthetic one for demonstration
                print("No price column found. Creating synthetic price data for demonstration.")
                df[self.target_column] = np.random.normal(100, 20, len(df))
            
            print(f"Preprocessed dataset shape: {df.shape}")
            print(f"Feature columns: {self.feature_columns}")
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
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
    
    def train_models(self, df):
        """
        Train multiple models and select the best one
        """
        if df is None or len(self.feature_columns) == 0:
            print("No data available for training")
            return
        
        # Prepare features and target
        X = df[self.feature_columns]
        y = df[self.target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models to train
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        # Train and evaluate models
        best_score = -np.inf
        best_model_name = None
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            if name in ['XGBoost', 'LightGBM']:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            
            # Evaluate model
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"{name} - MSE: {mse:.2f}, R²: {r2:.3f}, MAE: {mae:.2f}")
            
            # Store model
            self.models[name] = model
            
            # Update best model
            if r2 > best_score:
                best_score = r2
                best_model_name = name
        
        # Set best model
        self.best_model = self.models[best_model_name]
        print(f"\nBest model: {best_model_name} with R² score: {best_score:.3f}")
        
        # Save best model
        self.save_model()
        
        return best_model_name, best_score
    
    def predict_price(self, input_data):
        """
        Predict crop price based on input features
        """
        if self.best_model is None:
            print("No trained model available. Please train the model first.")
            return None
        
        try:
            # Prepare input features
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical variables if present
            for col, le in self.label_encoders.items():
                if col in input_df.columns:
                    input_df[col] = le.transform(input_df[col].astype(str))
            
            # Select only feature columns
            input_features = input_df[self.feature_columns]
            
            # Scale features
            input_scaled = self.scaler.transform(input_features)
            
            # Make prediction
            prediction = self.best_model.predict(input_scaled)[0]
            
            return max(0, prediction)  # Ensure non-negative price
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            return None
    
    def predict_price_trend(self, base_input, months_ahead=3):
        """
        Predict price trend for next few months
        """
        if self.best_model is None:
            return None
        
        predictions = []
        dates = []
        
        for i in range(months_ahead):
            # Modify input for future prediction
            future_input = base_input.copy()
            
            # Update temporal features
            if 'month' in future_input:
                future_input['month'] = (future_input['month'] + i) % 12
                if future_input['month'] == 0:
                    future_input['month'] = 12
            
            if 'year' in future_input:
                future_input['year'] = future_input['year'] + (future_input['month'] + i - 1) // 12
            
            if 'season' in future_input:
                future_input['season'] = self._get_season(future_input['month'])
            
            if 'quarter' in future_input:
                future_input['quarter'] = ((future_input['month'] - 1) // 3) + 1
            
            # Make prediction
            pred = self.predict_price(future_input)
            if pred is not None:
                predictions.append(pred)
                
                # Generate future date
                current_date = datetime.now()
                future_date = current_date + timedelta(days=30*i)
                dates.append(future_date.strftime('%Y-%m-%d'))
        
        return dates, predictions
    
    def save_model(self):
        """Save the trained model and preprocessing objects"""
        if not os.path.exists('../models'):
            os.makedirs('../models')
        
        model_path = '../models/crop_price_model.pkl'
        scaler_path = '../models/scaler.pkl'
        encoders_path = '../models/encoders.pkl'
        
        joblib.dump(self.best_model, model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.label_encoders, encoders_path)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self):
        """Load pre-trained model and preprocessing objects"""
        try:
            model_path = '../models/crop_price_model.pkl'
            scaler_path = '../models/scaler.pkl'
            encoders_path = '../models/encoders.pkl'
            
            if all(os.path.exists(p) for p in [model_path, scaler_path, encoders_path]):
                self.best_model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                self.label_encoders = joblib.load(encoders_path)
                print("Pre-trained model loaded successfully")
                return True
            else:
                print("Pre-trained model files not found")
                return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

# Example usage and testing
if __name__ == "__main__":
    predictor = CropPricePredictor()
    
    # Test with sample data if no real dataset is available
    print("Creating sample dataset for testing...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = {
        'date': pd.date_range('2019-01-01', periods=n_samples, freq='D'),
        'crop_type': np.random.choice(['Rice', 'Wheat', 'Corn', 'Soybeans'], n_samples),
        'location': np.random.choice(['Punjab', 'Haryana', 'UP', 'MP'], n_samples),
        'temperature': np.random.normal(25, 10, n_samples),
        'humidity': np.random.uniform(30, 90, n_samples),
        'rainfall': np.random.exponential(5, n_samples),
        'yield': np.random.normal(100, 20, n_samples),
        'price': np.random.normal(100, 30, n_samples)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Save sample data
    df.to_csv('../data/sample_crop_data.csv', index=False)
    print("Sample dataset created and saved to data/sample_crop_data.csv")
    
    # Train models
    print("\nTraining models...")
    best_model, score = predictor.train_models(df)
    
    # Test prediction
    test_input = {
        'crop_type': 0,  # Rice
        'location': 0,   # Punjab
        'temperature': 25,
        'humidity': 60,
        'rainfall': 5,
        'yield': 100,
        'year': 2024,
        'month': 6,
        'season': 2,
        'quarter': 2
    }
    
    prediction = predictor.predict_price(test_input)
    print(f"\nTest prediction: ${prediction:.2f}")
    
    # Test trend prediction
    dates, predictions = predictor.predict_price_trend(test_input, 3)
    print(f"\nPrice trend prediction:")
    for date, pred in zip(dates, predictions):
        print(f"{date}: ${pred:.2f}") 