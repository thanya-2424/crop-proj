"""
AI Agriculture Advisor - Simplified ML Pipeline
Uses only scikit-learn for compatibility and disk space efficiency
"""

import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class SimpleCropPricePredictor:
    """
    Simplified crop price predictor using only scikit-learn
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = 'price'
        
    def load_and_preprocess_data(self, data_path):
        """Load and preprocess the dataset"""
        print("📊 Loading and preprocessing data...")
        
        try:
            # Load data
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.xlsx'):
                df = pd.read_excel(data_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or XLSX.")
            
            print(f"✅ Loaded {len(df):,} records from {data_path}")
            
            # Convert date column
            df['date'] = pd.to_datetime(df['date'])
            
            # Extract temporal features
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            df['day_of_year'] = df['date'].dt.dayofyear
            df['week_of_year'] = df['date'].dt.isocalendar().week
            
            # Create season column
            df['season'] = df['month'].map(self._get_season)
            
            # Encode categorical variables
            categorical_columns = ['crop_type', 'location', 'market_demand', 'price_trend']
            for col in categorical_columns:
                if col in df.columns:
                    le = LabelEncoder()
                    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
            
            # Select numerical features for ML
            numerical_features = [
                'year', 'month', 'day', 'day_of_year', 'week_of_year', 'season',
                'temperature', 'humidity', 'rainfall', 'wind_speed', 'pressure',
                'soil_moisture', 'ph_level', 'nitrogen', 'phosphorus', 'potassium',
                'organic_matter', 'yield', 'weather_stress_score', 'soil_health_score'
            ]
            
            # Add encoded categorical features
            for col in categorical_columns:
                if col + '_encoded' in df.columns:
                    numerical_features.append(col + '_encoded')
            
            # Filter to available columns
            available_features = [col for col in numerical_features if col in df.columns]
            self.feature_columns = available_features
            
            print(f"🔍 Using {len(self.feature_columns)} features: {available_features[:5]}...")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def train_models(self, df):
        """Train multiple models and select the best one"""
        print("🤖 Training ML models...")
        
        try:
            # Prepare features and target
            X = df[self.feature_columns].fillna(0)
            y = df[self.target_column]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Define models
            self.models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Linear Regression': LinearRegression()
            }
            
            # Train and evaluate models
            best_score = -1
            best_model_name = None
            
            for name, model in self.models.items():
                print(f"   Training {name}...")
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                print(f"     {name} - R²: {r2:.4f}, MSE: {mse:.2f}, MAE: {mae:.2f}")
                
                # Store model
                self.models[name] = model
                
                # Update best model
                if r2 > best_score:
                    best_score = r2
                    best_model_name = name
                    self.best_model = model
            
            print(f"🏆 Best model: {best_model_name} (R²: {best_score:.4f})")
            
            return True
            
        except Exception as e:
            print(f"❌ Error training models: {e}")
            return False
    
    def predict_price(self, input_data):
        """Predict price for given input"""
        try:
            # Prepare input features
            input_features = []
            for col in self.feature_columns:
                if col in input_data:
                    input_features.append(input_data[col])
                else:
                    input_features.append(0)  # Default value
            
            # Scale features
            input_scaled = self.scaler.transform([input_features])
            
            # Make prediction
            if self.best_model:
                prediction = self.best_model.predict(input_scaled)[0]
                return max(0, prediction)  # Ensure non-negative price
            else:
                return None
                
        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            return None
    
    def predict_price_trend(self, base_input, months_ahead=3):
        """Predict price trend for next few months"""
        try:
            predictions = []
            current_date = datetime.now()
            
            for i in range(months_ahead):
                # Create future date
                future_date = current_date + timedelta(days=30*i)
                
                # Update input with future date features
                future_input = base_input.copy()
                future_input['year'] = future_date.year
                future_input['month'] = future_date.month
                future_input['day'] = future_date.day
                future_input['day_of_year'] = future_date.timetuple().tm_yday
                future_input['week_of_year'] = future_date.isocalendar()[1]
                future_input['season'] = self._get_season(future_date.month)
                
                # Make prediction
                price = self.predict_price(future_input)
                if price:
                    predictions.append({
                        'date': future_date.strftime('%Y-%m-%d'),
                        'month': future_date.strftime('%B %Y'),
                        'price': round(price, 2)
                    })
            
            return predictions
            
        except Exception as e:
            print(f"❌ Error predicting trend: {e}")
            return []
    
    def save_model(self):
        """Save the trained model and preprocessing objects"""
        try:
            # Ensure models directory exists
            models_dir = '../models'
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
            
            # Save best model
            model_path = os.path.join(models_dir, 'best_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.best_model, f)
            
            # Save scaler
            scaler_path = os.path.join(models_dir, 'scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Save label encoders
            encoders_path = os.path.join(models_dir, 'label_encoders.pkl')
            with open(encoders_path, 'wb') as f:
                pickle.dump(self.label_encoders, f)
            
            # Save feature columns
            features_path = os.path.join(models_dir, 'feature_columns.pkl')
            with open(features_path, 'wb') as f:
                pickle.dump(self.feature_columns, f)
            
            print(f"💾 Model saved to {models_dir}/")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {e}")
            return False
    
    def load_model(self):
        """Load pre-trained model and preprocessing objects"""
        try:
            models_dir = '../models'
            
            # Load best model
            model_path = os.path.join(models_dir, 'best_model.pkl')
            with open(model_path, 'rb') as f:
                self.best_model = pickle.load(f)
            
            # Load scaler
            scaler_path = os.path.join(models_dir, 'scaler.pkl')
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Load label encoders
            encoders_path = os.path.join(models_dir, 'label_encoders.pkl')
            with open(encoders_path, 'rb') as f:
                self.label_encoders = pickle.load(f)
            
            # Load feature columns
            features_path = os.path.join(models_dir, 'feature_columns.pkl')
            with open(features_path, 'rb') as f:
                self.feature_columns = pickle.load(f)
            
            print("✅ Pre-trained model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def _get_season(self, month):
        """Convert month to season (0-3)"""
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Autumn

def main():
    """Test the ML pipeline"""
    print("🌾 Testing Simplified ML Pipeline...")
    
    # Initialize predictor
    predictor = SimpleCropPricePredictor()
    
    # Load and preprocess data
    data_path = '../data/realistic_crop_data.csv'
    df = predictor.load_and_preprocess_data(data_path)
    
    if df is not None:
        # Train models
        success = predictor.train_models(df)
        
        if success:
            # Save model
            predictor.save_model()
            
            # Test prediction
            test_input = {
                'year': 2024,
                'month': 8,
                'day': 15,
                'day_of_year': 228,
                'week_of_year': 33,
                'season': 2,
                'temperature': 28.5,
                'humidity': 65.0,
                'rainfall': 5.2,
                'wind_speed': 3.1,
                'pressure': 1013.2,
                'soil_moisture': 45.8,
                'ph_level': 7.2,
                'nitrogen': 180,
                'phosphorus': 95,
                'potassium': 140,
                'organic_matter': 1.8,
                'yield': 110.5,
                'weather_stress_score': 0,
                'soil_health_score': 0.75
            }
            
            # Add encoded categorical features
            for col in ['crop_type', 'location', 'market_demand', 'price_trend']:
                if col + '_encoded' in predictor.feature_columns:
                    test_input[col + '_encoded'] = 0  # Default value
            
            price = predictor.predict_price(test_input)
            print(f"💰 Test prediction: ₹{price:.2f}")
            
            # Test trend prediction
            trend = predictor.predict_price_trend(test_input, 3)
            print(f"📈 Price trend: {len(trend)} months predicted")
            
    print("✅ ML Pipeline test complete!")

if __name__ == "__main__":
    main() 