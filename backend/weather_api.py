"""
AI Agriculture Advisor - Weather API Integration
This module handles weather data fetching and provides crop-specific recommendations.
"""

import requests
import json
from datetime import datetime, timedelta
import os
from typing import Dict, Optional, Tuple

class WeatherAPI:
    def __init__(self, api_key: str = None):
        """
        Initialize Weather API with OpenWeatherMap API key
        """
        self.api_key = api_key or os.getenv('OPENWEATHER_API_KEY')
        self.base_url = "http://api.openweathermap.org/data/2.5"
        
        if not self.api_key:
            print("Warning: No OpenWeatherMap API key provided. Using demo mode.")
            self.demo_mode = True
        else:
            self.demo_mode = False
    
    def get_weather_by_city(self, city: str, state: str = None, country: str = "IN") -> Dict:
        """
        Get current weather data for a city
        """
        if self.demo_mode:
            return self._get_demo_weather(city, state)
        
        try:
            # Construct location string
            location = f"{city}"
            if state:
                location += f",{state}"
            location += f",{country}"
            
            # Get current weather
            current_url = f"{self.base_url}/weather"
            params = {
                'q': location,
                'appid': self.api_key,
                'units': 'metric'  # Use Celsius
            }
            
            response = requests.get(current_url, params=params, timeout=10)
            response.raise_for_status()
            
            weather_data = response.json()
            
            # Extract relevant information
            weather_info = {
                'city': weather_data['name'],
                'country': weather_data['sys']['country'],
                'temperature': weather_data['main']['temp'],
                'feels_like': weather_data['main']['feels_like'],
                'humidity': weather_data['main']['humidity'],
                'pressure': weather_data['main']['pressure'],
                'description': weather_data['weather'][0]['description'],
                'wind_speed': weather_data['wind']['speed'],
                'wind_direction': weather_data['wind'].get('deg', 0),
                'clouds': weather_data['clouds']['all'],
                'visibility': weather_data.get('visibility', 10000),
                'sunrise': datetime.fromtimestamp(weather_data['sys']['sunrise']).strftime('%H:%M'),
                'sunset': datetime.fromtimestamp(weather_data['sys']['sunset']).strftime('%H:%M'),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return weather_info
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {str(e)}")
            return self._get_demo_weather(city, state)
        except Exception as e:
            print(f"Error processing weather data: {str(e)}")
            return self._get_demo_weather(city, state)
    
    def get_weather_forecast(self, city: str, state: str = None, country: str = "IN", days: int = 5) -> Dict:
        """
        Get weather forecast for upcoming days
        """
        if self.demo_mode:
            return self._get_demo_forecast(city, state, days)
        
        try:
            # Construct location string
            location = f"{city}"
            if state:
                location += f",{state}"
            location += f",{country}"
            
            # Get forecast
            forecast_url = f"{self.base_url}/forecast"
            params = {
                'q': location,
                'appid': self.api_key,
                'units': 'metric',
                'cnt': min(days * 8, 40)  # OpenWeatherMap provides 3-hour intervals
            }
            
            response = requests.get(forecast_url, params=params, timeout=10)
            response.raise_for_status()
            
            forecast_data = response.json()
            
            # Process forecast data
            daily_forecast = {}
            for item in forecast_data['list']:
                date = datetime.fromtimestamp(item['dt']).strftime('%Y-%m-%d')
                
                if date not in daily_forecast:
                    daily_forecast[date] = {
                        'date': date,
                        'temp_min': item['main']['temp'],
                        'temp_max': item['main']['temp'],
                        'humidity': item['main']['humidity'],
                        'pressure': item['main']['pressure'],
                        'wind_speed': item['wind']['speed'],
                        'description': item['weather'][0]['description'],
                        'rain': item.get('rain', {}).get('3h', 0),
                        'snow': item.get('snow', {}).get('3h', 0)
                    }
                else:
                    # Update min/max temperatures
                    daily_forecast[date]['temp_min'] = min(daily_forecast[date]['temp_min'], item['main']['temp'])
                    daily_forecast[date]['temp_max'] = max(daily_forecast[date]['temp_max'], item['main']['temp'])
                    # Average other metrics
                    daily_forecast[date]['humidity'] = (daily_forecast[date]['humidity'] + item['main']['humidity']) / 2
                    daily_forecast[date]['pressure'] = (daily_forecast[date]['pressure'] + item['main']['pressure']) / 2
                    daily_forecast[date]['wind_speed'] = (daily_forecast[date]['wind_speed'] + item['wind']['speed']) / 2
                    # Accumulate precipitation
                    daily_forecast[date]['rain'] += item.get('rain', {}).get('3h', 0)
                    daily_forecast[date]['snow'] += item.get('snow', {}).get('3h', 0)
            
            return {
                'city': forecast_data['city']['name'],
                'country': forecast_data['city']['country'],
                'forecast': list(daily_forecast.values())[:days]
            }
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching forecast data: {str(e)}")
            return self._get_demo_forecast(city, state, days)
        except Exception as e:
            print(f"Error processing forecast data: {str(e)}")
            return self._get_demo_forecast(city, state, days)
    
    def _get_demo_weather(self, city: str, state: str = None) -> Dict:
        """
        Generate demo weather data for testing purposes
        """
        import random
        
        # Generate realistic demo data
        base_temp = random.uniform(20, 35)
        humidity = random.uniform(40, 80)
        
        return {
            'city': city,
            'state': state,
            'country': 'IN',
            'temperature': round(base_temp, 1),
            'feels_like': round(base_temp + random.uniform(-2, 2), 1),
            'humidity': round(humidity, 1),
            'pressure': round(random.uniform(1000, 1020), 1),
            'description': random.choice(['clear sky', 'scattered clouds', 'light rain', 'overcast']),
            'wind_speed': round(random.uniform(0, 15), 1),
            'wind_direction': random.randint(0, 360),
            'clouds': random.randint(0, 100),
            'visibility': random.randint(5000, 10000),
            'sunrise': '06:30',
            'sunset': '18:30',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'demo_mode': True
        }
    
    def _get_demo_forecast(self, city: str, state: str = None, days: int = 5) -> Dict:
        """
        Generate demo forecast data for testing purposes
        """
        import random
        
        forecast = []
        base_temp = random.uniform(20, 35)
        
        for i in range(days):
            temp_variation = random.uniform(-3, 3)
            forecast.append({
                'date': (datetime.now().date() + timedelta(days=i)).strftime('%Y-%m-%d'),
                'temp_min': round(base_temp + temp_variation - 2, 1),
                'temp_max': round(base_temp + temp_variation + 2, 1),
                'humidity': round(random.uniform(40, 80), 1),
                'pressure': round(random.uniform(1000, 1020), 1),
                'wind_speed': round(random.uniform(0, 15), 1),
                'description': random.choice(['clear sky', 'scattered clouds', 'light rain', 'overcast']),
                'rain': round(random.uniform(0, 5), 1),
                'snow': 0
            })
        
        return {
            'city': city,
            'state': state,
            'country': 'IN',
            'forecast': forecast,
            'demo_mode': True
        }

class CropAdvisor:
    """
    Provides crop-specific advice based on weather conditions
    """
    
    def __init__(self):
        self.crop_recommendations = {
            'Rice': {
                'optimal_temp': (20, 35),
                'optimal_humidity': (60, 80),
                'water_requirement': 'High',
                'fertilizer_timing': 'Before transplanting and at panicle initiation',
                'irrigation_frequency': 'Daily during early growth, every 2-3 days later'
            },
            'Wheat': {
                'optimal_temp': (15, 25),
                'optimal_humidity': (40, 60),
                'water_requirement': 'Medium',
                'fertilizer_timing': 'At sowing and first irrigation',
                'irrigation_frequency': 'Every 7-10 days'
            },
            'Corn': {
                'optimal_temp': (18, 32),
                'optimal_humidity': (50, 70),
                'water_requirement': 'High',
                'fertilizer_timing': 'At sowing and knee-high stage',
                'irrigation_frequency': 'Every 3-5 days during critical growth'
            },
            'Soybeans': {
                'optimal_temp': (20, 30),
                'optimal_humidity': (50, 70),
                'water_requirement': 'Medium',
                'fertilizer_timing': 'At sowing and flowering stage',
                'irrigation_frequency': 'Every 5-7 days'
            },
            'Cotton': {
                'optimal_temp': (25, 35),
                'optimal_humidity': (40, 60),
                'water_requirement': 'Medium-High',
                'fertilizer_timing': 'At sowing and square formation',
                'irrigation_frequency': 'Every 7-10 days'
            }
        }
    
    def get_crop_advice(self, crop_type: str, weather_data: Dict) -> Dict:
        """
        Generate crop-specific advice based on current weather
        """
        if crop_type not in self.crop_recommendations:
            return {
                'error': f'No recommendations available for {crop_type}',
                'general_advice': 'Please consult local agricultural experts for specific crop advice.'
            }
        
        crop_info = self.crop_recommendations[crop_type]
        temp = weather_data.get('temperature', 25)
        humidity = weather_data.get('humidity', 60)
        rainfall = weather_data.get('rain', 0)
        
        # Analyze conditions
        temp_status = self._analyze_temperature(temp, crop_info['optimal_temp'])
        humidity_status = self._analyze_humidity(humidity, crop_info['optimal_humidity'])
        
        # Generate recommendations
        advice = {
            'crop_type': crop_type,
            'current_conditions': {
                'temperature': f"{temp}°C ({temp_status})",
                'humidity': f"{humidity}% ({humidity_status})",
                'rainfall': f"{rainfall}mm"
            },
            'recommendations': {
                'water_management': self._get_water_advice(crop_type, temp, humidity, rainfall),
                'fertilizer_application': crop_info['fertilizer_timing'],
                'irrigation_schedule': crop_info['irrigation_frequency'],
                'general_care': self._get_general_care_advice(crop_type, temp, humidity)
            },
            'risk_assessment': self._assess_risks(crop_type, temp, humidity, rainfall),
            'next_actions': self._get_next_actions(crop_type, temp, humidity)
        }
        
        return advice
    
    def _analyze_temperature(self, temp: float, optimal_range: Tuple[float, float]) -> str:
        """Analyze if temperature is optimal for crop growth"""
        min_temp, max_temp = optimal_range
        if min_temp <= temp <= max_temp:
            return "Optimal"
        elif temp < min_temp:
            return "Too Cold"
        else:
            return "Too Hot"
    
    def _analyze_humidity(self, humidity: float, optimal_range: Tuple[float, float]) -> str:
        """Analyze if humidity is optimal for crop growth"""
        min_humidity, max_humidity = optimal_range
        if min_humidity <= humidity <= max_humidity:
            return "Optimal"
        elif humidity < min_humidity:
            return "Too Dry"
        else:
            return "Too Humid"
    
    def _get_water_advice(self, crop_type: str, temp: float, humidity: float, rainfall: float) -> str:
        """Generate water management advice"""
        if rainfall > 10:
            return "Reduce irrigation due to recent rainfall"
        elif temp > 30 and humidity < 50:
            return "Increase irrigation frequency due to high temperature and low humidity"
        elif humidity > 80:
            return "Reduce irrigation to prevent waterlogging"
        else:
            return "Maintain normal irrigation schedule"
    
    def _get_general_care_advice(self, crop_type: str, temp: float, humidity: float) -> str:
        """Generate general care advice"""
        if temp > 35:
            return "Provide shade and increase irrigation to protect from heat stress"
        elif temp < 15:
            return "Consider using row covers or greenhouses to protect from cold"
        elif humidity > 85:
            return "Monitor for fungal diseases and ensure proper ventilation"
        else:
            return "Conditions are favorable for crop growth"
    
    def _assess_risks(self, crop_type: str, temp: float, humidity: float, rainfall: float) -> Dict:
        """Assess potential risks to crop health"""
        risks = []
        risk_level = "Low"
        
        if temp > 35:
            risks.append("Heat stress - may reduce yield")
            risk_level = "High"
        elif temp < 15:
            risks.append("Cold stress - may slow growth")
            risk_level = "Medium"
        
        if humidity > 85:
            risks.append("High humidity - increased disease risk")
            risk_level = "Medium"
        elif humidity < 40:
            risks.append("Low humidity - increased water requirement")
            risk_level = "Low"
        
        if rainfall > 20:
            risks.append("Heavy rainfall - potential waterlogging")
            risk_level = "Medium"
        
        return {
            'risk_level': risk_level,
            'risks': risks if risks else ["No significant risks identified"]
        }
    
    def _get_next_actions(self, crop_type: str, temp: float, humidity: float) -> list:
        """Suggest next actions for farmers"""
        actions = []
        
        if temp > 35:
            actions.append("Schedule irrigation for early morning or evening")
            actions.append("Monitor soil moisture levels")
        elif temp < 15:
            actions.append("Check soil temperature before planting")
            actions.append("Consider delayed planting if cold persists")
        
        if humidity > 80:
            actions.append("Inspect for signs of fungal diseases")
            actions.append("Ensure proper field drainage")
        
        actions.append("Regular monitoring of crop health")
        actions.append("Prepare for next fertilizer application")
        
        return actions

# Example usage
if __name__ == "__main__":
    # Test weather API (demo mode)
    weather_api = WeatherAPI()
    
    # Get current weather
    weather = weather_api.get_weather_by_city("Punjab", "Punjab")
    print("Current Weather:")
    print(json.dumps(weather, indent=2))
    
    # Get forecast
    forecast = weather_api.get_weather_forecast("Punjab", "Punjab", days=3)
    print("\nWeather Forecast:")
    print(json.dumps(forecast, indent=2))
    
    # Test crop advisor
    advisor = CropAdvisor()
    advice = advisor.get_crop_advice("Rice", weather)
    print("\nCrop Advice:")
    print(json.dumps(advice, indent=2)) 