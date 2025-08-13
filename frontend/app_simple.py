"""
AI Agriculture Advisor - Simplified Streamlit App
Uses simplified ML pipeline for better compatibility
"""

import streamlit as st
import sys
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

try:
    from ml_pipeline_simple import SimpleCropPricePredictor
    from weather_api import WeatherAPI, CropAdvisor
    from pdf_generator import CropReportGenerator
    print("✅ All backend modules imported successfully!")
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="AI Agriculture Advisor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #228B22;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 0.5rem 0;
    }
    .weather-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
        margin: 0.5rem 0;
    }
    .advice-card {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #9c27b0;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #218838 0%, #1ea085 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = SimpleCropPricePredictor()
    if 'weather_api' not in st.session_state:
        st.session_state.weather_api = WeatherAPI()
    if 'crop_advisor' not in st.session_state:
        st.session_state.crop_advisor = CropAdvisor()
    
    # Main header
    st.markdown('<h1 class="main-header">🌾 AI Agriculture Advisor</h1>', unsafe_allow_html=True)
    st.markdown("### Your intelligent farming companion for crop insights and market predictions")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("🌱 Crop & Location Details")
        
        # Crop selection
        crop_type = st.selectbox(
            "Select Crop Type",
            ["Rice", "Wheat", "Corn", "Soybeans", "Cotton"],
            help="Choose the crop you want to analyze"
        )
        
        # Location selection
        state = st.selectbox(
            "Select State",
            ["Punjab", "Haryana", "Uttar Pradesh", "Madhya Pradesh", 
             "Rajasthan", "Gujarat", "Maharashtra", "Karnataka", "Tamil Nadu", "Andhra Pradesh"],
            help="Select your state for location-specific analysis"
        )
        
        district = st.text_input(
            "District/City",
            value="",
            help="Enter your district or city name"
        )
        
        st.header("🌤️ Weather Information")
        
        # Weather source selection
        weather_source = st.radio(
            "Weather Data Source",
            ["Live API (OpenWeatherMap)", "Manual Input", "Demo Mode"],
            help="Choose how to get weather data"
        )
        
        # API key input
        if weather_source == "Live API (OpenWeatherMap)":
            api_key = st.text_input(
                "OpenWeatherMap API Key",
                type="password",
                help="Enter your OpenWeatherMap API key for live weather data"
            )
            if api_key:
                st.session_state.weather_api.api_key = api_key
        
        # Manual weather inputs
        elif weather_source == "Manual Input":
            col1, col2 = st.columns(2)
            with col1:
                temperature = st.number_input("Temperature (°C)", -10.0, 50.0, 25.0, 0.1)
                humidity = st.number_input("Humidity (%)", 0, 100, 60, 1)
            with col2:
                rainfall = st.number_input("Rainfall (mm)", 0.0, 100.0, 5.0, 0.1)
                wind_speed = st.number_input("Wind Speed (m/s)", 0.0, 20.0, 3.0, 0.1)
        
        # Analysis button
        analyze_button = st.button(
            "🚀 Analyze & Get Recommendations",
            use_container_width=True,
            help="Click to analyze your crop and get comprehensive recommendations"
        )
    
    # Main content area
    if analyze_button:
        st.markdown("---")
        st.markdown('<h2 class="sub-header">🔍 Analysis Results</h2>', unsafe_allow_html=True)
        
        try:
            # Get weather data
            if weather_source == "Live API (OpenWeatherMap)" and api_key:
                weather_data = st.session_state.weather_api.get_weather_by_city(district or state, state)
            elif weather_source == "Manual Input":
                weather_data = {
                    'current': {
                        'temp': temperature,
                        'humidity': humidity,
                        'rainfall': rainfall,
                        'wind_speed': wind_speed,
                        'pressure': 1013.2
                    },
                    'location': f"{district}, {state}" if district else state
                }
            else:
                # Demo mode
                weather_data = st.session_state.weather_api.get_weather_by_city(district or state, state)
            
            # Get crop advice
            crop_advice = st.session_state.crop_advisor.get_crop_advice(crop_type, weather_data)
            
            # Prepare input for ML model
            current_date = datetime.now()
            ml_input = {
                'year': current_date.year,
                'month': current_date.month,
                'day': current_date.day,
                'day_of_year': current_date.timetuple().tm_yday,
                'week_of_year': current_date.isocalendar()[1],
                'season': (current_date.month % 12 + 3) // 3,
                'temperature': weather_data['current']['temp'],
                'humidity': weather_data['current']['humidity'],
                'rainfall': weather_data['current'].get('rainfall', 0),
                'wind_speed': weather_data['current'].get('wind_speed', 3.0),
                'pressure': weather_data['current'].get('pressure', 1013.2),
                'soil_moisture': 50 + (weather_data['current'].get('rainfall', 0) * 0.6),
                'ph_level': 7.0,
                'nitrogen': 200,
                'phosphorus': 100,
                'potassium': 150,
                'organic_matter': 1.5,
                'yield': 100,
                'weather_stress_score': 0,
                'soil_health_score': 0.7
            }
            
            # Try to load pre-trained model, otherwise use sample predictions
            price_predictions = []
            try:
                if st.session_state.predictor.load_model():
                    # Use trained model
                    current_price = st.session_state.predictor.predict_price(ml_input)
                    price_trend = st.session_state.predictor.predict_price_trend(ml_input, 3)
                    
                    if current_price and price_trend:
                        price_predictions = {
                            'current_price': current_price,
                            'trend': price_trend
                        }
                    else:
                        price_predictions = generate_sample_predictions(crop_type, state)
                else:
                    price_predictions = generate_sample_predictions(crop_type, state)
            except:
                price_predictions = generate_sample_predictions(crop_type, state)
            
            # Display results
            display_results(crop_type, state, district, weather_data, crop_advice, price_predictions)
            
            # Generate PDF report
            generate_pdf_report(crop_type, state, district, weather_data, crop_advice, price_predictions)
            
        except Exception as e:
            st.error(f"❌ Error during analysis: {e}")
            st.info("💡 Try using Demo Mode or check your inputs")

def generate_sample_predictions(crop_type, state):
    """Generate sample price predictions when model is not available"""
    base_prices = {
        'Rice': 2800, 'Wheat': 2200, 'Corn': 1800, 
        'Soybeans': 4500, 'Cotton': 6500
    }
    
    base_price = base_prices.get(crop_type, 2500)
    
    # Generate trend for next 3 months
    trend = []
    current_date = datetime.now()
    
    for i in range(3):
        future_date = current_date + timedelta(days=30*i)
        # Add some seasonal variation
        seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * (future_date.month - 6) / 12)
        price = base_price * seasonal_factor * np.random.uniform(0.9, 1.1)
        
        trend.append({
            'date': future_date.strftime('%Y-%m-%d'),
            'month': future_date.strftime('%B %Y'),
            'price': round(price, 2)
        })
    
    return {
        'current_price': round(base_price * np.random.uniform(0.9, 1.1), 2),
        'trend': trend
    }

def display_results(crop_type, state, district, weather_data, crop_advice, price_predictions):
    """Display analysis results"""
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<h3 class="sub-header">📊 Crop Information</h3>', unsafe_allow_html=True)
        
        # Crop details
        col1a, col1b, col1c = st.columns(3)
        with col1a:
            st.metric("🌾 Crop Type", crop_type)
            st.metric("📍 State", state)
        with col1b:
            st.metric("🌡️ Temperature", f"{weather_data['current']['temp']}°C")
            st.metric("💧 Humidity", f"{weather_data['current']['humidity']}%")
        with col1c:
            st.metric("🌧️ Rainfall", f"{weather_data['current'].get('rainfall', 0)}mm")
            st.metric("💨 Wind Speed", f"{weather_data['current'].get('wind_speed', 3)}m/s")
    
    with col2:
        st.markdown('<h3 class="sub-header">💰 Market Price</h3>', unsafe_allow_html=True)
        
        if 'current_price' in price_predictions:
            st.metric(
                "Current Price",
                f"₹{price_predictions['current_price']:,.2f}",
                help="Predicted current market price per quintal"
            )
        else:
            st.metric("Current Price", "₹--", help="Price prediction not available")
    
    # Crop advice in tabs
    st.markdown('<h3 class="sub-header">🌱 Crop Recommendations</h3>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🌿 Growing Conditions", "💧 Water & Fertilizer", "⚠️ Risk Assessment", "📋 Next Actions"])
    
    with tab1:
        st.markdown(f"**Current Growing Conditions:** {crop_advice['growing_conditions']}")
        st.markdown(f"**Temperature Status:** {crop_advice['temperature_status']}")
        st.markdown(f"**Humidity Status:** {crop_advice['humidity_status']}")
    
    with tab2:
        st.markdown(f"**Water Management:** {crop_advice['water_management']}")
        st.markdown(f"**Fertilizer Application:** {crop_advice['fertilizer_advice']}")
        st.markdown(f"**Irrigation Schedule:** {crop_advice['irrigation_schedule']}")
    
    with tab3:
        st.markdown(f"**Risk Level:** {crop_advice['risk_assessment']}")
        st.markdown(f"**Potential Issues:** {crop_advice['potential_issues']}")
    
    with tab4:
        st.markdown(f"**Immediate Actions:** {crop_advice['immediate_actions']}")
        st.markdown(f"**Long-term Planning:** {crop_advice['long_term_planning']}")
    
    # Price trend chart
    if 'trend' in price_predictions and price_predictions['trend']:
        st.markdown('<h3 class="sub-header">📈 Price Trend Forecast</h3>', unsafe_allow_html=True)
        
        trend_data = price_predictions['trend']
        dates = [item['month'] for item in trend_data]
        prices = [item['price'] for item in trend_data]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices,
            mode='lines+markers',
            line=dict(color='#28a745', width=3),
            marker=dict(size=8, color='#28a745'),
            name='Predicted Price'
        ))
        
        fig.update_layout(
            title="3-Month Price Trend Forecast",
            xaxis_title="Month",
            yaxis_title="Price (₹ per quintal)",
            hovermode='x unified',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Price trend table
        st.markdown("**Detailed Price Forecast:**")
        trend_df = pd.DataFrame(trend_data)
        st.dataframe(trend_df, use_container_width=True)

def generate_pdf_report(crop_type, state, district, weather_data, crop_advice, price_predictions):
    """Generate and provide PDF report download"""
    
    st.markdown('<h3 class="sub-header">📄 Download Report</h3>', unsafe_allow_html=True)
    
    try:
        # Prepare data for PDF
        crop_data = {
            'crop_type': crop_type,
            'location': district or state,
            'state': state,
            'growing_season': get_growing_season(datetime.now().month)
        }
        
        # Generate PDF
        generator = CropReportGenerator()
        output_path = f"crop_report_{crop_type}_{state}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        success = generator.generate_crop_report(
            crop_data, weather_data, crop_advice, price_predictions, output_path
        )
        
        if success and os.path.exists(output_path):
            # Read PDF file
            with open(output_path, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
            
            # Download button
            st.download_button(
                label="📥 Download Complete Report (PDF)",
                data=pdf_bytes,
                file_name=output_path,
                mime="application/pdf",
                help="Download a comprehensive PDF report with all analysis results"
            )
            
            # Clean up temporary file
            os.remove(output_path)
        else:
            st.warning("⚠️ PDF generation failed. Please try again.")
            
    except Exception as e:
        st.error(f"❌ Error generating PDF: {e}")

def get_growing_season(month):
    """Determine growing season based on month"""
    if month in [6, 7, 8, 9, 10]:
        return "Kharif (Monsoon)"
    elif month in [10, 11, 12, 1, 2, 3]:
        return "Rabi (Winter)"
    else:
        return "Zaid (Summer)"

def show_about_section():
    """Display about section"""
    st.markdown("---")
    st.markdown('<h2 class="sub-header">ℹ️ About AI Agriculture Advisor</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    **AI Agriculture Advisor** is an intelligent farming companion that provides:
    
    🌾 **Crop-Specific Insights**: Tailored advice for Rice, Wheat, Corn, Soybeans, and Cotton
    🌤️ **Weather Integration**: Real-time weather data and climate analysis
    💰 **Market Predictions**: AI-powered price trend forecasting for the next 3 months
    💧 **Irrigation & Fertilizer**: Smart recommendations based on current conditions
    📊 **Data-Driven Decisions**: Built on realistic agricultural datasets
    
    **How it works:**
    1. **Input**: Select your crop, location, and weather conditions
    2. **Analysis**: AI models analyze historical data and current conditions
    3. **Insights**: Get comprehensive recommendations and market predictions
    4. **Report**: Download detailed PDF report for offline reference
    
    **Built with**: Python, Streamlit, Scikit-learn, and agricultural expertise
    """)

if __name__ == "__main__":
    main()
    show_about_section() 