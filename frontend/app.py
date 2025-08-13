"""
AI Agriculture Advisor - Streamlit Frontend
Main application interface for farmers to get crop advice and price predictions.
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

from ml_pipeline import CropPricePredictor
from weather_api import WeatherAPI, CropAdvisor
from pdf_generator import CropReportGenerator

# Page configuration
st.set_page_config(
    page_title="AI Agriculture Advisor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for responsive design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E8B57, #3CB371);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .weather-card {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .advice-card {
        background: linear-gradient(135deg, #f3e5f5, #e1bee7);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #9c27b0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .price-card {
        background: linear-gradient(135deg, #fff3e0, #ffe0b2);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #28a745, #20c997);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa, #e9ecef);
    }
    
    @media (max-width: 768px) {
        .main-header {
            padding: 1rem;
            font-size: 1.5rem;
        }
        
        .metric-card, .weather-card, .advice-card, .price-card {
            padding: 1rem;
            margin: 0.5rem 0;
        }
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Initialize session state
    if 'predictor' not in st.session_state:
        st.session_state.predictor = None
    if 'weather_api' not in st.session_state:
        st.session_state.weather_api = None
    if 'crop_advisor' not in st.session_state:
        st.session_state.crop_advisor = None
    if 'pdf_generator' not in st.session_state:
        st.session_state.pdf_generator = None
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌾 AI Agriculture Advisor</h1>
        <p>Get intelligent crop advice, weather insights, and price predictions for better farming decisions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("📝 Input Parameters")
        
        # Crop selection
        crop_type = st.selectbox(
            "Select Crop Type",
            ["Rice", "Wheat", "Corn", "Soybeans", "Cotton"],
            help="Choose the crop you want to grow or are currently growing"
        )
        
        # Location inputs
        st.subheader("📍 Location")
        state = st.selectbox(
            "Select State",
            ["Punjab", "Haryana", "Uttar Pradesh", "Madhya Pradesh", "Rajasthan", 
             "Gujarat", "Maharashtra", "Karnataka", "Tamil Nadu", "Andhra Pradesh",
             "Telangana", "Bihar", "West Bengal", "Odisha", "Assam"]
        )
        
        district = st.text_input(
            "Enter District/City",
            placeholder="e.g., Ludhiana, Amritsar",
            help="Enter your specific district or city for accurate weather data"
        )
        
        # Weather options
        st.subheader("🌤️ Weather Data")
        use_weather_api = st.checkbox(
            "Fetch Live Weather Data",
            value=True,
            help="Enable to get real-time weather data (requires API key)"
        )
        
        if use_weather_api:
            api_key = st.text_input(
                "OpenWeatherMap API Key (Optional)",
                type="password",
                help="Enter your API key for real-time weather data, or leave empty for demo mode"
            )
        else:
            api_key = None
            st.info("Using manual weather input")
        
        # Manual weather inputs (if not using API)
        if not use_weather_api:
            st.subheader("🌡️ Manual Weather Input")
            temperature = st.slider("Temperature (°C)", -10, 50, 25)
            humidity = st.slider("Humidity (%)", 0, 100, 60)
            rainfall = st.slider("Recent Rainfall (mm)", 0, 100, 0)
        
        # Analysis button
        analyze_button = st.button(
            "🚀 Analyze & Get Recommendations",
            type="primary",
            use_container_width=True
        )
    
    # Main content area
    if analyze_button:
        with st.spinner("Analyzing crop conditions and generating recommendations..."):
            # Initialize components
            if st.session_state.predictor is None:
                st.session_state.predictor = CropPricePredictor()
            
            if st.session_state.weather_api is None:
                st.session_state.weather_api = WeatherAPI(api_key)
            
            if st.session_state.crop_advisor is None:
                st.session_state.crop_advisor = CropAdvisor()
            
            if st.session_state.pdf_generator is None:
                st.session_state.pdf_generator = CropReportGenerator()
            
            # Get weather data
            if use_weather_api:
                weather_data = st.session_state.weather_api.get_weather_by_city(district or state, state)
                if 'demo_mode' in weather_data:
                    st.warning("⚠️ Using demo weather data. For real-time data, provide OpenWeatherMap API key.")
            else:
                weather_data = {
                    'temperature': temperature,
                    'humidity': humidity,
                    'rainfall': rainfall,
                    'description': 'Manual input',
                    'demo_mode': True
                }
            
            # Prepare input for ML model
            current_month = datetime.now().month
            current_year = datetime.now().year
            
            ml_input = {
                'crop_type': crop_type,
                'location': district or state,
                'state': state,
                'temperature': weather_data['temperature'],
                'humidity': weather_data['humidity'],
                'rainfall': weather_data.get('rainfall', 0),
                'year': current_year,
                'month': current_month,
                'season': (current_month % 12 + 3) // 3,  # 1-4 for seasons
                'quarter': (current_month - 1) // 3 + 1
            }
            
            # Get crop advice
            crop_advice = st.session_state.crop_advisor.get_crop_advice(crop_type, weather_data)
            
            # Get price predictions
            try:
                # Try to load pre-trained model
                if not st.session_state.predictor.load_model():
                    # If no model, create sample predictions
                    st.info("ℹ️ Using sample price predictions. Train the model with real data for accurate predictions.")
                    price_predictions = {
                        'current_price': np.random.normal(2500, 500),
                        'trend': {
                            'dates': [(datetime.now() + timedelta(days=30*i)).strftime('%Y-%m-%d') for i in range(1, 4)],
                            'predictions': [np.random.normal(2500, 500) for _ in range(3)]
                        }
                    }
                else:
                    # Get predictions from trained model
                    current_price = st.session_state.predictor.predict_price(ml_input)
                    dates, predictions = st.session_state.predictor.predict_price_trend(ml_input, 3)
                    
                    price_predictions = {
                        'current_price': current_price,
                        'trend': {
                            'dates': dates,
                            'predictions': predictions
                        }
                    }
            except Exception as e:
                st.error(f"Error getting price predictions: {str(e)}")
                price_predictions = None
            
            # Display results
            display_results(crop_type, state, district, weather_data, crop_advice, price_predictions)
            
            # Generate and provide PDF download
            generate_pdf_report(crop_type, state, district, weather_data, crop_advice, price_predictions)

def display_results(crop_type, state, district, weather_data, crop_advice, price_predictions):
    """Display analysis results in a structured format"""
    
    # Results header
    st.success("✅ Analysis Complete! Here are your personalized recommendations:")
    
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Crop Information
        st.markdown("""
        <div class="metric-card">
            <h3>🌾 Crop Information</h3>
        </div>
        """, unsafe_allow_html=True)
        
        crop_info_data = {
            "Crop Type": crop_type,
            "Location": district or state,
            "State": state,
            "Growing Season": get_growing_season(datetime.now().month)
        }
        
        crop_df = pd.DataFrame(list(crop_info_data.items()), columns=["Parameter", "Value"])
        st.dataframe(crop_df, use_container_width=True, hide_index=True)
    
    with col2:
        # Current Weather Summary
        st.markdown("""
        <div class="weather-card">
            <h3>🌤️ Current Weather</h3>
        </div>
        """, unsafe_allow_html=True)
        
        weather_summary = f"""
        **Temperature:** {weather_data.get('temperature', 'N/A')}°C  
        **Humidity:** {weather_data.get('humidity', 'N/A')}%  
        **Conditions:** {weather_data.get('description', 'N/A')}
        """
        st.markdown(weather_summary)
    
    # Crop Advice Section
    st.markdown("""
    <div class="advice-card">
        <h3>💡 Crop-Specific Advice</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if 'error' in crop_advice:
        st.error(f"Error: {crop_advice['error']}")
    else:
        # Create tabs for different advice categories
        advice_tabs = st.tabs(["🌱 Growing Conditions", "💧 Water & Fertilizer", "⚠️ Risk Assessment", "📋 Next Actions"])
        
        with advice_tabs[0]:
            st.subheader("Current Growing Conditions")
            conditions_data = crop_advice.get('current_conditions', {})
            for key, value in conditions_data.items():
                st.metric(key.replace('_', ' ').title(), value)
        
        with advice_tabs[1]:
            st.subheader("Water & Fertilizer Management")
            recommendations = crop_advice.get('recommendations', {})
            for key, value in recommendations.items():
                if 'water' in key.lower() or 'fertilizer' in key.lower() or 'irrigation' in key.lower():
                    st.info(f"**{key.replace('_', ' ').title()}:** {value}")
        
        with advice_tabs[2]:
            st.subheader("Risk Assessment")
            risk_assessment = crop_advice.get('risk_assessment', {})
            risk_level = risk_assessment.get('risk_level', 'Unknown')
            
            # Color-coded risk level
            if risk_level == 'Low':
                st.success(f"**Risk Level: {risk_level}**")
            elif risk_level == 'Medium':
                st.warning(f"**Risk Level: {risk_level}**")
            else:
                st.error(f"**Risk Level: {risk_level}**")
            
            risks = risk_assessment.get('risks', [])
            for risk in risks:
                st.write(f"• {risk}")
        
        with advice_tabs[3]:
            st.subheader("Recommended Actions")
            actions = crop_advice.get('next_actions', [])
            for i, action in enumerate(actions, 1):
                st.write(f"{i}. {action}")
    
    # Price Predictions Section
    if price_predictions:
        st.markdown("""
        <div class="price-card">
            <h3>💰 Market Price Predictions</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            current_price = price_predictions.get('current_price', 0)
            st.metric("Current Market Price", f"₹{current_price:.2f}")
        
        with col2:
            # Price trend chart
            trend_data = price_predictions.get('trend', {})
            if trend_data and 'dates' in trend_data and 'predictions' in trend_data:
                dates = trend_data['dates']
                predictions = trend_data['predictions']
                
                # Create price trend chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=predictions,
                    mode='lines+markers',
                    name='Predicted Price',
                    line=dict(color='#ff9800', width=3),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title="Price Trend Forecast (Next 3 Months)",
                    xaxis_title="Date",
                    yaxis_title="Price (₹)",
                    hovermode='x unified',
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    # Weather Forecast Section
    st.markdown("""
    <div class="weather-card">
        <h3>🌦️ Weather Forecast</h3>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        forecast_data = st.session_state.weather_api.get_weather_forecast(district or state, state, days=5)
        
        if forecast_data and 'forecast' in forecast_data:
            forecast_df = pd.DataFrame(forecast_data['forecast'])
            
            # Create weather forecast chart
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['temp_max'],
                mode='lines+markers',
                name='Max Temperature',
                line=dict(color='#ff6b6b', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['temp_min'],
                mode='lines+markers',
                name='Min Temperature',
                line=dict(color='#4ecdc4', width=3),
                fill='tonexty'
            ))
            
            fig.update_layout(
                title="5-Day Temperature Forecast",
                xaxis_title="Date",
                yaxis_title="Temperature (°C)",
                hovermode='x unified',
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Display forecast table
            st.subheader("Detailed Forecast")
            forecast_display = forecast_df[['date', 'temp_min', 'temp_max', 'humidity', 'description']].copy()
            forecast_display.columns = ['Date', 'Min Temp (°C)', 'Max Temp (°C)', 'Humidity (%)', 'Conditions']
            st.dataframe(forecast_display, use_container_width=True, hide_index=True)
        
    except Exception as e:
        st.error(f"Error fetching weather forecast: {str(e)}")

def generate_pdf_report(crop_type, state, district, weather_data, crop_advice, price_predictions):
    """Generate and provide PDF report download"""
    
    st.markdown("""
    <div class="metric-card">
        <h3>📄 Download Complete Report</h3>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        # Prepare data for PDF
        crop_data = {
            'crop_type': crop_type,
            'location': district or state,
            'state': state,
            'season': get_growing_season(datetime.now().month)
        }
        
        # Generate PDF
        output_path = f"../data/crop_report_{crop_type}_{state}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        success = st.session_state.pdf_generator.generate_crop_report(
            crop_data,
            weather_data,
            crop_advice,
            price_predictions,
            output_path
        )
        
        if success:
            # Read the generated PDF
            with open(output_path, "rb") as pdf_file:
                pdf_bytes = pdf_file.read()
            
            # Provide download button
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=f"AI_Agriculture_Advisor_Report_{crop_type}_{state}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
            
            st.success("✅ PDF report generated successfully! Click the button above to download.")
            
            # Clean up the temporary file
            try:
                os.remove(output_path)
            except:
                pass
        else:
            st.error("❌ Failed to generate PDF report. Please try again.")
    
    except Exception as e:
        st.error(f"Error generating PDF report: {str(e)}")

def get_growing_season(month):
    """Get growing season based on month"""
    if month in [6, 7, 8, 9]:
        return "Kharif (Monsoon)"
    elif month in [10, 11, 12, 1, 2, 3]:
        return "Rabi (Winter)"
    else:
        return "Zaid (Summer)"

def show_about_section():
    """Show about section with project information"""
    st.markdown("""
    <div class="metric-card">
        <h3>ℹ️ About AI Agriculture Advisor</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **AI Agriculture Advisor** is an intelligent farming assistant that provides:
    
    🌾 **Crop-Specific Advice**: Tailored recommendations based on your crop type and local conditions
    
    🌤️ **Weather Intelligence**: Real-time weather data and forecasts for optimal farming decisions
    
    💰 **Price Predictions**: Market trend analysis to help with harvest timing and marketing
    
    📊 **Data-Driven Insights**: Machine learning-powered analysis for better crop management
    
    📱 **Mobile-Friendly**: Responsive design that works on all devices
    
    ---
    
    **How it works:**
    1. Select your crop type and location
    2. Get real-time weather data (or input manually)
    3. Receive personalized farming advice
    4. View price predictions and market trends
    5. Download comprehensive PDF reports
    
    **Note**: This application uses demo data for demonstration. For production use, 
    provide real datasets and OpenWeatherMap API key for accurate predictions.
    """)

# Main execution
if __name__ == "__main__":
    main()
    
    # Show about section at the bottom
    st.markdown("---")
    show_about_section() 