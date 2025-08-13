# 🌾 AI Agriculture Advisor

An intelligent web application that provides farmers with data-driven crop advice, weather insights, and market price predictions to optimize farming operations.

## 🚀 Features

- **🌾 Crop-Specific Advice**: Tailored recommendations based on crop type and local conditions
- **🌤️ Real-Time Weather**: Live weather data integration with OpenWeatherMap API
- **💰 Price Predictions**: ML-powered market trend analysis for better harvest timing
- **📊 Data Visualization**: Interactive charts and graphs for weather and price trends
- **📱 Responsive Design**: Mobile-friendly interface that works on all devices
- **📄 PDF Reports**: Comprehensive downloadable reports with all recommendations
- **🤖 Machine Learning**: Multiple ML models (Random Forest, XGBoost, LightGBM) for accurate predictions

## 🏗️ Architecture

```
crop_project/
├── frontend/           # Streamlit web application
│   └── app.py         # Main Streamlit app with UI
├── backend/            # Core ML and business logic
│   ├── ml_pipeline.py # ML model training and prediction
│   ├── weather_api.py # Weather API integration and crop advice
│   └── pdf_generator.py # PDF report generation
├── data/               # Data storage and sample datasets
├── models/             # Trained ML models and preprocessing objects
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Python web framework)
- **Backend**: Python (Pandas, NumPy, scikit-learn)
- **ML Models**: Random Forest, XGBoost, LightGBM
- **Weather API**: OpenWeatherMap
- **Data Visualization**: Plotly, Matplotlib
- **PDF Generation**: ReportLab
- **Deployment**: Streamlit Cloud

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd crop_project
```

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables (Optional)

For real-time weather data, create a `.env` file in the project root:

```bash
OPENWEATHER_API_KEY=your_api_key_here
```

**Note**: You can get a free API key from [OpenWeatherMap](https://openweathermap.org/api)

### 5. Prepare Data (Optional)

The application includes sample data generation for testing. For production use:

1. Place your crop price dataset in the `data/` folder
2. Supported formats: CSV, Excel (.xlsx)
3. Required columns: date, crop_type, location, price, and other relevant features

## 🎯 Usage

### Running the Application

1. **Start the Streamlit app**:
   ```bash
   cd frontend
   streamlit run app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

### Using the Application

1. **Select Crop Type**: Choose from Rice, Wheat, Corn, Soybeans, or Cotton
2. **Enter Location**: Select state and enter district/city
3. **Weather Data**: Choose between live API data or manual input
4. **Get Analysis**: Click "Analyze & Get Recommendations"
5. **Review Results**: View crop advice, weather forecast, and price predictions
6. **Download Report**: Generate and download comprehensive PDF reports

## 🔧 Configuration

### Weather API Configuration

- **Demo Mode**: Works without API key (uses sample data)
- **Live Mode**: Requires OpenWeatherMap API key for real-time data
- **Fallback**: Automatically switches to demo mode if API fails

### ML Model Configuration

- **Auto-training**: Models train automatically on first run
- **Model Selection**: Automatically selects best-performing model
- **Persistence**: Trained models are saved and reused

## 📊 Data Requirements

### Sample Dataset Structure

```csv
date,crop_type,location,temperature,humidity,rainfall,yield,price
2019-01-01,Rice,Punjab,25.5,65,0,100,2500
2019-01-02,Rice,Punjab,26.2,68,5,98,2480
...
```

### Required Fields

- **date**: Date of observation
- **crop_type**: Type of crop (Rice, Wheat, etc.)
- **location**: Geographic location
- **price**: Target variable for prediction
- **Additional features**: temperature, humidity, rainfall, yield, etc.

## 🚀 Deployment

### Streamlit Cloud Deployment

1. **Push to GitHub**: Ensure your code is in a GitHub repository
2. **Connect to Streamlit Cloud**: 
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select the repository
3. **Configure Deployment**:
   - Set main file path: `frontend/app.py`
   - Add requirements.txt path: `requirements.txt`
4. **Deploy**: Click deploy and wait for build completion

### Local Production Deployment

```bash
# Install production dependencies
pip install gunicorn

# Run with gunicorn (if using Flask backend)
gunicorn -w 4 -b 0.0.0.0:8000 app:app

# Or use Streamlit's built-in server
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   # Ensure you're in the correct directory
   cd crop_project
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Weather API Issues**:
   - Check API key validity
   - Verify internet connection
   - Application automatically falls back to demo mode

3. **ML Model Errors**:
   - Ensure sufficient data for training
   - Check data format and required columns
   - Models will use sample predictions if training fails

4. **PDF Generation Issues**:
   - Check write permissions in data/ folder
   - Ensure ReportLab is properly installed

### Performance Optimization

- **Data Size**: Limit dataset size for faster processing
- **Model Caching**: Trained models are automatically cached
- **API Rate Limiting**: Respect OpenWeatherMap API limits

## 📈 Model Performance

### Current Model Metrics

- **Random Forest**: R² score varies by dataset
- **XGBoost**: Generally provides best performance
- **LightGBM**: Fast training and good accuracy
- **Linear Regression**: Baseline model for comparison

### Improving Predictions

1. **Better Data**: Use larger, more diverse datasets
2. **Feature Engineering**: Add relevant agricultural features
3. **Hyperparameter Tuning**: Optimize model parameters
4. **Ensemble Methods**: Combine multiple model predictions

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes** and add tests
4. **Commit your changes**: `git commit -m 'Add feature'`
5. **Push to the branch**: `git push origin feature-name`
6. **Submit a pull request**

### Development Guidelines

- Follow PEP 8 Python style guidelines
- Add docstrings to all functions
- Include error handling for robustness
- Test with sample data before committing

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenWeatherMap**: Weather data API
- **Streamlit**: Web application framework
- **scikit-learn**: Machine learning library
- **Indian Agricultural Community**: For domain expertise and feedback

## 📞 Support

For support and questions:

- **Issues**: Create an issue on GitHub
- **Documentation**: Check this README and code comments
- **Community**: Join agricultural technology forums

## 🔮 Future Enhancements

- **Multi-language Support**: Hindi, Punjabi, and other regional languages
- **Mobile App**: Native mobile application
- **IoT Integration**: Sensor data integration
- **Advanced Analytics**: Deep learning models
- **Market Integration**: Real-time market price feeds
- **Expert System**: Integration with agricultural experts

## 📊 Project Status

- ✅ **Core ML Pipeline**: Complete
- ✅ **Weather Integration**: Complete
- ✅ **Web Interface**: Complete
- ✅ **PDF Reports**: Complete
- 🔄 **Data Integration**: In Progress
- 🔄 **Model Optimization**: Ongoing
- 📋 **Deployment**: Ready for Streamlit Cloud

---

**🌾 Happy Farming with AI!** 🚀

*This project aims to democratize agricultural technology and help farmers make data-driven decisions for better crop yields and profitability.* 