#!/usr/bin/env python3
"""
AI Agriculture Advisor - Application Launcher
Simple script to launch the Streamlit application
"""

import os
import sys
import subprocess
import webbrowser
import time

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'scikit-learn', 
        'xgboost', 'lightgbm', 'plotly', 'reportlab'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Installing missing packages...")
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
            print("✅ All packages installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please run:")
            print("   pip install -r requirements.txt")
            return False
    
    return True

def create_sample_data():
    """Create sample dataset if none exists"""
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Check if sample data exists
    sample_file = os.path.join(data_dir, "sample_crop_data.csv")
    if not os.path.exists(sample_file):
        print("📊 Creating sample dataset...")
        try:
            # Import and run data loader
            sys.path.append('backend')
            from data_loader import CropDataLoader
            
            loader = CropDataLoader()
            loader.create_sample_dataset()
            print("✅ Sample dataset created successfully!")
        except Exception as e:
            print(f"⚠️ Warning: Could not create sample dataset: {e}")
            print("   The app will work with demo data.")

def launch_app():
    """Launch the Streamlit application"""
    print("🚀 Launching AI Agriculture Advisor...")
    
    # Change to frontend directory
    frontend_dir = "frontend"
    if not os.path.exists(frontend_dir):
        print("❌ Frontend directory not found!")
        return False
    
    os.chdir(frontend_dir)
    
    # Launch Streamlit
    try:
        print("🌐 Starting Streamlit server...")
        print("📱 The app will open in your browser automatically")
        print("🔗 If it doesn't open, go to: http://localhost:8501")
        print("\n⏹️  Press Ctrl+C to stop the application")
        print("="*50)
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(3)
            webbrowser.open("http://localhost:8501")
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Launch Streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
        
    except KeyboardInterrupt:
        print("\n\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("🌾 AI Agriculture Advisor")
    print("="*40)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install dependencies and try again")
        return
    
    # Create sample data
    create_sample_data()
    
    # Launch application
    success = launch_app()
    
    if not success:
        print("\n❌ Failed to launch application")
        print("💡 Try running manually:")
        print("   cd frontend")
        print("   streamlit run app.py")

if __name__ == "__main__":
    main() 