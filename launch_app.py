"""
AI Agriculture Advisor - App Launcher
Simple script to launch the Streamlit application
"""

import os
import sys
import subprocess
import webbrowser
import time

def main():
    """Launch the AI Agriculture Advisor app"""
    
    print("🌾 AI Agriculture Advisor - Launching...")
    print("="*50)
    
    # Check if we're in the right directory
    if not os.path.exists('frontend/app_simple.py'):
        print("❌ Error: app_simple.py not found in frontend/ directory")
        print("💡 Make sure you're running this from the project root directory")
        return
    
    # Check if data exists
    if not os.path.exists('data/realistic_crop_data.csv'):
        print("❌ Error: Dataset not found!")
        print("💡 Please run 'python generate_data.py' first to create the dataset")
        return
    
    print("✅ All files found!")
    print("🚀 Launching Streamlit app...")
    
    try:
        # Change to frontend directory
        os.chdir('frontend')
        
        # Launch Streamlit
        print("🌐 Starting Streamlit server...")
        print("📱 The app will open in your browser automatically")
        print("🔗 If it doesn't open, go to: http://localhost:8501")
        print("\n⏹️  To stop the app, press Ctrl+C in this terminal")
        print("="*50)
        
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app_simple.py",
            "--server.port", "8501",
            "--server.headless", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 App stopped by user")
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        print("💡 Try running: streamlit run frontend/app_simple.py")

if __name__ == "__main__":
    main() 