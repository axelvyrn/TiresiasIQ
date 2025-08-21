#!/usr/bin/env python3
"""
TiresiasIQ v2 - Behavior Prediction Engine
Simple startup script for the Streamlit web dashboard
"""

import subprocess
import sys
import os

def main():
    print("🚀 Starting TiresiasIQ v2 - Behavior Prediction Engine")
    print("📱 Opening Streamlit web dashboard...")
    print("🌐 The app will open in your default web browser")
    print("📊 Dashboard URL: http://localhost:8501")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Run streamlit app
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n👋 Shutting down TiresiasIQ v2...")
    except Exception as e:
        print(f"❌ Error starting the app: {e}")
        print("💡 Make sure you have installed all dependencies:")
        print("   pip install -r requirements.txt")
        print("   python -m spacy download en_core_web_sm")

if __name__ == "__main__":
    main()
