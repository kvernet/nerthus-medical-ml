#!/usr/bin/env python3
"""
Runner script for the Nerthus Medical ML Streamlit app
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit app"""
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    
    print("🚀 Starting Nerthus Medical ML Web App...")
    print("📱 Open your browser to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    
    # Run streamlit
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path])

if __name__ == "__main__":
    main()