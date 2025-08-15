# run_dashboard.py
"""
Streamlit Dashboard Entry Point

Usage:
    python run_dashboard.py

For development:
    streamlit run app/dashboard.py --server.runOnSave true
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit dashboard."""
    
    # Check if data exists
    if not os.path.exists("data/sentiment.db"):
        print("⚠️  No data found. Please run data ingestion first:")
        print("python ingestion/orchestrator.py")
        print("\nOr run the dashboard anyway to see the template:")
        
    # Run Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app/dashboard.py",
            "--server.runOnSave", "true"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")

if __name__ == "__main__":
    main()