import os
import subprocess
import sys

def main():
    # Check if model exists
    if not os.path.exists("models/regression_model.pkl"):
        print("Model not found. Training model...")
        result = subprocess.run([sys.executable, "train_model.py"])
        if result.returncode != 0:
            print("Model training failed.")
            return
    
    # Run streamlit
    print("Starting Streamlit App...")
    # Use -m streamlit run app.py to be safe
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

if __name__ == "__main__":
    main()
