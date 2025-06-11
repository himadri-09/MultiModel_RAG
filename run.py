import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False
    return True

def run_streamlit():
    """Run the Streamlit app"""
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
    except KeyboardInterrupt:
        print("Application stopped by user.")
    except Exception as e:
        print(f"Error running Streamlit: {e}")

if __name__ == "__main__":
    print("Multimodal PDF RAG System")
    print("=" * 50)
    
    # Check if requirements should be installed
    if len(sys.argv) > 1 and sys.argv[1] == "--install":
        if install_requirements():
            print("Setup complete! Run 'python run.py' to start the application.")
        sys.exit()
    
    # Run the application
    print("Starting Streamlit application...")
    print("Open your browser and go to: http://localhost:8501")
    print("Press Ctrl+C to stop the application.")
    run_streamlit()
