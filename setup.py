#!/usr/bin/env python3
"""
Setup script for the Titanic Survival Predictor project
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages. Please install manually:")
        print("   pip install pandas numpy scikit-learn matplotlib seaborn")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Python 3.7 or higher is required!")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def main():
    """Main setup function"""
    print("🚢" * 25)
    print("🚢 TITANIC SURVIVAL PREDICTOR SETUP")
    print("🚢" * 25)
    
    print("\n🔧 Setting up your machine learning environment...")
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    print("\n🎉 Setup complete!")
    print("\n🚀 You can now run:")
    print("   • python titanic_predictor.py    (Complete analysis)")
    print("   • python interactive_demo.py    (Interactive demo)")
    
    print("\n📚 What's included:")
    print("   • README.md                     (Learning guide)")
    print("   • titanic_predictor.py          (Main ML pipeline)")
    print("   • interactive_demo.py           (Interactive predictor)")
    print("   • requirements.txt              (Dependencies)")
    
    print("\n🎓 Happy learning!")

if __name__ == "__main__":
    main() 