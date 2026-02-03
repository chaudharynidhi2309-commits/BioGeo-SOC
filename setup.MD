#!/bin/bash

echo "========================================"
echo "🌱 BioGeo-SOC Setup Script"
echo "========================================"
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+')
if [ -z "$python_version" ]; then
    echo "❌ Python 3 not found. Please install Python 3.11 or higher."
    exit 1
fi
echo "✅ Python $python_version found"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
echo "✅ Virtual environment created"
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"
echo ""

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models
mkdir -p data
mkdir -p reports
echo "✅ Directories created"
echo ""

# Train model
echo "🤖 Training model (this may take a few minutes)..."
python src/train.py --augment
echo "✅ Model trained"
echo ""

echo "========================================"
echo "✅ Setup complete!"
echo "========================================"
echo ""
echo "To run the application:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run: streamlit run app.py"
echo ""
echo "For command-line predictions:"
echo "  python main.py --lat 22.75 --lon 72.45"
echo ""
echo "========================================"