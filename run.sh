#!/bin/bash
# Quick start script for FastAPI Microservice Backend

echo "🚀 Starting FastAPI Microservice Backend"
echo "========================================"

# Check Python version
if ! command -v python3.14 &> /dev/null; then
    echo "⚠️  Python 3.14 not found, using default python3"
    PYTHON_CMD=python3
else
    PYTHON_CMD=python3.14
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check for Tesseract
if ! command -v tesseract &> /dev/null; then
    echo "⚠️  Tesseract OCR not found!"
    echo "   Install it for full deskew functionality:"
    echo "   - macOS: brew install tesseract"
    echo "   - Ubuntu: sudo apt-get install tesseract-ocr"
    echo ""
fi

# Start the server
echo "✅ Starting server on http://localhost:8000"
echo "📚 API docs available at http://localhost:8000/docs"
echo ""
uvicorn main:app --reload --host 0.0.0.0 --port 8000
