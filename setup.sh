#!/bin/bash

# Mini RAG Application Setup Script
# This script helps you set up the Mini RAG application

echo "🚀 Mini RAG Application Setup"
echo "=============================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ and try again."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed. Please install pip and try again."
    exit 1
fi

echo "✅ pip found: $(pip --version)"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing Python dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys:"
    echo "   - OPENAI_API_KEY"
    echo "   - PINECONE_API_KEY"
    echo "   - PINECONE_ENVIRONMENT"
    echo "   - COHERE_API_KEY"
    echo ""
fi

# Check if .env file has been configured
if grep -q "your_.*_key_here" .env; then
    echo "⚠️  Warning: .env file still contains placeholder values."
    echo "   Please update your API keys before running the application."
    echo ""
fi

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run: source venv/bin/activate"
echo "3. Run: python main.py"
echo "4. Open: http://localhost:8000"
echo ""
echo "Happy coding! 🎉"