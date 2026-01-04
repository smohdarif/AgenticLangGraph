#!/bin/bash

# Quickstart Script for Agentic RAG System
# This script helps you set up and run the application

echo "🤖 Agentic RAG System - Quickstart"
echo "=================================="
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate
echo ""

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt
echo "✅ Dependencies installed"
echo ""

# Run the application
echo "🚀 Starting Streamlit app..."
echo ""
echo "👉 The app will open in your browser at http://localhost:8501"
echo "👉 Press Ctrl+C to stop the server"
echo ""
streamlit run agentic_rag_app.py
