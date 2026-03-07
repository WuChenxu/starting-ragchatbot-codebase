#!/bin/bash

# Code Formatting Script
# This script formats all Python code using Black and sorts imports using isort

set -e

echo "======================================"
echo "  Running Code Formatters"
echo "======================================"
echo ""

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "📍 Project root: $PROJECT_ROOT"
echo ""

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed. Please install uv first."
    echo "   Visit: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

echo "🎨 Running Black (code formatter)..."
uv run black backend/ main.py
echo "✅ Black formatting complete!"
echo ""

echo "📋 Running isort (import sorter)..."
uv run isort backend/ main.py
echo "✅ Import sorting complete!"
echo ""

echo "======================================"
echo "  Formatting Complete! ✨"
echo "======================================"
