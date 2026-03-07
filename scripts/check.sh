#!/bin/bash

# Code Quality Check Script
# This script checks code formatting without modifying files
# Useful for CI/CD pipelines and pre-commit validation

set -e

echo "======================================"
echo "  Running Code Quality Checks"
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

echo "🔍 Checking code formatting with Black..."
if uv run black --check backend/ main.py; then
    echo "✅ Black check passed!"
else
    echo ""
    echo "⚠️  Black formatting issues found. Run './scripts/format.sh' to fix."
    exit 1
fi
echo ""

echo "🔍 Checking import order with isort..."
if uv run isort --check-only backend/ main.py; then
    echo "✅ isort check passed!"
else
    echo ""
    echo "⚠️  Import order issues found. Run './scripts/format.sh' to fix."
    exit 1
fi
echo ""

echo "======================================"
echo "  All Checks Passed! ✅"
echo "======================================"
