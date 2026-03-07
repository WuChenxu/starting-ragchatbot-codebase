#!/bin/bash

# Linting Script
# This script runs comprehensive code quality checks including:
# - Black (formatting check)
# - isort (import order check)
# - Basic syntax validation

set -e

echo "======================================"
echo "  Running Linting Suite"
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

EXIT_CODE=0

echo "🔍 Running Black (formatting check)..."
if uv run black --check --diff backend/ main.py; then
    echo "✅ Black check passed!"
else
    echo ""
    echo "⚠️  Black formatting issues found."
    EXIT_CODE=1
fi
echo ""

echo "🔍 Running isort (import order check)..."
if uv run isort --check-only --diff backend/ main.py; then
    echo "✅ isort check passed!"
else
    echo ""
    echo "⚠️  Import order issues found."
    EXIT_CODE=1
fi
echo ""

echo "🔍 Validating Python syntax..."
if uv run python -m py_compile backend/*.py main.py; then
    echo "✅ Syntax validation passed!"
else
    echo ""
    echo "❌ Syntax errors found."
    EXIT_CODE=1
fi
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "======================================"
    echo "  All Linting Checks Passed! ✅"
    echo "======================================"
else
    echo "======================================"
    echo "  Linting Issues Found ⚠️"
    echo "======================================"
    echo ""
    echo "Run './scripts/format.sh' to fix formatting issues."
    exit 1
fi
