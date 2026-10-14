#!/usr/bin/env bash
# Create a virtual environment and install TBC dependencies.
# Usage: ./setup.sh
set -euo pipefail

PYTHON="${PYTHON:-python3}"

echo "Creating virtual environment in .venv/ ..."
"$PYTHON" -m venv .venv

# shellcheck disable=SC1091
source .venv/bin/activate

echo "Upgrading pip and installing requirements ..."
python -m pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Done. To use the environment:"
echo "  source .venv/bin/activate"
echo "  streamlit run app.py"
