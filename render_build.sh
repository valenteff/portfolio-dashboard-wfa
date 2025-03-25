#!/usr/bin/env bash

# Exit on error
set -e

# Install dependencies
pip install -r requirements.txt

# Generate sample data for the dashboard (if needed)
python update_data.py

# Create assets directory if it doesn't exist
mkdir -p assets

# Print completion message
echo "Build completed successfully!" 