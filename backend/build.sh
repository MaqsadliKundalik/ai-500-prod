# Render Build Script
# This file is executed during the build phase

#!/bin/bash
set -e

echo "🔨 Building Sentinel-RX..."

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Build completed successfully!"
