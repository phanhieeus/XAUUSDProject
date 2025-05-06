#!/bin/bash

# Function to handle errors
handle_error() {
    echo "Error: $1"
    exit 1
}

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate || handle_error "Failed to activate virtual environment"

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt || handle_error "Failed to install requirements"

# Install gdown for Google Drive downloads
pip install gdown || handle_error "Failed to install gdown"

# Create necessary directories
mkdir -p data/processed
mkdir -p checkpoints

# Download data files using gdown
echo "Downloading data files..."

# Using new gdown syntax without --id flag
gdown "https://drive.google.com/uc?id=1QVA5hYGdwnMvyuKbj9UdtappXKqqXz1E" -O data/dynamic_labeled_train.csv || handle_error "Failed to download training data"
gdown "https://drive.google.com/uc?id=1N5-JMtmrEE7zN8rDHXjCG3-hgsBKWIvI" -O data/dynamic_labeled_dev.csv || handle_error "Failed to download dev data"
gdown "https://drive.google.com/uc?id=1pT7DuO6Ql2GEZg-hjZKTOWblLpEAe7nj" -O data/dynamic_labeled_test.csv || handle_error "Failed to download test data"

# Verify files were downloaded
for file in data/dynamic_labeled_{train,dev,test}.csv; do
    if [ ! -f "$file" ]; then
        handle_error "Failed to download $file"
    fi
done

echo "Setup completed successfully!"
echo "Next steps:"
echo "1. Run preprocessing: python utils/preprocess_data.py"
echo "2. Start training: python train.py" 