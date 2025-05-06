#!/bin/bash

# Tạo thư mục data nếu chưa tồn tại
mkdir -p data

# Tạo môi trường ảo Python
echo "Creating Python virtual environment..."
python3 -m venv venv

# Kích hoạt môi trường ảo
echo "Activating virtual environment..."
source venv/bin/activate

# Cài đặt các dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Tạo thư mục .kaggle nếu chưa tồn tại
mkdir -p ~/.kaggle

# Kiểm tra xem đã có kaggle.json chưa
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo "Please place your kaggle.json file in ~/.kaggle/"
    echo "You can download it from https://www.kaggle.com/settings/account"
    exit 1
fi

# Cấp quyền thực thi cho kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# Tải dataset
echo "Downloading dataset..."
kaggle datasets download phnvnh/preprocessed-auxusd -p data/

# Giải nén dataset
echo "Extracting dataset..."
unzip -o data/preprocessed-auxusd.zip -d data/

# Xóa file zip sau khi giải nén
rm data/preprocessed-auxusd.zip

echo "Setup completed successfully!"
echo "To activate the virtual environment, run: source venv/bin/activate" 