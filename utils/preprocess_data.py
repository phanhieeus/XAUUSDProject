import pandas as pd
import os

def preprocess_data(input_path, output_path):
    """
    Xử lý dữ liệu và lưu thành file CSV mới đã chuẩn hóa
    
    Args:
        input_path: đường dẫn đến file CSV gốc
        output_path: đường dẫn để lưu file CSV đã xử lý
    """
    print(f"Processing {input_path}...")
    
    # Đọc dữ liệu
    df = pd.read_csv(input_path)
    
    # Kết hợp Date và Time thành timestamp
    df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    
    # Normalize timestamps to seconds from start
    df['timestamp_norm'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds()
    df['timestamp_norm'] = df['timestamp_norm'] / df['timestamp_norm'].max()
    
    # Normalize Close và Volume
    for col in ['Close', 'Volume']:
        df[col] = (df[col] - df[col].mean()) / df[col].std()
    
    # Chuyển Label thành số
    LABEL_MAP = {"HOLD": 0, "BUY": 1, "SELL": 2}
    df['action'] = df['Label'].map(LABEL_MAP)
    
    # Chọn và sắp xếp các cột cần thiết
    df = df[['timestamp_norm', 'Close', 'Volume', 'action']]
    
    # Lưu file
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")

def main():
    # Tạo thư mục processed_data nếu chưa tồn tại
    os.makedirs("data/processed", exist_ok=True)
    
    # Xử lý từng file
    preprocess_data(
        "data/dynamic_labeled_train.csv",
        "data/processed/train_processed.csv"
    )
    preprocess_data(
        "data/dynamic_labeled_dev.csv",
        "data/processed/dev_processed.csv"
    )
    preprocess_data(
        "data/dynamic_labeled_test.csv",
        "data/processed/test_processed.csv"
    )
    
    print("\nData preprocessing completed!")

if __name__ == "__main__":
    main() 