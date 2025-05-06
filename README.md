# XAUUSD Trading Model with Time-Aware Transformer

Dự án này sử dụng mô hình Time-Aware Transformer để dự đoán tín hiệu giao dịch (BUY/SELL/HOLD) cho cặp tiền tệ XAUUSD (Vàng) dựa trên dữ liệu lịch sử.

## Cấu trúc dự án

```
XAUUSDProject/
├── data/
│   ├── dynamic_labeled_train.csv    # Dữ liệu training gốc
│   ├── dynamic_labeled_dev.csv      # Dữ liệu validation gốc
│   ├── dynamic_labeled_test.csv     # Dữ liệu test gốc
│   └── processed/                   # Thư mục chứa dữ liệu đã xử lý
│       ├── train_processed.csv
│       ├── dev_processed.csv
│       └── test_processed.csv
├── models/
│   └── time_aware_transformer.py    # Mô hình Time-Aware Transformer
├── utils/
│   ├── data_utils.py               # Tiện ích xử lý dữ liệu
│   └── preprocess_data.py          # Script tiền xử lý dữ liệu
├── config.py                       # Cấu hình mô hình
├── train.py                        # Script training
└── README.md                       # Tài liệu hướng dẫn
```

## Cài đặt

1. Tạo môi trường Python và cài đặt các thư viện cần thiết:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

2. Đặt file `kaggle.json` vào thư mục `~/.kaggle/` để có thể tải dữ liệu từ Kaggle.

## Dữ liệu

Dự án sử dụng 3 file CSV chứa dữ liệu XAUUSD đã được gán nhãn:
- `dynamic_labeled_train.csv`: Dữ liệu training
- `dynamic_labeled_dev.csv`: Dữ liệu validation
- `dynamic_labeled_test.csv`: Dữ liệu test

Mỗi file có các cột sau:
- `Date`: Ngày giao dịch (YYYY-MM-DD)
- `Time`: Thời gian giao dịch (HH:MM:SS)
- `Close`: Giá đóng cửa
- `Volume`: Khối lượng giao dịch
- `Label`: Nhãn giao dịch (BUY/SELL/HOLD)

## Quy trình xử lý dữ liệu

1. Chạy script tiền xử lý dữ liệu:
```bash
python utils/preprocess_data.py
```

Script này sẽ:
- Đọc dữ liệu từ các file CSV gốc
- Kết hợp Date và Time thành timestamp
- Chuẩn hóa timestamp về khoảng [0,1]
- Chuẩn hóa Close và Volume về phân phối chuẩn
- Chuyển đổi Label thành số (0: HOLD, 1: BUY, 2: SELL)
- Lưu dữ liệu đã xử lý vào thư mục `data/processed/`

## Training

1. Chạy script training:
```bash
python train.py
```

Quá trình training sẽ:
- Tự động kiểm tra sự tồn tại của dữ liệu đã xử lý
- Khởi tạo mô hình Time-Aware Transformer
- Training với các tham số trong `config.py`
- Lưu model tốt nhất vào thư mục `checkpoints/`
- Hiển thị kết quả đánh giá trên tập validation và test

## Cấu hình

Các tham số mô hình có thể điều chỉnh trong file `config.py`:
- `SEQUENCE_LENGTH`: Độ dài chuỗi thời gian (mặc định: 128)
- `BATCH_SIZE`: Kích thước batch (mặc định: 16)
- `EMBEDDING_DIM`: Kích thước embedding (mặc định: 64)
- `NUM_HEADS`: Số lượng attention heads (mặc định: 8)
- `NUM_EPOCHS`: Số epoch training (mặc định: 20)
- `LEARNING_RATE`: Tốc độ học (mặc định: 0.0001)
- `DROPOUT`: Tỷ lệ dropout (mặc định: 0.1)
- `WEIGHT_DECAY`: Hệ số L2 regularization (mặc định: 1e-5)
- `TIME_DECAY_REG`: Hệ số regularization cho time decay (mặc định: 0.01)

## Đánh giá

Mô hình được đánh giá dựa trên:
- Accuracy: Tỷ lệ dự đoán đúng
- Classification Report: Precision, Recall, F1-score cho từng lớp
- Loss: Cross-entropy loss trên tập validation và test

## Lưu ý

- Đảm bảo đã chạy script tiền xử lý dữ liệu trước khi training
- Model tốt nhất được lưu tại `checkpoints/best_model.pth`
- Có thể điều chỉnh các tham số trong `config.py` để tối ưu hiệu suất 

Input (Close, Volume) ──┐
                       ├─> Feature Embedding ─┐
Timestamp ────────────┐│                     │
                      │└─> Time2Vec ─────────┼─> Combined Embedding ─> Time-Aware Attention ─> Classification Head