import torch
from torch.utils.data import Dataset, DataLoader
import os

LABEL_MAP = {"HOLD": 0, "BUY": 1, "SELL": 2}

class XAUUSDDataset(Dataset):
    def __init__(self, data_path, sequence_length, transform=None):
        """
        Dataset cho XAUUSD với dữ liệu đã được xử lý
        
        Args:
            data_path: đường dẫn đến file CSV đã xử lý
            sequence_length: độ dài chuỗi thời gian
            transform: các biến đổi bổ sung (nếu có)
        """
        # Kiểm tra file tồn tại
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {data_path}")
            
        # Đọc dữ liệu đã được xử lý
        import pandas as pd
        self.data = pd.read_csv(data_path)
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Kiểm tra các cột cần thiết
        required_columns = ['Close', 'Volume', 'timestamp_norm', 'action']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        if missing_columns:
            raise ValueError(f"Thiếu các cột sau trong dữ liệu: {missing_columns}")
        
        # Lấy các cột cần thiết
        self.features = self.data[['Close', 'Volume']].values
        self.timestamps = self.data['timestamp_norm'].values
        self.labels = self.data['action'].values
        
        # Kiểm tra tính hợp lệ của dữ liệu
        if len(self.data) < sequence_length:
            raise ValueError(f"Độ dài dữ liệu ({len(self.data)}) nhỏ hơn sequence_length ({sequence_length})")
    
    def __len__(self):
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx):
        """
        Lấy một chuỗi dữ liệu
        
        Args:
            idx: chỉ số bắt đầu của chuỗi
            
        Returns:
            features: tensor chứa Close và Volume (shape: [sequence_length, 2])
            timestamps: tensor chứa timestamp đã chuẩn hóa (shape: [sequence_length, 1])
            label: tensor chứa nhãn (0: HOLD, 1: BUY, 2: SELL)
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Chỉ số {idx} nằm ngoài phạm vi [0, {len(self)-1}]")
            
        # Lấy chuỗi dữ liệu
        sequence = self.data.iloc[idx:idx + self.sequence_length]
        
        # Lấy features (Close, Volume)
        features = sequence[['Close', 'Volume']].values
        timestamps = sequence['timestamp_norm'].values.reshape(-1, 1)
        
        # Lấy nhãn của điểm cuối chuỗi
        label = sequence.iloc[-1]['action']
        
        # Chuyển đổi thành tensor
        features = torch.FloatTensor(features)
        timestamps = torch.FloatTensor(timestamps)
        label = torch.LongTensor([label])[0]
        
        if self.transform:
            features = self.transform(features)
        
        return features, timestamps, label

def get_data_loaders(train_path, dev_path, test_path, sequence_length, batch_size):
    """
    Tạo DataLoader cho train, dev và test
    
    Args:
        train_path: đường dẫn đến file train đã xử lý
        dev_path: đường dẫn đến file dev đã xử lý
        test_path: đường dẫn đến file test đã xử lý
        sequence_length: độ dài chuỗi thời gian
        batch_size: kích thước batch
    
    Returns:
        train_loader, dev_loader, test_loader
    """
    # Kiểm tra các file tồn tại
    for path in [train_path, dev_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {path}")
    
    # Tạo dataset
    train_dataset = XAUUSDDataset(train_path, sequence_length)
    dev_dataset = XAUUSDDataset(dev_path, sequence_length)
    test_dataset = XAUUSDDataset(test_path, sequence_length)
    
    # Tạo DataLoader với các tham số tối ưu
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Không shuffle để giữ thứ tự thời gian
        num_workers=4,  # Số worker để load dữ liệu
        pin_memory=True,  # Sử dụng pin_memory để tăng tốc độ transfer lên GPU
        drop_last=True  # Bỏ qua batch cuối nếu không đủ kích thước
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,  # Không shuffle cho validation và test
        num_workers=4,
        pin_memory=True,
        drop_last=False  # Giữ lại batch cuối cho validation và test
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, dev_loader, test_loader 