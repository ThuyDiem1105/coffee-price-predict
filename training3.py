import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Thiết lập thiết bị chạy (Ưu tiên GPU nếu có, không thì dùng CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Sử dụng thiết bị tính toán: {device}")

# ==========================================
# PHẦN 1: TẢI DỮ LIỆU TỪ POSTGRESQL
# ==========================================
def load_data_from_postgres():
    print("--- [Phần 1] Kết nối PostgreSQL để tải Feature Store ---")
    DB_USER = "postgres"
    DB_PASSWORD = "0101"
    DB_HOST = "localhost"
    DB_PORT = "5432"
    DB_NAME = "coffee_db"
    
    DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(DATABASE_URL)
    
    query = "SELECT * FROM coffee_master_feature_store ORDER BY \"Date\" ASC;"
    df = pd.read_sql(query, engine)
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
    print(f"[+] Tải thành công {len(df)} dòng và {len(df.columns)} cột.")
    return df


# ==========================================
# PHẦN 2: CHUẨN BỊ WINDOWED DATA CHO DEEP LEARNING
# ==========================================
class CoffeeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_sequence_data(df, horizon=7, seq_length=15, test_size=0.2, val_size=0.1):
    print(f"\n--- [Phần 2] Xử lý chuỗi thời gian (Seq Length: {seq_length} ngày, Horizon: {horizon} ngày) ---")
    
    target_abs_col = f"Target_Price_{horizon}d_Ahead"
    base_price_col = "Price_DakLak"
    
    df = df.dropna(subset=[target_abs_col, base_price_col]).reset_index(drop=True)
    
    # Biến mục tiêu Log Return để chuẩn hóa miền giá trị
    target_return_col = f"Target_Return_{horizon}d"
    df[target_return_col] = np.log(df[target_abs_col] / df[base_price_col])
    
    drop_cols = ['Date', base_price_col, target_abs_col, target_return_col]
    other_targets = [c for c in df.columns if 'Target_' in c and c != target_return_col]
    drop_cols.extend(other_targets)
    
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    # Chuẩn hóa dữ liệu đặc trưng về khoảng [0, 1] cho mạng nơ-ron
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(df[feature_cols])
    target_values = df[target_return_col].values
    base_prices = df[base_price_col].values
    actual_prices = df[target_abs_col].values
    dates = df['Date'].values
    
    # Tạo cửa sổ thời gian (Sliding Window sequences)
    X, y, b_prices, a_prices, d_list = [], [], [], [], []
    for i in range(seq_length, len(df)):
        X.append(scaled_features[i-seq_length:i])
        y.append(target_values[i])
        b_prices.append(base_prices[i])
        a_prices.append(actual_prices[i])
        d_list.append(dates[i])
        
    X = np.array(X)
    y = np.array(y)
    b_prices = np.array(b_prices)
    a_prices = np.array(a_prices)
    
    # Chia tập theo thứ tự thời gian (Train / Val / Test)
    n_total = len(X)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)
    n_train = n_total - n_test - n_val
    
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
    
    test_base_prices = b_prices[n_train+n_val:]
    test_actual_prices = a_prices[n_train+n_val:]
    
    print(f" - Train samples: {len(X_train)} | Val samples: {len(X_val)} | Test samples: {len(X_test)}")
    
    return (X_train, y_train, X_val, y_val, X_test, y_test, 
            test_base_prices, test_actual_prices, scaler, len(feature_cols))


# ==========================================
# PHẦN 3: XÂY DỰNG KIẾN TRÚC HYBRID (CNN + LSTM)
# ==========================================
class CNN_LSTM_CoffeeModel(nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_layers=2, dropout=0.2):
        super(CNN_LSTM_CoffeeModel, self).__init__()
        
        # 1D-CNN trích xuất đặc trưng cục bộ từ không gian cột
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # LSTM nắm bắt chuỗi thời gian
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Fully connected layer dự báo giá trị Return cuối cùng
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, num_features)
        # Đổi chiều để phù hợp với Conv1d: (batch_size, num_features, seq_length)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Đổi lại chiều cho LSTM: (batch_size, seq_length, channels)
        x = x.permute(0, 2, 1)
        out, (hn, cn) = self.lstm(x)
        
        # Lấy hidden state ở bước thời gian cuối cùng
        out = self.fc(out[:, -1, :])
        return out.squeeze(-1)


# ==========================================
# PHẦN 4: HUẤN LUYỆN MÔ HÌNH DEEP LEARNING
# ==========================================
if __name__ == "__main__":
    try:
        df = load_data_from_postgres()
        
        horizon_days = 7
        seq_length = 14 # Nhìn lại 14 ngày trước để dự báo
        
        (X_train, y_train, X_val, y_val, X_test, y_test, 
         test_base_prices, test_actual_prices, scaler, num_features) = prepare_sequence_data(
            df, horizon=horizon_days, seq_length=seq_length, test_size=0.2, val_size=0.1
        )
        
        train_dataset = CoffeeDataset(X_train, y_train)
        val_dataset = CoffeeDataset(X_val, y_val)
        test_dataset = CoffeeDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False) # Không shuffle chuỗi thời gian
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Khởi tạo mô hình
        model = CNN_LSTM_CoffeeModel(num_features=num_features, hidden_dim=64, num_layers=2, dropout=0.2).to(device)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        print("\n--- [Phần 4] Bắt đầu huấn luyện mô hình CNN-LSTM ---")
        epochs = 100
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_path = "coffee_cnn_lstm_model.pth"
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_X.size(0)
                
            train_loss /= len(train_loader.dataset)
            
            # Đánh giá trên tập Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    preds = model(batch_X)
                    loss = criterion(preds, batch_y)
                    val_loss += loss.item() * batch_X.size(0)
                    
            val_loss /= len(val_loader.dataset)
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
                
            # Early Stopping & Lưu mô hình tốt nhất
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    print(f"[*] Early stopping kích hoạt tại epoch {epoch+1}!")
                    break
                    
        # Tải lại mô hình tốt nhất để đánh giá trên tập Test
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        
        preds_list = []
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(device)
                preds = model(batch_X)
                preds_list.extend(preds.cpu().numpy())
                
        pred_returns = np.array(preds_list)
        # Suy luận ngược từ Log Return sang giá tuyệt đối VND
        pred_prices = test_base_prices * np.exp(pred_returns)
        
        rmse = np.sqrt(mean_squared_error(test_actual_prices, pred_prices))
        mae = mean_absolute_error(test_actual_prices, pred_prices)
        
        print("\n---------------------------------------------------------------------")
        print(f"      ĐÁNH GIÁ HIỆU SUẤT MÔ HÌNH CNN-LSTM (Dự báo {horizon_days} ngày tới)")
        print("---------------------------------------------------------------------")
        print(f" - RMSE (Root Mean Squared Error): {rmse:.2f} VND")
        print(f" - MAE  (Mean Absolute Error):     {mae:.2f} VND")
        print("---------------------------------------------------------------------")
        
        # Lưu scaler để phục vụ dự báo thực tế sau này
        joblib.dump(scaler, "coffee_scaler.pkl")
        print("Đã lưu mô hình CNN-LSTM và Scaler thành công!")
        
    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình thực thi: {e}")