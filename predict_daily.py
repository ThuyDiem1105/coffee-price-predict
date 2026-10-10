import os
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine

# ==========================================
# 1. CẤU HÌNH THÔNG SỐ & KẾT NỐI DATABASE
# ==========================================
DB_CONNECTION_STRING = 'postgresql+psycopg2://postgres:0101@localhost:5432/coffee_db'
MODEL_PATH = 'coffee_cnn_lstm_model.pth'

SEQ_LENGTH = 14
HORIZON = 7
TARGET_COL = 'Price_DakLak'
EXPECTED_NUM_FEATURES = 198  # Số lượng đặc trưng chuẩn khớp với checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. ĐỊNH NGHĨA KIẾN TRÚC MÔ HÌNH CNN-LSTM (Khớp Checkpoint)
# ==========================================
class CNN_LSTM_CoffeeModel(nn.Module):
    def __init__(self, num_features):
        super(CNN_LSTM_CoffeeModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Features] -> Permute sang [Batch, Features, Seq_Len] cho Conv1d
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.permute(0, 2, 1)  # Trở lại [Batch, Seq_Len/2, Channels] cho LSTM
        x, _ = self.lstm(x)
        x = x[:, -1, :]         # Lấy đầu ra ở bước thời gian cuối cùng
        x = self.fc(x)
        return x

def run_daily_forecast():
    print(f"[{datetime.datetime.now()}] Bắt đầu chạy pipeline dự báo giá cà phê tự động...")

    try:
        # Bước 1: Kết nối và đọc dữ liệu từ PostgreSQL
        engine = create_engine(DB_CONNECTION_STRING)
        query = 'SELECT * FROM coffee_master_feature_store ORDER BY "Date" ASC'
        df = pd.read_sql(query, con=engine)
        
        if len(df) < SEQ_LENGTH:
            print("Lỗi: Dữ liệu trong cơ sở dữ liệu không đủ để tạo chuỗi thời gian.")
            return

        print(f"Đã tải thành công {len(df)} bản ghi từ cơ sở dữ liệu.")

        # Bước 2: Lọc đặc trưng và đảm bảo khớp chính xác 198 features như lúc train
        feature_cols = [col for col in df.columns if col.lower() != 'date']
        feature_data = df[feature_cols].values
        
        if feature_data.shape[1] > EXPECTED_NUM_FEATURES:
            print(f"Cảnh báo: Số lượng cột hiện tại ({feature_data.shape[1]}) nhiều hơn lúc train. Đang tự động cắt lấy {EXPECTED_NUM_FEATURES} đặc trưng đầu tiên.")
            feature_data = feature_data[:, :EXPECTED_NUM_FEATURES]
        elif feature_data.shape[1] < EXPECTED_NUM_FEATURES:
            raise ValueError(f"Lỗi: Số lượng đặc trưng hiện tại ({feature_data.shape[1]}) ít hơn yêu cầu của mô hình ({EXPECTED_NUM_FEATURES}).")

        # Chuẩn hóa dữ liệu
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(feature_data)

        # Lấy cửa sổ dữ liệu gần nhất (SEQ_LENGTH ngày cuối cùng)
        recent_window = scaled_data[-SEQ_LENGTH:]
        X_input = np.array([recent_window])  # Shape: (1, 14, 198)

        # Bước 3: Nạp mô hình và trọng số từ file .pth
        model = CNN_LSTM_CoffeeModel(num_features=EXPECTED_NUM_FEATURES).to(device)
        
        if not os.path.exists(MODEL_PATH):
            print(f"Lỗi: Không tìm thấy file trọng số mô hình tại '{MODEL_PATH}'.")
            return
            
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()

        # Thực hiện suy luận (Inference)
        with torch.no_grad():
            X_tensor = torch.tensor(X_input, dtype=torch.float32).to(device)
            pred_scaled = model(X_tensor).cpu().numpy()

        # Bước 4: Inverse Transform đưa về giá trị gốc (VND/kg)
        target_idx = 0  # Giả định cột Price_DakLak nằm ở vị trí index 0
        
        def inverse_single_value(scaler, pred_val, col_idx):
            dummy = np.zeros((1, scaler.mean_.shape[0]))
            dummy[0, col_idx] = pred_val
            return scaler.inverse_transform(dummy)[0, col_idx]

        predicted_price_value = inverse_single_value(scaler, pred_scaled[0, 0], target_idx)
        
        # Bước 5: Tạo bản ghi dự báo mở rộng cho 7 ngày tới (Horizon = 7)
        base_date = pd.to_datetime(df['Date'].iloc[-1])
        forecast_records = []
        
        current_pred = predicted_price_value
        for i in range(1, HORIZON + 1):
            next_date = base_date + datetime.timedelta(days=i)
            # Có thể tinh chỉnh biến động nhẹ theo xu hướng dự báo
            forecast_records.append({
                'forecast_date': next_date.date(),
                'predicted_price': float(current_pred),
                'model_name': 'CNN-LSTM',
                'created_at': datetime.datetime.now()
            })

        forecast_df = pd.DataFrame(forecast_records)

        # Bước 6: Lưu kết quả dự báo vào bảng 'coffee_price_forecasts' trên PostgreSQL
        forecast_df.to_sql('coffee_price_forecasts', con=engine, if_exists='append', index=False)
        print(f"[{datetime.datetime.now()}] Đã lưu thành công {HORIZON} bản ghi dự báo mới vào bảng 'coffee_price_forecasts'!")

    except Exception as e:
        print(f"[{datetime.datetime.now()}] Đã xảy ra lỗi trong quá trình chạy pipeline: {e}")

if __name__ == '__main__':
    run_daily_forecast()