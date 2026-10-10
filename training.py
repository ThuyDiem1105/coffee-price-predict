import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os

# ==========================================
# PHẦN 1: KẾT NỐI POSTGRESQL & TẢI FEATURE STORE
# ==========================================
def load_data_from_postgres():
    print("--- [Phần 1] Kết nối PostgreSQL để lấy dữ liệu Master Feature Store ---")
    
    DB_USER = "postgres"
    DB_PASSWORD = "0101"
    DB_HOST = "localhost"
    DB_PORT = "5432"
    DB_NAME = "coffee_db"
    
    DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(DATABASE_URL)
    
    # Truy vấn bảng master feature store đã được tổng hợp từ notebook của bạn
    query = "SELECT * FROM coffee_master_feature_store ORDER BY \"Date\" ASC;"
    df = pd.read_sql(query, engine)
    
    # Đảm bảo cột Date chuẩn datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
    print(f"[+] Tải thành công {len(df)} dòng và {len(df.columns)} cột từ bảng 'coffee_master_feature_store'.")
    return df


# ==========================================
# PHẦN 2: CHIA TẬP TRAIN / VAL / TEST (TIME-SERIES)
# ==========================================
def split_time_series_data(df, target_col, test_size=0.2, val_size=0.1):
    print("\n--- [Phần 2] Chia tập Train / Val / Test theo thời gian ---")
    
    # Loại bỏ các dòng bị khuyết giá trị ở cột Target (do shift tương lai)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    
    n_total = len(df)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)
    n_train = n_total - n_test - n_val
    
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]
    
    print(f"Tổng số mẫu hợp lệ: {n_total}")
    print(f" - Tập Train: {len(train_df)} mẫu (từ {train_df['Date'].min().strftime('%Y-%m-%d')} đến {train_df['Date'].max().strftime('%Y-%m-%d')})")
    print(f" - Tập Val:   {len(val_df)} mẫu (từ {val_df['Date'].min().strftime('%Y-%m-%d')} đến {val_df['Date'].max().strftime('%Y-%m-%d')})")
    print(f" - Tập Test:  {len(test_df)} mẫu (từ {test_df['Date'].min().strftime('%Y-%m-%d')} đến {test_df['Date'].max().strftime('%Y-%m-%d')})")
    
    # Các cột cần loại bỏ khỏi tập đặc trưng (Features X)
    drop_cols = ['Date', target_col]
    
    # Loại bỏ cả các cột Target khác (ví dụ nếu train 7 ngày thì bỏ target 30 ngày để tránh rò rỉ)
    other_targets = [c for c in df.columns if 'Target_' in c and c != target_col]
    drop_cols.extend(other_targets)
    
    X_train = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    y_train = train_df[target_col]
    
    X_val = val_df.drop(columns=[c for c in drop_cols if c in val_df.columns])
    y_val = val_df[target_col]
    
    X_test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])
    y_test = test_df[target_col]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# ==========================================
# PHẦN 3: HUẤN LUYỆN MÔ HÌNH (LIGHTGBM)
# ==========================================
def train_coffee_model(X_train, y_train, X_val, y_val):
    print("\n--- [Phần 3] Huấn luyện mô hình LightGBM Regressor ---")
    
    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    # Huấn luyện mô hình kết hợp Early Stopping trên tập Validation
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=50)
        ]
    )
    
    print("Huấn luyện mô hình hoàn tất!")
    return model


if __name__ == "__main__":
    try:
        # 1. Tải dữ liệu từ PostgreSQL
        df = load_data_from_postgres()
        
        # 2. Định nghĩa mục tiêu dự đoán (Ví dụ: Dự đoán giá Đắk Lắk 7 ngày tới)
        target_column = "Target_Price_7d_Ahead"
        
        # 3. Chia tập dữ liệu theo thứ tự thời gian
        X_train, y_train, X_val, y_val, X_test, y_test = split_time_series_data(df, target_col=target_column)
        
        # 4. Huấn luyện mô hình
        model = train_coffee_model(X_train, y_train, X_val, y_val)
        
        # 5. Đánh giá sơ bộ trên tập Test
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print("\n---------------------------------------------------------------------")
        print("                  ĐÁNH GIÁ HIỆU SUẤT TRÊN TẬP TEST")
        print("---------------------------------------------------------------------")
        print(f" - RMSE (Root Mean Squared Error): {rmse:.2f}")
        print(f" - MAE  (Mean Absolute Error):     {mae:.2f}")
        print("---------------------------------------------------------------------")
        
        # Lưu mô hình đã train
        model_filename = "coffee_price_lgbm_model.pkl"
        joblib.dump(model, model_filename)
        print(f"Đã lưu mô hình thành công vào file: '{model_filename}'")
        
    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình thực thi: {e}")