import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

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
    
    query = "SELECT * FROM coffee_master_feature_store ORDER BY \"Date\" ASC;"
    df = pd.read_sql(query, engine)
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
    print(f"[+] Tải thành công {len(df)} dòng và {len(df.columns)} cột từ bảng 'coffee_master_feature_store'.")
    return df


# ==========================================
# PHẦN 2: TIỀN XỬ LÝ & CHIA TẬP TIME-SERIES
# ==========================================
def prepare_and_split_data(df, horizon=7, test_size=0.2, val_size=0.1):
    print(f"\n--- [Phần 2] Xử lý biến mục tiêu (Dự báo chênh lệch giá {horizon} ngày tới) ---")
    
    target_abs_col = f"Target_Price_{horizon}d_Ahead"
    base_price_col = "Price_DakLak"
    
    # Loại bỏ dòng thiếu dữ liệu target
    df = df.dropna(subset=[target_abs_col, base_price_col]).reset_index(drop=True)
    
    # CHIẾN LƯỢC CẢI TIẾN: Dự báo tỷ lệ thay đổi (Return/PctChange) thay vì giá tuyệt đối
    # Target mới = (Giá tương lai - Giá hiện tại) / Giá hiện tại
    target_return_col = f"Target_Return_{horizon}d"
    df[target_return_col] = (df[target_abs_col] - df[base_price_col]) / df[base_price_col]
    
    # Chia tập theo thời gian
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
    
    # Các cột loại bỏ khỏi đặc trưng (Features X)
    drop_cols = ['Date', base_price_col, target_abs_col, target_return_col]
    other_targets = [c for c in df.columns if 'Target_' in c and c != target_return_col]
    drop_cols.extend(other_targets)
    
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X_train, y_train = train_df[feature_cols], train_df[target_return_col]
    X_val, y_val = val_df[feature_cols], val_df[target_return_col]
    X_test, y_test_return = test_df[feature_cols], test_df[target_return_col]
    
    # Lưu lại giá cơ sở và giá thực tế của tập test để đánh giá ngược lại giá VND
    test_base_prices = test_df[base_price_col].values
    test_actual_prices = test_df[target_abs_col].values
    
    return X_train, y_train, X_val, y_val, X_test, y_test_return, test_base_prices, test_actual_prices, feature_cols


# ==========================================
# PHẦN 3: HUẤN LUYỆN & LỌC ĐẶC TRƯNG QUAN TRỌNG
# ==========================================
def train_and_optimize_model(X_train, y_train, X_val, y_val, feature_cols):
    print("\n--- [Phần 3] Huấn luyện mô hình LightGBM tối ưu ---")
    
    model = lgb.LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.015,
        max_depth=5,
        num_leaves=25,
        min_child_samples=30,  # Tránh overfitting cho dữ liệu nhỏ
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.1,         # L1 Regularization
        reg_lambda=0.5,        # L2 Regularization
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=60),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # Lọc các đặc trưng có độ quan trọng > 0
    importance = model.feature_importances_
    feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': importance})
    active_features = feat_imp[feat_imp['importance'] > 0]['feature'].tolist()
    print(f"\n[+] Số lượng đặc trưng được giữ lại sau khi loại bỏ nhiễu: {len(active_features)} / {len(feature_cols)}")
    
    return model, active_features


if __name__ == "__main__":
    try:
        # 1. Tải dữ liệu từ PostgreSQL
        df = load_data_from_postgres()
        
        # 2. Tiền xử lý & chuyển đổi target sang dạng Return (%) cho horizon 7 ngày tới
        horizon_days = 7
        X_train, y_train, X_val, y_val, X_test, y_test_return, test_base_prices, test_actual_prices, feature_cols = prepare_and_split_data(
            df, horizon=horizon_days, test_size=0.2, val_size=0.1
        )
        
        # 3. Huấn luyện mô hình lần 1
        model, active_features = train_and_optimize_model(X_train, y_train, X_val, y_val, feature_cols)
        
        # Huấn luyện lại trên tập đặc trưng đã được lọc sạch nhiễu
        if len(active_features) < len(feature_cols):
            print("\n--- Huấn luyện lại tinh chỉnh trên các đặc trưng đã lọc ---")
            model, _ = train_and_optimize_model(
                X_train[active_features], y_train, 
                X_val[active_features], y_val, 
                active_features
            )
            X_test_final = X_test[active_features]
        else:
            X_test_final = X_test
            
        # 4. Dự báo và quy đổi ngược lại về giá tuyệt đối (VND) để đánh giá thực tế
        pred_returns = model.predict(X_test_final)
        pred_prices = test_base_prices * (1 + pred_returns)
        
        rmse = np.sqrt(mean_squared_error(test_actual_prices, pred_prices))
        mae = mean_absolute_error(test_actual_prices, pred_prices)
        
        print("\n---------------------------------------------------------------------")
        print(f"      ĐÁNH GIÁ HIỆU SUẤT SAU KHI TỐI ƯU (Dự báo {horizon_days} ngày tới)")
        print("---------------------------------------------------------------------")
        print(f" - RMSE (Root Mean Squared Error): {rmse:.2f} VND")
        print(f" - MAE  (Mean Absolute Error):     {mae:.2f} VND")
        print("---------------------------------------------------------------------")
        
        # Lưu mô hình và danh sách đặc trưng đã chọn
        joblib.dump((model, active_features), "coffee_optimized_model.pkl")
        print("Đã lưu mô hình tối ưu thành công vào file: 'coffee_optimized_model.pkl'")
        
    except Exception as e:
        print(f"Đã xảy ra lỗi trong quá trình thực thi: {e}")