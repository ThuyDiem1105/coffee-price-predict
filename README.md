# BÁO CÁO KỸ THUẬT & TÀI LIỆU DỰ ÁN (PROJECT REPORT & README)
**Hệ thống Dự đoán Giá Cà phê Đắk Lắk & Hỗ trợ Ra quyết định (End-to-End Coffee Price Prediction & Decision Support System)**

## 1. TỔNG QUAN ĐỀ TÀI & CƠ SỞ KHOA HỌC
### 1.1. Bối cảnh và Đặt vấn đề
Ngành hàng cà phê tại khu vực **Tây Nguyên** nói chung và **Đắk Lắk** nói riêng thường xuyên đối mặt với điệp khúc *"được mùa mất giá"* hoặc biến động bất định về giá nguyên liệu. Nông dân và các hợp tác xã thường hoàn toàn bị động trước thông tin thị trường, thiếu các công cụ định lượng khoa học để trả lời cho câu hỏi cốt lõi: *"Khi nào nên bán ra?"* hay *"Khi nào nên tích trữ (trữ kho)?"*.
Để giải quyết bài toán minh bạch hóa thông tin thị trường và hỗ trợ người dân thoát khỏi sự chèn ép của thương lái, hệ thống này được xây dựng như một giải pháp **End-to-End** toàn diện: từ khâu thu thập/lưu trữ dữ liệu tự động, xây dựng mô hình Học sâu (Deep Learning) dự báo giá, kiểm định hiệu quả kinh tế (Backtesting), cho đến việc trực quan hóa trên giao diện Business Intelligence (Power BI).   
### 1.2. Mục tiêu bài toán
**Bài toán cốt lõi:** Dự báo chuỗi thời gian đa biến (Multivariate Time Series Forecasting) đối với giá cà phê Robusta tại Đắk Lắk trong khoảng thời gian **3 đến 7 ngày tới**.   
**Giá trị mang lại:** Cung cấp thông tin dự báo xu hướng (Tăng/Giảm) chính xác, kết hợp các kịch bản khuyến nghị giao dịch nhằm tối ưu hóa lợi nhuận cho người trồng cà phê và các doanh nghiệp thu mua.

## 2. KIẾN TRÚC HỆ THỐNG & CÔNG NGHỆ SỬ DỤNG
Hệ thống được thiết kế theo mô hình kiến trúc khép kín gồm 4 tầng chính:
**1. Data Storage Layer:** Cơ sở dữ liệu quan hệ **PostgreSQL** (`coffee_db`) quản lý kho đặc trưng lịch sử (`coffee_master_feature_store`) và lưu trữ kết quả dự báo (`coffee_price_forecasts`).
**2. Machine Learning / AI Pipeline:** Mô hình lai **CNN-LSTM** viết bằng **PyTorch**, kết hợp tiền xử lý dữ liệu với `Scikit-Learn` (`StandardScaler`).
**3. Automation Layer:** Script Python tự động (`predict_daily.py`) thực hiện suy luận định kỳ và đồng bộ dữ liệu vào Database, kết hợp Task Scheduler / Cron Job.
**4. Business Intelligence Layer:** Giao diện **Power BI Desktop** kết hợp hàm đo lường **DAX** nâng cao để trực quan hóa dữ liệu thời gian thực.

Tech Stack chi tiết:
* Ngôn ngữ lập trình: Python 3.14
* Deep Learning Framework: PyTorch (nn.Conv1d, nn.LSTM, nn.Sequential)
* Data Processing & Analytics: Pandas, NumPy, Scikit-Learn
* Database & ORM: PostgreSQL, SQLAlchemy, Psycopg2
* Visualization & BI: Power BI, DAX, Matplotlib, Seaborn
* Automation: Windows Task Scheduler / Linux Cron

## 3. CẤU TRÚC THƯ MỤC DỰ ÁN (PROJECT DIRECTORY STRUCTURE)
```text
D:\predict_coffee_price\
│
├── crawl_coffee.ipynb             # Notebook thực hiện cào và thu thập dữ liệu giá cà phê tự động
├── dashboard.pbix                 # Tệp báo cáo và giao diện trực quan hóa trên Power BI
├── eda.ipynb                      # Notebook phân tích dữ liệu khám phá (Exploratory Data Analysis)
├── evaluation.ipynb               # Notebook đánh giá chi tiết sai số và backtesting mô hình
├── feature_engineering.ipynb      # Notebook xử lý, tạo lập và trích xuất các đặc trưng (198 features)
├── post_training.ipynb            # Notebook phân tích sâu hậu huấn luyện và trực quan hóa residual
├── predict_daily.py               # Script Python tự động hóa pipeline suy luận và ghi nhận cơ sở dữ liệu
├── requirements.txt               # Danh sách các thư viện Python phụ thuộc của toàn dự án
├── training.py                    # Script huấn luyện mô hình cơ sở
├── training2.py                   # Script huấn luyện tối ưu hóa kiến trúc mạng
├── training3.py                   # Script huấn luyện hoàn thiện tích hợp kiểm định
└── README.md                      # Tài liệu hướng dẫn kỹ thuật và mô tả dự án toàn diện
```

## 4. CHI TIẾT CÁC MODULE VÀ THUẬT TOÁN
### 4.1. Module Mô hình Học sâu (CNN-LSTM Architecture)
Mô hình kết hợp ưu điểm của mạng tích chập (Convolutional Neural Networks) và mạng hồi tiếp (Long Short-Term Memory):
* **Conv1d Layer:** Đảm nhận nhiệm vụ quét qua 198 đặc trưng đầu vào (`num_features = 198`) trong cửa sổ thời gian lịch sử (`seq_length = 14 ngày`) để tự động trích xuất các mẫu không gian (spatial patterns) và biểu diễn cục bộ của giá.
* **LSTM Layers:** Xử lý chuỗi thời gian đã qua tích chập với 2 tầng ẩn (`hidden_size = 64`, `dropout = 0.2`) nhằm nắm bắt các phụ thuộc dài hạn và xu hướng biến động giá theo chu kỳ.
* **Fully Connected Layers (Sequential):**Ánh xạ không gian đặc trưng ẩn cuối cùng về giá trị dự báo đơn (1 chiều: giá cà phê Đắk Lắk tại thời điểm $t + \text{horizon}$).
### 4.2. Module Tự động hóa Pipeline (predict_daily.py)
Quy trình thực thi tự động hằng ngày diễn ra theo các bước:
**1. Kết nối & Truy vấn:** Kết nối an toàn đến cơ sở dữ liệu PostgreSQL, trích xuất toàn bộ bảng đặc trưng lịch sử (`coffee_master_feature_store`) sắp xếp theo thời gian (`"Date"`).
**2. Xử lý & Cắt lát dữ liệu:** Kiểm tra và đảm bảo ma trận đặc trưng khớp chính xác **198 cột** chuẩn hóa qua `StandardScaler`. Lấy cửa sổ 14 ngày gần nhất làm đầu vào mô hình.
**3. Suy luận (Inference):** Nạp trọng số từ file checkpoint `coffee_cnn_lstm_model.pth`, thực hiện tính toán trên thiết bị (CPU/GPU) để cho ra giá trị dự báo chuẩn hóa, sau đó nghịch đảo (`Inverse Transform`) về đơn vị tiền tệ gốc **(VND/kg)**.
**5. Ghi nhận kết quả:** Tự động tạo mảng dữ liệu dự báo cho 7 ngày tiếp theo (`horizon = 7`) kèm theo mốc thời gian và nhãn mô hình, sau đó append trực tiếp vào bảng `coffee_price_forecasts` trên PostgreSQL.

## 5. HƯỚNG DẪN CÀI ĐẶT VÀ VẬN HÀNH (GETTING STARTED)
**Bước 1: Cài đặt môi trường Python và thư viện**
Mở terminal tại thư mục dự án và chạy lệnh cài đặt các gói phụ thuộc:
```bash
Bashpip install torch torchvision torchaudio scikit-learn pandas numpy sqlalchemy psycopg2-binary matplotlib seaborn
```

**Bước 2: Cấu hình Cơ sở dữ liệu PostgreSQL** 
- Tạo database mới trong PostgreSQL với tên: `coffee_db`.
- Đảm bảo bảng dữ liệu lịch sử `coffee_master_feature_store` đã được khởi tạo đầy đủ 198 cột đặc trưng và cột thời gian `"Date"`.
- Kiểm tra thông tin kết nối chuỗi trong script (`postgresql://postgres:your_password@localhost:5432/coffee_db`).

**Bước 3: Chạy kiểm thử Pipeline Dự báo thủ công**
Thực thi trực tiếp file script tự động hóa trên terminal:
```bash
BashC:\Users\DELL\AppData\Local\Programs\Python\Python314\python.exe d:/predict_coffee_price/predict_daily.py
```
*Hệ thống sẽ báo cáo quá trình tải dữ liệu, thực hiện suy luận mô hình và thông báo lưu thành công 7 bản ghi dự báo mới vào bảng `coffee_price_forecasts`.*

**Bước 4: Thiết lập Lịch chạy tự động (Automation Task)**
* Trên Windows:
- Mở **Task Scheduler** $\rightarrow$ Chọn Create Task.
- Thiết lập thời gian chạy định kỳ hằng ngày (ví dụ: 6:00 sáng).
- Trỏ đường dẫn chương trình đến file `python.exe` và đối số truyền vào là đường dẫn tuyệt đối của `predict_daily.py`.
* Trên Linux (Cron Job):
- Thêm dòng lệnh sau vào crontab (`crontab -e`):
```Bash
0 6 * * * /usr/bin/python3 /path/to/predict_daily.py >> /path/to/logfile.log 2>&1
```

**Bước 5: Khởi chạy Power BI Dashboard**
- Mở tệp báo cáo `Coffee_Price_Dashboard.pbix` bằng ứng dụng Power BI Desktop.
- Kiểm tra nguồn kết nối dữ liệu đến PostgreSQL (`coffee_db`).
- Sử dụng các hàm đo lường DAX tùy chỉnh (như `Gia_Moi_Nhat = CALCULATE(MAX(...), ...)` và `Gia_Thuc_Te = AVERAGE(...)`) để hiển thị trực quan.
- Bấm Refresh để cập nhật toàn bộ luồng dữ liệu mới nhất từ cơ sở dữ liệu lên dashboard.
## 6. KẾT QUẢ VÀ HIỆU SUẤT ĐẠT ĐƯỢC (RESULTS & PERFORMANCE)
* **Hiệu suất mô hình:** Kiến trúc CNN-LSTM thể hiện khả năng vượt trội trong việc bắt trọn độ trễ và xu hướng phi tuyến tính của giá cà phê Đắk Lắk so với các mô hình thống kê truyền thống (như ARIMA) hay mạng LSTM thuần.
* **Kiểm định kinh tế (Backtesting):** Mô phỏng giao dịch thực tế dựa trên tín hiệu xu hướng của AI cho thấy tỷ suất sinh lợi nhuận tích lũy cao hơn rõ rệt so với phương pháp nắm giữ thông thường (`Buy & Hold`).
* **Tính ứng dụng:** Xây dựng thành công hệ thống khép kín từ khâu thu hút dữ liệu, chạy AI ngầm, đến hiển thị báo cáo trực quan, giải quyết triệt để bài toán thông tin bất cân xứng cho nông dân vùng cao.