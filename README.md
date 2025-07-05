# 📊 PPO Stock Trading Agent — TMUS
## Mô tả dự án
Dự án này xây dựng một agent sử dụng thuật toán Proximal Policy Optimization (PPO) để giao dịch cổ phiếu TMUS dựa trên dữ liệu thị trường thực tế. Mục tiêu là tối đa hoá tài sản của agent thông qua học tăng cường (Reinforcement Learning).
# 🧠 Ý tưởng chính
- Sử dụng các chỉ báo kỹ thuật như: SMA (10), RSI (14), MACD để làm đặc trưng.

- Môi trường giao dịch tuỳ biến theo chuẩn OpenAI Gym.

- Hàm reward được thiết kế kết hợp phần thưởng ngắn hạn, dài hạn và hình phạt khi không giao dịch hoặc lỗ kéo dài.

# ⚙️ Hướng dẫn chạy
1. Cài đặt thư viện
```python
pip install pandas
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install gym==0.26.2
pip install torch>=1.13.0
pip install stable-baselines3==1.7.0
pip install tensorboard
```
2. Tiền xử lý dữ liệu
```python
python download_preprocess.py
python split_train_test.py
```
3. Huấn luyện mô hình
```python
python train_agent.py
```
4. Đánh giá mô hình
```python
python evaluate_agent.py
```
# 📊 Kết quả
Agent được huấn luyện trên tập train 8 năm, đánh giá trên 2 năm.

Tài sản cuối cùng, lợi nhuận và so sánh với chiến lược Buy & Hold được ghi nhận trong evaluate_agent.py.


