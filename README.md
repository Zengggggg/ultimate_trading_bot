# 📊 PPO Stock Trading Agent — TMUS
## Mô tả dự án
Dự án này xây dựng một agent sử dụng thuật toán Proximal Policy Optimization (PPO) để giao dịch cổ phiếu TMUS dựa trên dữ liệu thị trường thực tế. Mục tiêu là tối đa hoá tài sản của agent thông qua học tăng cường (Reinforcement Learning).
# 🧠 Ý tưởng chính
- Sử dụng các chỉ báo kỹ thuật như: SMA (10), RSI (14), MACD để làm đặc trưng.

- Môi trường giao dịch tuỳ biến theo chuẩn OpenAI Gym.

- Hàm reward được thiết kế kết hợp phần thưởng ngắn hạn, dài hạn và hình phạt khi không giao dịch hoặc lỗ kéo dài.

⚙️ Hướng dẫn chạy
<i>1. Cài đặt thư viện</i>
