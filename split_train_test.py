import pandas as pd

# Load dữ liệu đã được chuẩn hoá và lưu
df = pd.read_csv("data/TMUS_history.csv", index_col=0, parse_dates=True)

# Kiểm tra ngày đầu & cuối
print("Ngày bắt đầu:", df.index.min())
print("Ngày kết thúc:", df.index.max())

# Tách 8 năm đầu để huấn luyện
df_train = df[df.index < "2023-01-01"]

# Tách 2 năm cuối để test
df_test = df[df.index >= "2023-01-01"]

# Lưu ra file riêng
df_train.to_csv("data/train_TMUS.csv")
df_test.to_csv("data/test_TMUS.csv")

print(f"✅ Tập train: {df_train.shape[0]} dòng | test: {df_test.shape[0]} dòng")
