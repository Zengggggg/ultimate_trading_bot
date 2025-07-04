import yfinance as yf
import pandas as pd
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from sklearn.preprocessing import MinMaxScaler

# Cấu hình
TICKER = "TMUS"
START_DATE = "2015-01-01"
END_DATE = "2024-12-31"
SAVE_PATH = f"data/{TICKER}_history.csv"

df = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=False)

# Nếu cột là MultiIndex (2 cấp), xoá cấp Ticker đi
if isinstance(df.columns, pd.MultiIndex):
    df = df.droplevel(1, axis=1)  # giữ lại 'Close', 'Open', ...

# Xử lý nếu chỉ có 'Adj Close' mà không có 'Close'
if "Close" not in df.columns and "Adj Close" in df.columns:
    df["Close"] = df["Adj Close"]

# Kiểm tra tồn tại
if "Close" not in df.columns:
    raise ValueError("❌ Không tìm thấy cột 'Close'!")

df.dropna(subset=["Close"], inplace=True)

# Bỏ các dòng NaN
df = df.dropna(subset=["Close"])
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

# Tính chỉ báo kỹ thuật
close = df["Close"].squeeze()
df["sma_10"] = SMAIndicator(close=close, window=10).sma_indicator()
df["rsi_14"] = RSIIndicator(close=close, window=14).rsi()
df["macd"] = MACD(close=close).macd()

# Loại bỏ các dòng chưa đủ chỉ báo
df.dropna(inplace=True)

# Chọn cột cần dùng cho huấn luyện
features = ["Open", "High", "Low", "Close", "Volume", "sma_10", "rsi_14", "macd"]

# Chuẩn hoá dữ liệu
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Lưu file SẠCH
df[features].to_csv(SAVE_PATH, index=True)  # index=True để giữ ngày

print(f"✅ Dữ liệu sạch đã lưu tại: {SAVE_PATH}")
