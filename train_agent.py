import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from sklearn.preprocessing import MinMaxScaler
from environment import StockTradingEnv

# === Load dữ liệu TRAIN ===
df = pd.read_csv("data/train_TMUS.csv", index_col=0)

features = ["Open", "High", "Low", "Close", "Volume", "sma_10", "rsi_14", "macd"]

# === Tiền xử lý ===
if df[features].max().max() > 1:
    print("⚠️ Dữ liệu chưa chuẩn hoá → tiến hành chuẩn hoá...")
    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

df = df[features]  # Giữ lại đúng các cột

# === Cấu hình môi trường ===
window_size = 30
train_env = DummyVecEnv([lambda: Monitor(StockTradingEnv(df, window_size=window_size))])

# === Callback lưu mô hình tốt nhất ===
eval_env = DummyVecEnv([lambda: Monitor(StockTradingEnv(df, window_size=window_size))])
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/best_ppo_tmustrader",
    log_path="./logs/",
    eval_freq=5000,
    deterministic=True,
    render=False,
    verbose=1,
)

# === Tạo và train mô hình PPO ===
model = PPO(
    "MlpPolicy",
    train_env,
    verbose=1,
    tensorboard_log="./ppo_trading_log",
    learning_rate=5e-5,         # Nhỏ hơn mặc định
    gamma=0.99,                 # Discount nhẹ hơn
    gae_lambda=0.92,            # Thử giảm nhẹ
    batch_size=256,             # Đủ lớn để ổn định
    n_steps=2048                # Tăng n_steps nếu RAM đủ
)

# === Huấn luyện ===
model.learn(total_timesteps=500_000, callback=eval_callback)

# === Lưu mô hình ===
model.save("models/ppo_tmustrader")
print("✅ Mô hình đã được lưu tại: models/ppo_tmustrader.zip")
