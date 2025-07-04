import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from environment import StockTradingEnv  # Đảm bảo đã cập nhật hàm step()

# === Load dữ liệu TEST ===
df = pd.read_csv("data/test_TMUS.csv", index_col=0)

# === Các feature cần thiết ===
features = ["Open", "High", "Low", "Close", "Volume", "sma_10", "rsi_14", "macd"]
df = df[features]

# === Tạo môi trường test ===
window_size = 30
env = DummyVecEnv([lambda: StockTradingEnv(df, window_size=window_size)])

# === Load mô hình PPO đã huấn luyện ===
model = PPO.load("models/ppo_tmustrader")

# === Reset môi trường ===
obs = env.reset()
done = False
total_reward = 0
step = 0
assets = []

# === Đánh giá ===
while True:
    action, _states = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    
    total_reward += reward[0]
    total_asset = env.get_attr("total_asset")[0]

    if done:
        break  # Dừng lại trước khi bước reset bị log

    step += 1
    print(f"Step {step}: Action={action[0]}, Reward={reward[0]:.2f}, Total Asset={total_asset:.2f} USD")
    assets.append(total_asset)

# === Kết quả ===
initial_cash = 100_000

# Nếu bị reset ở bước cuối thì loại bỏ bước đó
if assets[-1] == initial_cash:
    assets = assets[:-1]

final_asset = assets[-1]
profit = final_asset - initial_cash

print(f"✅ Đánh giá mô hình PPO trên tập test:")
print(f"• Số bước: {len(assets)}")
print(f"• Tổng Reward : {total_reward:.2f}")
print(f"• Tổng tài sản cuối cùng: {final_asset:,.2f} USD")
print(f"• Lợi nhuận thu được: {profit:,.2f} USD")

# === Vẽ biểu đồ hiệu suất ===
plt.figure(figsize=(12, 6))
plt.plot(assets, label="Total Asset (PPO)", color='green')

# Benchmark Buy & Hold
initial_price = df["Close"].iloc[0 + window_size]  # do window_size đã dùng để lấy quan sát đầu tiên
final_price = df["Close"].iloc[-1]
buy_hold_asset = initial_cash * final_price / initial_price
plt.axhline(buy_hold_asset, color='red', linestyle='--', label="Buy & Hold Benchmark")

plt.title("📈 Hiệu suất tài sản trên dữ liệu TEST")
plt.xlabel("Step")
plt.ylabel("Total Asset (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
