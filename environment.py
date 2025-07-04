import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self, df, window_size=10, initial_cash=100000,
                 buy_min=0.1, buy_max=0.5, sell_min=0.1, sell_max=0.5,
                 transaction_fee=0.01, penalty_usd=100, no_trade_limit=5,
                 stop_loss=5000, goal_asset=1000000, min_asset=10000):
        
        super(StockTradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_cash = initial_cash
        self.buy_min = buy_min
        self.buy_max = buy_max
        self.sell_min = sell_min
        self.sell_max = sell_max
        self.transaction_fee = transaction_fee
        self.penalty_usd = penalty_usd
        self.no_trade_limit = no_trade_limit
        self.stop_loss = stop_loss
        self.goal_asset = goal_asset
        self.min_asset = min_asset
        
        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation: window_size * num_features + [cash, shares]
        self.num_features = df.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(window_size, self.num_features + 2),
            dtype=np.float32
        )

        self.reset()
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = self.window_size
        self.cash = self.initial_cash
        self.shares_held = 0
        self.total_asset = self.cash
        self.no_trade_count = 0
        self.done = False

        obs = self._get_observation()
        return obs, {}  # gymnasium yêu cầu return (obs, info)


    def _get_observation(self):
        window_data = self.df.iloc[self.current_step - self.window_size:self.current_step].values
        obs = np.hstack([
            window_data,
            np.full((self.window_size, 1), self.cash),
            np.full((self.window_size, 1), self.shares_held)
        ])
        return obs.astype(np.float32)

    def _get_current_price(self):
        return self.df.iloc[self.current_step]["Close"]

    def step(self, action):
        terminated = False
        truncated = False
        info = {}

        current_price = self._get_current_price()
        prev_asset = self.cash + self.shares_held * current_price

        # === Hành động ===
        if action == 1:  # Buy
            amount_to_spend = np.random.uniform(self.buy_min, self.buy_max) * self.cash
            shares_bought = (amount_to_spend * (1 - self.transaction_fee)) // current_price
            cost = shares_bought * current_price * (1 + self.transaction_fee)

            if shares_bought > 0 and self.cash >= cost:
                self.cash -= cost
                self.shares_held += shares_bought
                self.no_trade_count = 0
            else:
                self.no_trade_count += 1

        elif action == 2:  # Sell
            shares_to_sell = int(np.random.uniform(self.sell_min, self.sell_max) * self.shares_held)
            if shares_to_sell > 0:
                proceeds = shares_to_sell * current_price * (1 - self.transaction_fee)
                self.cash += proceeds
                self.shares_held -= shares_to_sell
                self.no_trade_count = 0
            else:
                self.no_trade_count += 1

        else:  # Hold
            self.no_trade_count += 1

        # === Cập nhật tài sản ===
        self.total_asset = self.cash + self.shares_held * current_price
        delta_asset = self.total_asset - prev_asset
        

        # === Reward mới (có yếu tố dài hạn và ngắn hạn) ===
        reward = delta_asset / prev_asset  # phần trăm thay đổi (short-term)
        
        # Thưởng khi tài sản tăng liên tục
        if reward > 0.005:
            reward += 0.01
        elif reward < -0.005:
            reward -= 0.01

        # Phạt khi lỗ dài hạn
        if self.total_asset < self.initial_cash * 0.95:
            reward -= 0.02

        # === Penalty khi không giao dịch quá lâu ===
        if self.no_trade_count >= self.no_trade_limit:
            self.cash -= self.penalty_usd
            reward -= 0.005
            self.no_trade_count = 0

        # === Điều kiện kết thúc ===
        self.current_step += 1
        if (
            self.total_asset >= self.goal_asset or
            self.total_asset <= self.min_asset or
            self.cash <= -self.stop_loss
        ):
            terminated = True

        if self.current_step >= len(self.df):
            truncated = True

        obs = self._get_observation()
        return obs, reward, terminated, truncated, info

