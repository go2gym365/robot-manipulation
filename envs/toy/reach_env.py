import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ReachEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # observation = [ee_x, ee_y, target_x, target_y] -> 현재위치 2개, 목표위치 2개
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(4,), dtype=np.float32
        )

        # action = [dx, dy]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        self.max_steps = 100
        self.step_count = 0

        self.ee_pos = None
        self.target_pos = None

    def _get_obs(self):
        return np.concatenate([self.ee_pos, self.target_pos]).astype(np.float32)

    def _get_info(self):
        dist = np.linalg.norm(self.ee_pos - self.target_pos)
        return {
            "distance": float(dist),
            "is_success": bool(dist < 0.05),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0
        self.ee_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.target_pos = self.np_random.uniform(
            low=-0.8, high=0.8, size=(2,)
        ).astype(np.float32)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        self.step_count += 1

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.ee_pos = self.ee_pos + 0.2 * action

        dist = np.linalg.norm(self.ee_pos - self.target_pos)

        # 기본 reward: 가까워질수록 덜 음수
        reward = float(-dist) # 리워드를 float로 기대함

        terminated = bool(dist < 0.05)
        truncated = self.step_count >= self.max_steps

        if terminated:
            reward += 10.0

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def render(self):
        print(f"EE: {self.ee_pos}, Target: {self.target_pos}")