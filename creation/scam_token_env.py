import gym
from gym import spaces
import numpy as np
import pandas as pd

class ScamTokenEnv(gym.Env):
    def __init__(self, parquet_file):
        super(ScamTokenEnv, self).__init__()

        # Load dataset
        self.data = pd.read_parquet(parquet_file).reset_index(drop=True)

        # Extract features (exclude label)
        self.features = self.data.drop(columns=['label']).values
        self.labels = self.data['label'].values  # 1=scam, 0=legit

        # Observation space: based on feature dimensions
        self.observation_space = spaces.Box(
        low=np.min(self.features, axis=0).astype(np.float32),
        high=np.max(self.features, axis=0).astype(np.float32),
        dtype=np.float32
    )


        # Action space: 0 = Wait, 1 = Flag, 2 = Classify Legit, 3 = Classify Scam
        self.action_space = spaces.Discrete(4)

        # Environment state
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self._get_observation()

    def step(self, action):
        obs = self._get_observation()
        true_label = self.labels[self.current_step]

        reward = 0
        done = False

        # Reward logic
        if action == 0:  # Wait
            reward = -0.01  # Small penalty for delay
        elif action == 1:  # Flag
            reward = -0.005  # Slightly lower penalty
        elif action == 2:  # Classify Legit
            reward = 1 if true_label == 0 else -1
            done = True
        elif action == 3:  # Classify Scam
            reward = 1 if true_label == 1 else -1
            done = True

        # Move to next sample
        self.current_step += 1
        if self.current_step >= len(self.data):
            done = True

        next_obs = self._get_observation() if not done else np.zeros(self.observation_space.shape)
        return next_obs, reward, done, {}

    def _get_observation(self):
        return self.features[self.current_step].astype(np.float32)

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Observation: {self._get_observation()}")

