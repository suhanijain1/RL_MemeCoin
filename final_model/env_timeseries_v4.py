import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any

class ScamTokenTimeseriesEnvV4(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(self,
                 parquet_file: str,
                 feature_cols: List[str],
                 train_mean_scaled: pd.Series,
                 train_std_scaled: pd.Series,
                 truncation_penalty: float = -0.5
                 ):
        super().__init__()

        try:
            self.data = pd.read_parquet(parquet_file)
            required_cols = feature_cols + ['token_id', 'time_step', 'label']
            missing_cols = [col for col in required_cols if col not in self.data.columns]
            if missing_cols: raise ValueError(f"Missing required columns: {missing_cols} in {parquet_file}")
            self.grouped_data = self.data.groupby('token_id')
            self.token_ids = list(self.grouped_data.groups.keys())
            if not self.token_ids: raise ValueError(f"No token IDs found in {parquet_file}")

            self.token_data_cache = {}
            self.token_label_cache = {}
            for token_id, group in self.grouped_data:
                 labels = group['label'].unique()
                 if len(labels) > 1:
                      print(f"Warning: Token {token_id} has inconsistent labels across timesteps. Using label from first timestep.")
                 self.token_label_cache[token_id] = int(labels[0])
                 self.token_data_cache[token_id] = group[feature_cols].reset_index(drop=True)

            missing_mean_keys = [col for col in feature_cols if col not in train_mean_scaled.index]
            if missing_mean_keys: raise ValueError(f"Features {missing_mean_keys} not found in train_mean_scaled index.")
            missing_std_keys = [col for col in feature_cols if col not in train_std_scaled.index]
            if missing_std_keys: raise ValueError(f"Features {missing_std_keys} not found in train_std_scaled index.")

        except FileNotFoundError:
            raise FileNotFoundError(f"Parquet file not found at {parquet_file}")
        except Exception as e:
            raise ValueError(f"Error loading/processing Parquet {parquet_file}: {e}")

        self.feature_cols = feature_cols
        self.num_features = len(feature_cols)
        self.train_mean = train_mean_scaled[self.feature_cols]
        self.train_std = train_std_scaled[self.feature_cols]
        self.truncation_penalty = truncation_penalty

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_features,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self.current_token_id = None
        self.current_token_data = None
        self.current_timestep_idx = 0
        self.max_timesteps_in_episode = 0
        self.true_token_label = None

    def _get_current_token_data(self):
        if self.current_token_id is None:
            raise ValueError("Environment not reset. Call reset() before accessing token data.")
        return self.token_data_cache[self.current_token_id]

    def _get_obs(self) -> np.ndarray:
        if self.current_token_data is None:
            raise ValueError("Environment not reset or token data is missing. Call reset().")

        if self.current_timestep_idx >= self.max_timesteps_in_episode or self.current_timestep_idx < 0:
             return np.zeros(self.observation_space.shape, dtype=np.float32)

        features = self.current_token_data.loc[self.current_timestep_idx, self.feature_cols].values.astype(np.float32)

        if np.any(~np.isfinite(features)):
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
        features = np.clip(features, 0.0, 1.0)

        return features

    def _get_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "token_id": self.current_token_id,
            "current_timestep_in_token": self.current_timestep_idx,
            "total_timesteps_in_token": self.max_timesteps_in_episode,
            "true_token_label": self.true_token_label
        }
        return info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        requested_token_id = None
        if options is not None and isinstance(options, dict):
            requested_token_id = options.get('token_id')

        if requested_token_id is not None and requested_token_id in self.token_label_cache:
            self.current_token_id = requested_token_id
        else:
            if requested_token_id is not None:
                print(f"Warning: Requested token_id '{requested_token_id}' not found in test set. Choosing random token instead.")
            self.current_token_id = self.np_random.choice(self.token_ids)

        self.current_token_data = self._get_current_token_data()
        self.true_token_label = self.token_label_cache[self.current_token_id]
        self.max_timesteps_in_episode = len(self.current_token_data)
        self.current_timestep_idx = 0

        observation = self._get_obs()
        info = self._get_info()
        info["start_token_id"] = self.current_token_id

        return observation, info


    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}. Action must be 0, 1, or 2.")
        if self.current_token_data is None or self.true_token_label is None:
            raise ValueError("Environment not reset. Call reset() before calling step().")

        reward = 0.0
        terminated = False
        truncated = False
        predicted_label = -1

        if action == 0:
            self.current_timestep_idx += 1
            if self.current_timestep_idx >= self.max_timesteps_in_episode:
                truncated = True
                reward = self.truncation_penalty
        elif action == 1:
            terminated = True
            predicted_label = 0
            reward = 1.0 if self.true_token_label == 0 else -1.0
        elif action == 2:
            terminated = True
            predicted_label = 1
            reward = 1.0 if self.true_token_label == 1 else -1.0

        observation = self._get_obs()
        info = self._get_info()

        info["action_taken"] = action
        info["reward_obtained"] = reward
        info["predicted_label"] = predicted_label
        info["steps_taken_in_episode"] = self.current_timestep_idx

        return observation, reward, terminated, truncated, info

    def render(self, mode: str = "human"):
        if self.current_token_id is None:
            print("Environment not reset.")
            return

        step_to_render = self.current_timestep_idx - 1 if self.current_timestep_idx > 0 else 0

        features_now = self.current_token_data.loc[step_to_render, self.feature_cols]
        features_str = " | ".join([f"{col}: {val:.3f}" for col, val in features_now.items()])

        output = (f"Token: {self.current_token_id} (True Label: {self.true_token_label}) | "
                   f"Step t={self.current_timestep_idx}/{self.max_timesteps_in_episode} | "
                   f"Feat(t={step_to_render}): {features_str}"
                   )

        if mode == "human":
            print(output)
        elif mode == "ansi":
            return output

    def close(self):
        self.data = None
        self.token_data_cache = {}
        self.token_label_cache = {}
        print("Environment V4 closed.")

