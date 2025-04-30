import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict, Any

class ScamTokenTimeseriesEnvV2(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(self,
                 parquet_file: str,
                 feature_cols: List[str],
                 train_mean_scaled: pd.Series,
                 train_std_scaled: pd.Series,
                 reward_clip_range: Optional[Tuple[float, float]] = None
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
            self.token_data_cache = {
                token_id: group[feature_cols + ['label']].reset_index(drop=True)
                for token_id, group in self.grouped_data
            }
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
        self.reward_clip_range = reward_clip_range

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_features,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self.current_token_id = None
        self.current_token_data = None
        self.current_timestep_idx = 0
        self.max_timesteps_in_episode = 0

    def _get_current_token_data(self):
        if self.current_token_id is None:
            raise ValueError("Environment not reset. Call reset() before accessing token data.")
        return self.token_data_cache[self.current_token_id]

    def _get_obs(self):
        if self.current_token_data is None:
            raise ValueError("Environment not reset or token data is missing. Call reset().")

        if self.current_timestep_idx >= self.max_timesteps_in_episode:
             return np.zeros(self.observation_space.shape, dtype=np.float32)

        features = self.current_token_data.loc[self.current_timestep_idx, self.feature_cols].values.astype(np.float32)

        if np.any(~np.isfinite(features)):
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)

        features = np.clip(features, 0.0, 1.0)

        return features

    def _get_info(self):
        info: Dict[str, Any] = {
            "token_id": self.current_token_id,
            "current_timestep_in_token": self.current_timestep_idx,
            "total_timesteps_in_token": self.max_timesteps_in_episode
        }
        current_true_label_idx = self.current_timestep_idx
        if self.current_token_data is not None and 0 <= current_true_label_idx < self.max_timesteps_in_episode:
             info["true_label_at_current_t"] = int(self.current_token_data.loc[current_true_label_idx, "label"])

        return info

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_token_id = self.np_random.choice(self.token_ids)
        self.current_token_data = self._get_current_token_data()
        self.max_timesteps_in_episode = len(self.current_token_data)
        self.current_timestep_idx = 0

        observation = self._get_obs()
        info = self._get_info()
        info["start_token_id"] = self.current_token_id

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}. Action must be 0 or 1.")
        if self.current_token_data is None:
            raise ValueError("Environment not reset. Call reset() before calling step().")

        label_timestep_idx = self.current_timestep_idx

        if label_timestep_idx < 0 or label_timestep_idx >= self.max_timesteps_in_episode:
             print(f"Warning: Invalid label_timestep_idx {label_timestep_idx} encountered for token {self.current_token_id}. Max steps: {self.max_timesteps_in_episode}.")
             true_label_for_reward = 0
             label_timestep_idx = self.current_timestep_idx -1
             if label_timestep_idx < 0: label_timestep_idx = 0
             true_label_for_reward = int(self.current_token_data.loc[label_timestep_idx, "label"])

        else:
             true_label_for_reward = int(self.current_token_data.loc[label_timestep_idx, "label"])

        raw_reward = 0.0
        if action == 0 and true_label_for_reward == 0: raw_reward = 0.5
        elif action == 0 and true_label_for_reward == 1: raw_reward = -2.0
        elif action == 1 and true_label_for_reward == 0: raw_reward = -1.0
        elif action == 1 and true_label_for_reward == 1: raw_reward = 1.0

        if self.reward_clip_range is not None:
            reward = np.clip(raw_reward, self.reward_clip_range[0], self.reward_clip_range[1])
        else:
            reward = raw_reward

        self.current_timestep_idx += 1

        terminated = False
        truncated = self.current_timestep_idx >= self.max_timesteps_in_episode

        observation = self._get_obs()
        info = self._get_info()

        info["action_taken"] = action
        info["raw_reward"] = raw_reward
        info["reward_obtained"] = reward
        info["true_label_for_reward"] = true_label_for_reward

        return observation, reward, terminated, truncated, info

    def render(self, mode: str = "human"):
        if self.current_token_id is None:
            print("Environment not reset.")
            return

        step_to_render = self.current_timestep_idx - 1

        if step_to_render < 0:
            output = f"Token: {self.current_token_id} - Start of Episode"
        elif step_to_render < self.max_timesteps_in_episode:
             features_now = self.current_token_data.loc[step_to_render, self.feature_cols]
             label_now = int(self.current_token_data.loc[step_to_render, "label"])
             features_str = " | ".join([f"{col}: {val:.3f}" for col, val in features_now.items()])
             output = (f"Token: {self.current_token_id} | "
                       f"Render Step t={step_to_render}/{self.max_timesteps_in_episode-1} | "
                       f"TrueLbl(t): {label_now} | "
                       f"Feat(t): {features_str}")
        else:
             output = f"Token: {self.current_token_id} - End of Episode (t={step_to_render})"

        if mode == "human":
            print(output)
        elif mode == "ansi":
            return output

    def close(self):
        self.data = None
        self.token_data_cache = {}
        print("Environment closed.")