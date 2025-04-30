# my_gym.py (Final Config: Delta Features + Tuned Rewards)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os

class ScamTokenEnv(gym.Env):
    """
    Scam Token Detection Environment (Final Config)
    - Loads data, normalizes
    - Adds delta features for top10_holder_pct, volume_spike_ratio
    - Uses tuned reward structure: Legit TP=+2, FP=-5, FN=-5
    """
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, parquet_file: str, max_steps=10, # Keeping max_steps=10
                 feature_cols=None, train_mean=None, train_std=None, subset_rows=None):
        super().__init__()

        if not os.path.exists(parquet_file):
             raise FileNotFoundError(f"Error: Parquet file not found at {parquet_file}")

        self.data_full = pd.read_parquet(parquet_file)
        if subset_rows is not None and subset_rows < len(self.data_full):
            print(f"Using subset of {subset_rows} rows from {parquet_file}.")
            self.data = self.data_full.head(subset_rows).reset_index(drop=True)
        else:
            # Using full dataset if subset_rows is None
            print(f"Using full dataset from {parquet_file}.")
            self.data = self.data_full.reset_index(drop=True)

        self.feature_cols = feature_cols if feature_cols else [
            'liquidity_lock', 'top5_holder_pct', 'top10_holder_pct',
            'volume_spike_ratio', 'comment_velocity', 'comment_spam_ratio',
            'hype_score', 'bundled_rug_selloff', 'dev_wallet_reputation',
            'wallet_clustering_hhi'
        ]
        self.delta_feature_cols = ['top10_holder_pct', 'volume_spike_ratio'] # Keep delta features

        # --- Column existence checks ---
        missing_cols = [col for col in self.feature_cols if col not in self.data.columns]
        if missing_cols: raise ValueError(f"Missing required feature columns: {missing_cols}")
        missing_delta_cols = [col for col in self.delta_feature_cols if col not in self.data.columns]
        if missing_delta_cols: raise ValueError(f"Missing columns needed for delta features: {missing_delta_cols}")
        try: self.labels = self.data["label"].values.astype(int)
        except KeyError: raise KeyError(f"'label' column not found in {parquet_file}.")
        # ---

        self.raw_features_df = self.data[self.feature_cols]

        # --- Normalization ---
        if train_mean is None or train_std is None:
            self.features_mean = self.raw_features_df.mean()
            self.features_std = self.raw_features_df.std().replace(0, 1e-6)
            print("Calculated normalization stats from this dataset.")
        else:
             self.features_mean = train_mean; self.features_std = train_std
             print("Using pre-calculated normalization stats.")
        self.features_normalized = (self.raw_features_df - self.features_mean) / self.features_std
        self.features = self.features_normalized.values.astype(np.float32)
        # ---

        self.n_samples = len(self.data)
        print(f"Loaded {self.n_samples} samples. Base features: {self.feature_cols}")
        print(f"Delta features calculated for: {self.delta_feature_cols}")

        self.max_steps = max_steps

        # --- Observation space ---
        num_delta_features = len(self.delta_feature_cols)
        self.observation_space = spaces.Box(
            low=-6.0, high=6.0,
            shape=(len(self.feature_cols) + 1 + num_delta_features,), # base + step_norm + deltas
            dtype=np.float32
        )
        print(f"Observation space shape: {self.observation_space.shape}")
        # ---

        self.action_space = spaces.Discrete(4)
        self.previous_step_raw_features = None
        self.current_sample_idx = 0
        self.current_step = 0
        self.classified = False

    def _get_observation(self):
        if self.current_sample_idx >= self.n_samples: self.current_sample_idx = self.np_random.integers(0, self.n_samples) if self.n_samples > 0 else 0
        if self.n_samples == 0: return np.zeros(self.observation_space.shape, dtype=np.float32)

        current_normalized_features = self.features[self.current_sample_idx]
        current_raw_features = self.raw_features_df.iloc[self.current_sample_idx]
        delta_values = np.zeros(len(self.delta_feature_cols), dtype=np.float32)

        if self.current_step > 0 and self.previous_step_raw_features is not None:
            try:
                for i, col in enumerate(self.delta_feature_cols):
                    delta_values[i] = current_raw_features[col] - self.previous_step_raw_features[col]
            except Exception as e:
                print(f"Warning: Error calculating delta features: {e}. Setting delta to 0.")

        step_feature = np.array([self.current_step / self.max_steps], dtype=np.float32)
        observation = np.concatenate([current_normalized_features, step_feature, delta_values]).astype(np.float32)
        self.previous_step_raw_features = current_raw_features[self.delta_feature_cols].copy() # Store copy

        if observation.shape != self.observation_space.shape:
             correct_shape_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
             size = min(len(observation), len(correct_shape_obs)); correct_shape_obs[:size] = observation[:size]
             print(f"Warning: Observation shape mismatch fixed! Expected {self.observation_space.shape}, got {observation.shape}")
             return correct_shape_obs
        return observation

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.n_samples == 0:
            self.current_sample_idx = 0
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = {"true_label": -1}
            return observation, info
        self.current_sample_idx = self.np_random.integers(0, self.n_samples)
        self.current_step = 0
        self.classified = False
        self.previous_step_raw_features = None # Reset previous features
        observation = self._get_observation()
        info = {"true_label": int(self.labels[self.current_sample_idx])}
        return observation, info

    def step(self, action):
        if self.n_samples == 0:
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, True, True, {"true_label": -1}
        assert self.action_space.contains(action), f"Invalid action: {action}"

        true_label = self.labels[self.current_sample_idx]
        reward = 0.0
        terminated = False
        truncated = False

        # === Final Tuned Rewards ===
        if action == 0: reward = -0.1   # Wait
        elif action == 1: reward = -0.05 # Flag
        elif action == 2:               # Classify Legitimate
            if true_label == 0: reward = 2.0 # Tuned Legit TP reward
            else: reward = -5.0              # FN penalty
            self.classified = True; terminated = True
        elif action == 3:               # Classify Scam
            early_detection_bonus = max(0, (self.max_steps - self.current_step) / self.max_steps)
            if true_label == 1: reward = 5.0 + (early_detection_bonus * 2.0) # TP reward
            else: reward = -5.0              # Tuned FP penalty
            self.classified = True; terminated = True
        # ===========================

        # Store current raw features *before* incrementing step, for *next* step's delta calc
        # Handled within _get_observation now

        self.current_step += 1 # Increment step

        # Check for truncation
        if not terminated and self.current_step >= self.max_steps:
            truncated = True
            if not self.classified: # Apply penalty if no decision made by end
                reward += -5.0 if true_label == 1 else -0.5

        observation = self._get_observation() # Get observation for the *new* state
        info = {"true_label": int(true_label), "action_taken": int(action), "steps": self.current_step}
        return observation, reward, terminated, truncated, info

    def render(self, mode="human"): pass # Keep render simple or implement as needed

    def get_normalization_stats(self):
        if hasattr(self, 'features_mean') and hasattr(self, 'features_std'):
            return self.features_mean, self.features_std
        else:
            raise AttributeError("Normalization statistics not available.")