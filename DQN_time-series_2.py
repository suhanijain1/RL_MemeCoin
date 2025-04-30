import os
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# 1) Delayed reward environment
class DelayedClassificationEnv(gym.Env):
    """
    Each episode: predict multiple steps, accumulate correct/incorrect predictions,
    and at the end of the episode, calculate a delayed reward based on accuracy.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, parquet_file: str, episode_length: int = 10):
        super().__init__()
        df = pd.read_parquet(parquet_file)
        numeric = df.select_dtypes(include=[np.number])
        assert "label" in numeric.columns, "Need numeric 'label' column"
        self.features = numeric.drop(columns=["label"]).values.astype(np.float32)
        self.labels = numeric["label"].values.astype(np.int64)

        n_feats = self.features.shape[1]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_feats,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(2)
        
        # Number of steps per episode (for delayed reward)
        self.episode_length = episode_length
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = random.randrange(len(self.features))
        self.step_count = 0
        return self.features[self.idx], {}

    def step(self, action):
        true_label = self.labels[self.idx]
        reward = 1.0 if action == true_label else -1.0
        done = self.step_count >= self.episode_length
        truncated = False  # For simplicity, we don't use truncated in this setup
        self.step_count += 1

        # Return next state, reward, done flag, truncated flag, and any additional info
        return self.features[self.idx], reward, done, truncated, {'true_label_at_t': true_label}

# 2) Q-network
class QNetwork(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, 128), nn.ReLU(),
            nn.Linear(128,  64),      nn.ReLU(),
            nn.Linear(64,   out_dim)
        )

    def forward(self, x):
        return self.net(x)

# 3) Training & evaluation
def train_and_evaluate():
    PARQUET_FILE = "/Users/vardanvij/Downloads/creation/train_data_timeseries.parquet"
    EPISODES = 10000
    BATCH_SIZE = 64
    MEMORY_SIZE = 10000
    LR = 1e-3
    GAMMA = 0.9
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = 0.9995
    TARGET_UPDATE = 500

    env = DelayedClassificationEnv(PARQUET_FILE)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = QNetwork(state_dim, action_dim)
    target_net = QNetwork(state_dim, action_dim)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    criterion = nn.MSELoss()
    memory = deque(maxlen=MEMORY_SIZE)

    eps = EPS_START
    losses = []

    for step in range(1, EPISODES + 1):
        state, _ = env.reset()
        # ε-greedy
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_vals = q_net(torch.from_numpy(state))
                action = int(q_vals.argmax().item())

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        memory.append((state, action, reward, next_state, done))

        if len(memory) >= BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            s_list, a_list, r_list, ns_list, d_list = zip(*batch)
            s_np = np.stack(s_list)
            ns_np = np.stack(ns_list)
            a_np = np.array(a_list)
            r_np = np.array(r_list, dtype=np.float32)
            d_np = np.array(d_list, dtype=np.float32)

            s_tensor = torch.from_numpy(s_np).float()
            ns_tensor = torch.from_numpy(ns_np).float()
            a_tensor = torch.from_numpy(a_np).long()
            r_tensor = torch.from_numpy(r_np)
            d_tensor = torch.from_numpy(d_np)

            q_vals = q_net(s_tensor).gather(1, a_tensor.unsqueeze(1)).squeeze()
            next_q = target_net(ns_tensor).max(1)[0].detach()
            target = r_tensor + GAMMA * next_q * (1 - d_tensor)
            loss = criterion(q_vals, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # decay epsilon
        eps = max(EPS_END, eps * EPS_DECAY)

        # update target network
        if step % TARGET_UPDATE == 0:
            target_net.load_state_dict(q_net.state_dict())

    # Plot loss convergence
    plt.figure(figsize=(6,4))
    plt.plot(losses)
    plt.title("DQN MSE Loss Over Training")
    plt.xlabel("Training Steps")
    plt.ylabel("MSE Loss")
    plt.show()

    # Full-dataset evaluation
    y_true, y_pred = [], []
    for _ in range(len(env.features)):
        state, _ = env.reset()
        with torch.no_grad():
            pred = int(q_net(torch.from_numpy(state)).argmax().item())
        y_true.append(env.labels[env.idx])
        y_pred.append(pred)

    # Compute overall accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {acc:.3f}")

    # Per-class precision, recall, F1, support
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    for cls in [0, 1]:
        print(f"Class {cls} → Precision: {p[cls]:.3f}, Recall: {r[cls]:.3f},"
              f" F1: {f1[cls]:.3f}, Support: {sup[cls]}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ['Legit', 'Scam'])
    plt.yticks([0, 1], ['Legit', 'Scam'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    train_and_evaluate()
