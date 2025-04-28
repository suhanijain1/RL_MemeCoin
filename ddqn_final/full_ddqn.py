# run_ddqn.py (Final Config: DeltaFeat + Tuned #2 Params - FULL DATASET RUN - Corrected Indentation)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# --- Import Environment ---
try:
    from my_gym2 import ScamTokenEnv # Assumes this is the final version with delta features and tuned rewards
except ImportError:
    print("Error: Could not import ScamTokenEnv from my_gym.py.")
    exit()
except Exception as e:
    print(f"An error occurred during import: {e}")
    exit()

# --- Deep Q-Network (DQN) ---
class DQNNetwork(nn.Module):
    """MLP for Q-value approximation (128x128)."""
    def __init__(self, state_dim, action_dim):
        super(DQNNetwork, self).__init__()
        hidden_size = 128 # Final network size
        self.layer1 = nn.Linear(state_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# --- Standard Experience Replay Buffer ---
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
class ReplayBuffer: # Uniform Buffer
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.experience = Experience

    def add(self, state, action, reward, next_state, done):
        self.memory.append(self.experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        # Handle case where buffer is smaller than batch_size
        actual_batch_size = min(batch_size, len(self.memory))
        if actual_batch_size <= 0: return [] # Cannot sample if empty
        return random.sample(self.memory, k=actual_batch_size)

    def __len__(self):
        return len(self.memory)

# --- Double DQN Agent (Standard Uniform Replay) ---
class DoubleDQNAgent:
    def __init__(self, state_dim, action_dim, buffer_size=10000, batch_size=64,
                 gamma=0.99, lr=1e-4, tau=1e-3, update_every=4, device="cpu"):
        self.state_dim, self.action_dim = state_dim, action_dim
        self.buffer_size = buffer_size
        self.batch_size, self.gamma, self.tau, self.update_every = batch_size, gamma, tau, update_every
        self.device = device
        self.qnetwork_local = DQNNetwork(state_dim, action_dim).to(device)
        self.qnetwork_target = DQNNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.qnetwork_target.eval()
        self.memory = ReplayBuffer(buffer_size)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        # Ensure buffer is large enough before sampling
        if self.t_step == 0 and len(self.memory) >= self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            if not experiences: return None # Should not happen if len >= batch_size
            loss = self.learn(experiences)
            return loss.item()
        return None

    def choose_action(self, state, epsilon=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_dim))

    def get_greedy_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        return np.argmax(action_values.cpu().data.numpy())

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.float32)).float().to(self.device)
        # Double DQN update
        q_local_next_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
        q_targets_next = self.qnetwork_target(next_states).detach().gather(1, q_local_next_actions)
        q_targets = rewards + (self.gamma * q_targets_next * (1.0 - dones))
        q_expected = self.qnetwork_local(states).gather(1, actions)
        # Loss calculation
        loss = F.smooth_l1_loss(q_expected, q_targets)
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), 1.0)
        self.optimizer.step()
        # Update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
        return loss

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self, filepath):
        torch.save(self.qnetwork_local.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
         if os.path.exists(filepath):
             state_dict = torch.load(filepath, map_location=self.device)
             self.qnetwork_local.load_state_dict(state_dict)
             self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
             self.qnetwork_local.eval()
             self.qnetwork_target.eval()
             print(f"Model loaded from {filepath}")
             return True
         else:
             print(f"Warning: Model file not found at {filepath}. Starting with random weights.")
             return False

# --- Training Function ---
def train_agent(env, agent, n_episodes=2000, max_t=10, eps_start=1.0, eps_end=0.01, eps_decay=0.995,
                print_every=100, checkpoint_path="ddqn_checkpoint.pth", final_model_path="ddqn_final.pth"):
    scores, losses = [], []
    scores_window = deque(maxlen=print_every)
    eps = eps_start
    start_time = time.time()
    print(f"Starting training for {n_episodes} episodes (max_t = {max_t})...")
    for i_episode in range(1, n_episodes + 1):
        state, info = env.reset()
        score = 0
        episode_losses = []
        if state is None:
            continue
        for t in range(max_t):
            action = agent.choose_action(state, eps)
            try:
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            except ValueError as e:
                print(f"Error during env.step(): {e}")
                break
            loss = agent.step(state, action, reward, next_state, done)
            if loss is not None:
                episode_losses.append(loss)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        losses.append(np.mean(episode_losses) if episode_losses else 0)
        eps = max(eps_end, eps_decay * eps)
        if i_episode % print_every == 0:
            avg_score = np.mean(scores_window) if scores_window else 0
            avg_loss = np.mean(losses[-print_every:]) if len(losses) >= print_every else (np.mean(losses) if losses else 0)
            elapsed_time = time.time() - start_time
            print(f'Episode {i_episode}\tAvg Score: {avg_score:.2f}\tAvg Loss: {avg_loss:.4f}\tEpsilon: {eps:.4f}\tTime: {elapsed_time:.1f}s')
            agent.save(checkpoint_path)
    print(f"\nTraining finished after {n_episodes} episodes.")
    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds")
    agent.save(final_model_path)
    print(f"Final model explicitly saved to {final_model_path}")
    return scores, losses

# --- Evaluation Function ---
def evaluate_agent(env_test, agent, num_test_episodes=1000, final_model_path="ddqn_final.pth"):
    print("\n--- Starting Evaluation ---")
    if not env_test or env_test.n_samples == 0:
        print("Test environment invalid/empty. Skipping evaluation.")
        return
    if not agent.load(final_model_path):
        print(f"Could not load final model from {final_model_path}. Cannot evaluate.")
        return
    agent.qnetwork_local.eval()
    true_labels = []
    predictions = []
    num_episodes_to_run = min(num_test_episodes, env_test.n_samples)
    print(f"Evaluating on {num_episodes_to_run} episodes (max_steps = {env_test.max_steps})...")
    for i in range(num_episodes_to_run):
        state, info = env_test.reset()
        true_label = info.get('true_label', -1)
        if true_label == -1:
            continue
        final_prediction = -1
        for t in range(env_test.max_steps):
            action = agent.get_greedy_action(state)
            if action == 2:
                final_prediction = 0
                break
            elif action == 3:
                final_prediction = 1
                break
            try:
                next_state, _, terminated, truncated, _ = env_test.step(action)
            except ValueError as e:
                print(f"Error during env_test.step(): {e}")
                break
            state = next_state
            if terminated or truncated:
                break
        if final_prediction == -1:
             state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(agent.device)
             with torch.no_grad():
                 q_values = agent.qnetwork_local(state_tensor).squeeze()
             if torch.is_tensor(q_values) and q_values.numel() >= 4:
                 final_prediction = 0 if q_values[2] >= q_values[3] else 1
             else:
                 final_prediction = 0 # Default fallback
                 print(f"Warning: Unexpected Q-values shape. Defaulting prediction.")
        true_labels.append(true_label)
        predictions.append(final_prediction)
        if (i + 1) % (num_episodes_to_run // 10 if num_episodes_to_run >= 10 else 1) == 0:
            print(f"  Evaluated episode {i+1}/{num_episodes_to_run}")
    if not true_labels or not predictions:
        print("No valid evaluation results collected.")
        return
    print("\n--- Evaluation Results ---")
    try:
        valid_indices = [i for i, label in enumerate(true_labels) if label in [0, 1]]
        filtered_true = [true_labels[i] for i in valid_indices]
        filtered_pred = [predictions[i] for i in valid_indices]
        if not filtered_true or not filtered_pred:
            print("No valid labels/predictions after filtering.")
            return
        report = classification_report(filtered_true, filtered_pred, target_names=['Legitimate', 'Scam'], zero_division=0)
        print("Classification Report:")
        print(report)
        cm = confusion_matrix(filtered_true, filtered_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legitimate', 'Scam'])
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
        plt.title(f'Confusion Matrix (Final Config - Full Dataset)') # Updated title
        plt.tight_layout()
        plt.savefig("confusion_matrix_final_full.png") # Updated filename
        print("Confusion matrix saved as confusion_matrix_final_full.png")
        plt.show()
    except Exception as e:
        print(f"Error during metrics calculation or plotting: {e}")

# --- Plotting Function ---
def plot_results(scores, losses, filename="training_performance_final_full.png", title_suffix=" (Final Config - Full Dataset)"): # Updated filename/title
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    moving_avg_window = 100 # Can increase for longer runs, e.g., 500
    ax1.plot(np.arange(1, len(scores) + 1), scores, label='Episode Score', alpha=0.6)
    if len(scores) >= moving_avg_window:
        moving_avg = np.convolve(scores, np.ones(moving_avg_window)/moving_avg_window, mode='valid')
        ax1.plot(np.arange(moving_avg_window, len(scores) + 1), moving_avg, label=f'{moving_avg_window}-Ep Moving Avg', color='red')
    ax1.set_ylabel('Score')
    ax1.set_title('Training Scores' + title_suffix)
    ax1.legend(); ax1.grid(True)
    valid_losses = [l for l in losses if l > 1e-9]
    valid_indices = [i + 1 for i, l in enumerate(losses) if l > 1e-9]
    if valid_losses:
        ax2.plot(valid_indices, valid_losses, label='Avg Episode Loss', alpha=0.8, marker='.', linestyle='None', markersize=2)
        if len(valid_losses) >= moving_avg_window:
            loss_moving_avg = np.convolve(valid_losses, np.ones(moving_avg_window)/moving_avg_window, mode='valid')
            ax2.plot(valid_indices[moving_avg_window - 1:], loss_moving_avg, label=f'{moving_avg_window}-Ep Loss Avg', color='orange')
    ax2.set_ylabel('Average Loss')
    ax2.set_xlabel('Episode #')
    ax2.set_title('Average Training Loss' + title_suffix)
    ax2.legend(); ax2.grid(True); ax2.set_yscale('log')
    plt.tight_layout(); plt.savefig(filename); print(f"Training performance graph saved as {filename}"); plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration for FULL DATASET ---
    TRAIN_FILE = "train_tokens.parquet"
    TEST_FILE = "test_tokens.parquet"
    SUBSET_ROWS = None # Use the full dataset
    MAX_STEPS_PER_EPISODE = 10 # Use 10 steps as per best observed config
    FEATURE_COLS = ['liquidity_lock', 'top5_holder_pct', 'top10_holder_pct', 'volume_spike_ratio', 'comment_velocity', 'comment_spam_ratio', 'hype_score', 'bundled_rug_selloff', 'dev_wallet_reputation', 'wallet_clustering_hhi']

    # Hyperparameters based on "Tuned #2" which performed best on subset with delta features
    BUFFER_SIZE = int(5e5)     # Larger buffer for full dataset
    BATCH_SIZE = 128
    GAMMA = 0.99
    TAU = 1e-3
    LR = 1e-4                  # Tuned LR
    UPDATE_EVERY = 2           # Tuned update frequency

    # Adjust training length and exploration for full dataset
    N_EPISODES = 30000         # Number of episodes for full run (Adjust if needed)
    EPS_START = 1.0
    EPS_END = 0.01
    EPS_DECAY = 0.9999         # Slower decay for more episodes (Adjust based on N_EPISODES)
    PRINT_EVERY = 500          # Print less often

    # Final File Names for this specific configuration run
    CHECKPOINT_PATH = "final_full_delta_tuned2_ddqn_checkpoint.pth"
    FINAL_MODEL_PATH = "final_full_delta_tuned2_ddqn_model.pth" # Consistent name

    NUM_TEST_EPISODES = 3000   # Evaluate on a larger portion of test set

    # === Setup ===
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"); print(f"Using device: {device}")
    print("--- Initializing Environments (Final Config - Full Dataset) ---")
    env_train = None; env_test = None; train_mean = None; train_std = None
    try:
        # Ensure my_gym.py has delta features and tuned rewards
        env_train = ScamTokenEnv(parquet_file=TRAIN_FILE, max_steps=MAX_STEPS_PER_EPISODE, feature_cols=FEATURE_COLS, subset_rows=SUBSET_ROWS)
        train_mean, train_std = env_train.get_normalization_stats(); state_dim = env_train.observation_space.shape[0]; action_dim = env_train.action_space.n
        print(f"Training Env: {env_train.n_samples} samples. State dim: {state_dim}");
        env_test = ScamTokenEnv(parquet_file=TEST_FILE, max_steps=MAX_STEPS_PER_EPISODE, feature_cols=FEATURE_COLS, train_mean=train_mean, train_std=train_std, subset_rows=SUBSET_ROWS)
        print(f"Test Env: {env_test.n_samples} samples.")
    except Exception as e:
        print(f"\n*** ERROR during environment setup: {e} ***"); exit()

    # --- Initialize Agent (Standard DDQN - 128x128 network) ---
    agent = DoubleDQNAgent(state_dim=state_dim, action_dim=action_dim, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE,
                           gamma=GAMMA, lr=LR, tau=TAU, update_every=UPDATE_EVERY, device=device)

    # --- Optional: Load Checkpoint if resuming ---
    # if os.path.exists(CHECKPOINT_PATH):
    #     agent.load(CHECKPOINT_PATH)
    #     print("Resuming training from checkpoint.")

    # --- Train ---
    print("\n--- Starting Training (Final Config - Full Dataset) ---")
    scores, losses = train_agent(env_train, agent, n_episodes=N_EPISODES, max_t=MAX_STEPS_PER_EPISODE,
                                 eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY,
                                 print_every=PRINT_EVERY, checkpoint_path=CHECKPOINT_PATH,
                                 final_model_path=FINAL_MODEL_PATH)

    # --- Plot Training Results ---
    plot_results(scores, losses, filename="training_performance_final_full.png", title_suffix=" (Final Config - Full Dataset)")

    # --- Evaluate ---
    # Ensure the final model is loaded for evaluation if training was interrupted/resumed
    evaluate_agent(env_test, agent, num_test_episodes=NUM_TEST_EPISODES, final_model_path=FINAL_MODEL_PATH)

    # --- Cleanup ---
    if env_train: env_train.close();
    if env_test: env_test.close();
    print("\n--- Finished (Final Config - Full Dataset) ---")