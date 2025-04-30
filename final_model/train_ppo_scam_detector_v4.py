import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import os
import time
import traceback

try:
    from env_timeseries_v4 import ScamTokenTimeseriesEnvV4
except ImportError:
    print("Error: Could not import ScamTokenTimeseriesEnvV4.")
    print("Make sure 'env_timeseries_v4.py' is in the same directory or Python path.")
    exit(1)

TRAIN_PARQUET = 'train_data_timeseries.parquet'
EVAL_PARQUET = 'test_data_timeseries.parquet'
FEATURE_COLS = [
    'liquidity_lock', 'top5_holder_pct', 'top10_holder_pct',
    'volume_spike_ratio', 'comment_velocity', 'comment_spam_ratio',
    'hype_score', 'bundled_rug_selloff', 'dev_wallet_reputation',
    'wallet_clustering_hhi'
]
SCALER_MEAN_FILE = 'train_mean_timeseries_scaled01.csv'
SCALER_STD_FILE = 'train_std_timeseries_scaled01.csv'
try:
    TRAIN_MEAN_SCALED = pd.read_csv(SCALER_MEAN_FILE, index_col=0).squeeze("columns").reindex(FEATURE_COLS)
    TRAIN_STD_SCALED = pd.read_csv(SCALER_STD_FILE, index_col=0).squeeze("columns").reindex(FEATURE_COLS).replace(0, 1e-6)
    if not isinstance(TRAIN_MEAN_SCALED, pd.Series) or not isinstance(TRAIN_STD_SCALED, pd.Series): raise TypeError("Loaded mean/std is not a Pandas Series.")
    if TRAIN_MEAN_SCALED.isnull().any() or TRAIN_STD_SCALED.isnull().any(): raise ValueError("Loaded mean/std contains NaN values after reindexing.")
    print("Loaded SCALED mean and std dev values successfully.")
except Exception as e:
    print(f"\nFATAL ERROR loading/processing scaling info: {e}")
    exit(1)

TRUNCATION_PENALTY = -0.5
SEED = 42
N_ENVS = 4
TOTAL_TIMESTEPS = 1_000_000
EVAL_FREQ = 25000
N_EVAL_EPISODES = 50
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
GAMMA = 0.99
ENT_COEF = 0.01
USE_SUBPROC_VEC_ENV = True

TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")

MODEL_NAME = f"ppo_scam_detector_v4_{TIMESTAMP}"
SAVE_DIR = "./models_timeseries_v4"
LOG_DIR = "./logs_timeseries_v4"
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, MODEL_NAME)
TENSORBOARD_LOG_PATH = os.path.join(LOG_DIR, MODEL_NAME + "_tb")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_PATH, exist_ok=True)

def make_env(parquet_file, rank, seed=0, trunc_penalty=-0.5):
    def _init():
        try:
            env_instance = ScamTokenTimeseriesEnvV4(
                parquet_file=parquet_file,
                feature_cols=FEATURE_COLS,
                train_mean_scaled=TRAIN_MEAN_SCALED,
                train_std_scaled=TRAIN_STD_SCALED,
                truncation_penalty=trunc_penalty
            )
            log_file = os.path.join(LOG_DIR, f"monitor_{rank}.csv") if rank is not None else None
            env_monitor = Monitor(env_instance, filename=log_file, allow_early_resets=True)
            env_monitor.reset(seed=seed + rank if rank is not None else seed)
            return env_monitor
        except Exception as e:
            print(f"FATAL ERROR (Rank {rank}): Error creating env V4: {e}. Exiting subprocess.")
            traceback.print_exc()
            exit(1)
    return _init

if __name__ == "__main__":
    print("--- Starting Training Script (V4 - Agent Decides Termination) ---")
    print(f"Reward Mechanism: Terminal classification (+1/-1), Truncation ({TRUNCATION_PENALTY})")
    print(f"Gamma: {GAMMA}")

    if not os.path.exists(TRAIN_PARQUET): print(f"FATAL: Training Parquet not found: '{TRAIN_PARQUET}'"); exit(1)
    if not os.path.exists(EVAL_PARQUET): print(f"FATAL: Evaluation Parquet not found: '{EVAL_PARQUET}'"); exit(1)

    print("\nCreating Training Environments...")
    VecEnvClass = SubprocVecEnv if USE_SUBPROC_VEC_ENV else DummyVecEnv
    train_env = None 
    try:
        env_fns = [make_env(TRAIN_PARQUET, rank=i, seed=SEED, trunc_penalty=TRUNCATION_PENALTY) for i in range(N_ENVS)]
        train_env = VecEnvClass(env_fns)
        print(f"Successfully created {N_ENVS} training environments ({VecEnvClass.__name__}).")
    except Exception as e:
         print(f"\nFATAL: Failed to create training environments VecEnv: {e}"); traceback.print_exc(); exit(1)

    print("\nCreating Evaluation Environment...")
    eval_env_instance = None
    try:
        eval_initializer = make_env(EVAL_PARQUET, rank=N_ENVS, seed=SEED, trunc_penalty=TRUNCATION_PENALTY)
        eval_env_instance = eval_initializer()
        print("Successfully created evaluation environment.")
    except Exception as e:
         print(f"\nERROR: Failed to create evaluation environment: {e}"); traceback.print_exc(); eval_env_instance = None

    eval_callback = None
    if eval_env_instance is not None:
        print("\nSetting up Evaluation Callback...")
        eval_callback = EvalCallback(
            eval_env_instance,
            best_model_save_path=os.path.join(SAVE_DIR, 'best_model_' + MODEL_NAME),
            log_path=os.path.join(LOG_DIR, 'eval_logs_' + MODEL_NAME),
            eval_freq=max(EVAL_FREQ // N_ENVS, 1),
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True,
            render=False,
            verbose=1
        )
        print("Evaluation callback configured. Monitors mean terminal episode reward.")
    else:
        print("Skipping evaluation callback setup as eval_env failed creation.")

    print("\nInitializing PPO Agent...")
    device = "auto"
    model = None 
    try:
        model = PPO(
            "MlpPolicy", train_env, verbose=1, tensorboard_log=TENSORBOARD_LOG_PATH,
            learning_rate=LEARNING_RATE, n_steps=N_STEPS, batch_size=BATCH_SIZE,
            n_epochs=10, gamma=GAMMA, gae_lambda=0.95, clip_range=0.2,
            ent_coef=ENT_COEF, vf_coef=0.5, max_grad_norm=0.5, seed=SEED, device=device
        )
        print(f"PPO Agent Initialized. Using device: {model.device}")
    except Exception as e:
        print(f"FATAL: Error initializing PPO Agent: {e}"); traceback.print_exc(); exit(1)


    print(f"\nStarting Training for {TOTAL_TIMESTEPS} timesteps...")
    final_model_path = None 
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=eval_callback,
            log_interval=10,
            tb_log_name="PPO_Run_V4"
        )
        print("\nTraining Finished.")
        final_model_path = f"{MODEL_SAVE_PATH}_final.zip"
        print(f"Saving final model to {final_model_path}")
        model.save(final_model_path)
        print("Final model saved.")
    except KeyboardInterrupt:
        print("\nTraining interrupted."); final_model_path = f"{MODEL_SAVE_PATH}_interrupted.zip"
        if model is not None: print(f"Saving model to {final_model_path}"); model.save(final_model_path); print("Model saved.")
        else: print("Model not initialized, cannot save.")
    except Exception as e:
        print(f"\nError during training: {e}"); traceback.print_exc()
        final_model_path = f"{MODEL_SAVE_PATH}_error.zip"; print(f"Attempting save to {final_model_path}...")
        if model is not None:
            try: model.save(final_model_path); print("Model saved.")
            except Exception as save_e: print(f"Could not save model after error: {save_e}")
        else: print("Model not initialized, cannot save.")


    print("\n--- Post-Training Evaluation (V4 Env) ---")
    best_model_path = os.path.join(SAVE_DIR, 'best_model_' + MODEL_NAME + '.zip')
    final_model_path_load = final_model_path if final_model_path else f"{MODEL_SAVE_PATH}_final.zip" # Use saved path if available

    eval_model_path = None
    if os.path.exists(best_model_path): eval_model_path = best_model_path
    elif final_model_path_load and os.path.exists(final_model_path_load): eval_model_path = final_model_path_load
    else:
        interrupted = f"{MODEL_SAVE_PATH}_interrupted.zip"; error = f"{MODEL_SAVE_PATH}_error.zip"
        if os.path.exists(interrupted): eval_model_path = interrupted
        elif os.path.exists(error): eval_model_path = error
    if eval_model_path: print(f"Selected model for evaluation: {eval_model_path}")
    else: print("No saved model found for evaluation.")

    if eval_model_path and eval_env_instance is not None:
        temp_eval_vec_env = None 
        try:
            print(f"Loading model from {eval_model_path} for evaluation...")
            temp_eval_vec_env = DummyVecEnv([lambda: eval_env_instance])
            eval_model = PPO.load(eval_model_path, env=temp_eval_vec_env, device=device)
            print("Model loaded. Running final evaluation...")

            total_eval_episodes_final = 100
            obs = eval_model.env.reset()
            episodes_done = 0
            episode_rewards = []
            final_predictions = []
            final_true_labels = []
            final_actions = []
            episode_lengths = []

            while episodes_done < total_eval_episodes_final:
                action, _states = eval_model.predict(obs, deterministic=True)
                obs, reward, terminated, infos = eval_model.env.step(action)

                done = terminated[0] or infos[0].get('truncated', False)
                info = infos[0]

                if done:
                    episodes_done += 1
                    ep_rew = info.get('episode', {}).get('r')
                    ep_len = info.get('episode', {}).get('l')
                    true_label = info.get('true_token_label')

                    final_action = -1 
                    last_step_action = info.get('action_taken') # Get action from final step info
                    if last_step_action is not None:
                        if terminated[0]: # Actions 1 or 2 cause termination
                            final_action = last_step_action
                        elif infos[0].get('truncated', False): # Action 0 causes truncation
                            final_action = 0
                        else: 
                             print(f"Warning: Episode done but neither terminated nor truncated? Last action: {last_step_action}, Reward: {ep_rew}")
                             final_action = last_step_action
                    elif ep_rew is not None: 
                         trunc_penalty = eval_env_instance.truncation_penalty 
                         if ep_rew == 1.0 or ep_rew == -1.0:
                             if (ep_rew == 1.0 and true_label == 0) or (ep_rew == -1.0 and true_label == 1): final_action = 1
                             elif (ep_rew == 1.0 and true_label == 1) or (ep_rew == -1.0 and true_label == 0): final_action = 2
                         elif np.isclose(ep_rew, trunc_penalty): final_action = 0
                         else: print(f"Warning: Cannot infer final action from reward {ep_rew}")
                    else:
                         print("Warning: Cannot determine final action (missing info and reward)")


                    if ep_rew is not None: episode_rewards.append(ep_rew)
                    if ep_len is not None: episode_lengths.append(ep_len)
                    final_actions.append(final_action)

                    if final_action == 1 or final_action == 2:
                         predicted_label = 0 if final_action == 1 else 1
                         final_predictions.append(predicted_label)
                         if true_label is not None: final_true_labels.append(true_label)
                         else: print("Warning: Missing true label for classified episode")


                    print(f"  Eval Episode {episodes_done}/{total_eval_episodes_final} | Reward: {ep_rew if ep_rew is not None else 'N/A':.2f} | Steps: {ep_len if ep_len is not None else 'N/A'} | Last Action: {final_action}", end='\r')


            print("\n\n--- Final Evaluation Results (V4 Env) ---")
            if episode_rewards:
                 avg_reward = np.mean(episode_rewards); std_reward = np.std(episode_rewards)
                 print(f"  Avg EPISODE Reward ({total_eval_episodes_final} eps): {avg_reward:.4f} +/- {std_reward:.4f}")
            else: print("  No episode rewards recorded during evaluation.")
            if episode_lengths:
                 avg_len = np.mean(episode_lengths); std_len = np.std(episode_lengths)
                 print(f"  Avg EPISODE Length ({total_eval_episodes_final} eps): {avg_len:.2f} +/- {std_len:.2f} steps")
            else: print("  No episode lengths recorded during evaluation.")


            action_counts = {0: 0, 1: 0, 2: 0, -1: 0} 
            for act in final_actions: action_counts[act] = action_counts.get(act, 0) + 1
            total_ended_episodes = action_counts[0] + action_counts[1] + action_counts[2] + action_counts[-1]
            if total_ended_episodes > 0:
                trunc_pct = (action_counts[0] / total_ended_episodes) * 100
                classify_legit_pct = (action_counts[1] / total_ended_episodes) * 100
                classify_scam_pct = (action_counts[2] / total_ended_episodes) * 100
                unknown_pct = (action_counts[-1] / total_ended_episodes) * 100
                print(f"\n  Episode End Action Distribution:")
                print(f"    Observe (Truncated): {action_counts[0]} ({trunc_pct:.1f}%)")
                print(f"    Classify Legit (Act 1): {action_counts[1]} ({classify_legit_pct:.1f}%)")
                print(f"    Classify Scam (Act 2): {action_counts[2]} ({classify_scam_pct:.1f}%)")
                if action_counts[-1] > 0: print(f"    Unknown: {action_counts[-1]} ({unknown_pct:.1f}%)")


            print(f"\n  Classification Metrics (based on {len(final_predictions)} classified episodes):")
            if final_predictions and final_true_labels and len(final_predictions)==len(final_true_labels):
                from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

                accuracy = accuracy_score(final_true_labels, final_predictions)
                report = classification_report(final_true_labels, final_predictions, target_names=['Legit (0)', 'Scam (1)'], zero_division=0, labels=[0,1])
                cm = confusion_matrix(final_true_labels, final_predictions, labels=[0, 1])

                print(f"    Accuracy (Classified only): {accuracy:.4f}")
                print("    Classification Report (Classified only):")
                print(report)
                print("    Confusion Matrix (Classified only - Pred\\True):")
                print(cm)
            elif not final_predictions:
                 print("    Could not calculate classification metrics (no episodes ended with action 1 or 2).")
            else:
                 print(f"    Could not calculate classification metrics (label mismatch: {len(final_predictions)} preds vs {len(final_true_labels)} true).")


        except Exception as e:
            print(f"\nAn error occurred during final evaluation: {e}")
            traceback.print_exc()
        finally: 
             if temp_eval_vec_env is not None:
                  print("Closing temporary evaluation VecEnv...")
                  temp_eval_vec_env.close()
    else:
        print("Skipping post-training evaluation (no model path or eval_env instance).")


    print("\nClosing environments...")
    try:
        if train_env is not None: train_env.close()
    except Exception as e: print(f"Error closing training environments: {e}")

    print("\n--- Script Finished (V4) ---")