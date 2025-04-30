import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os
import warnings
import time

try:
    from env_timeseries_v2 import ScamTokenTimeseriesEnvV2
except ImportError:
    print("Error: Could not import ScamTokenTimeseriesEnvV2.")
    print("Make sure 'env_timeseries_v2.py' is in the same directory or Python path.")
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
    if not isinstance(TRAIN_MEAN_SCALED, pd.Series) or not isinstance(TRAIN_STD_SCALED, pd.Series):
        raise TypeError("Loaded mean/std is not a Pandas Series.")
    if TRAIN_MEAN_SCALED.isnull().any() or TRAIN_STD_SCALED.isnull().any():
         raise ValueError("Loaded mean/std contains NaN values after reindexing. Check FEATURE_COLS match CSV index.")
    print("Loaded SCALED mean and std dev values successfully.")
except FileNotFoundError as e:
    print(f"\nFATAL ERROR: Could not find scaling info file: {e}. Run 'create_timeseries_data.py' first.")
    exit(1)
except Exception as e:
    print(f"\nFATAL ERROR: Failed to load or process scaling info: {e}")
    exit(1)

REWARD_CLIP_RANGE = (-2.0, 1.0)
SEED = 42
N_ENVS = 4
TOTAL_TIMESTEPS = 1_000_000
EVAL_FREQ = 25000
N_EVAL_EPISODES = 50
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
ENT_COEF = 0.01
USE_SUBPROC_VEC_ENV = True

TIMESTAMP = time.strftime("%Y%m%d-%H%M%S")
MODEL_NAME = f"ppo_scam_detector_v2_{TIMESTAMP}"
SAVE_DIR = "./models_timeseries"
LOG_DIR = "./logs_timeseries"
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, MODEL_NAME)
TENSORBOARD_LOG_PATH = os.path.join(LOG_DIR, MODEL_NAME + "_tb")

os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_PATH, exist_ok=True)

def make_env(parquet_file, rank, seed=0, reward_clip=None):
    def _init():
        try:
            env_instance = ScamTokenTimeseriesEnvV2(
                parquet_file=parquet_file,
                feature_cols=FEATURE_COLS,
                train_mean_scaled=TRAIN_MEAN_SCALED,
                train_std_scaled=TRAIN_STD_SCALED,
                reward_clip_range=reward_clip
            )
            log_file = os.path.join(LOG_DIR, f"monitor_{rank}.csv") if rank is not None else None
            env_monitor = Monitor(env_instance, filename=log_file, allow_early_resets=True)
            env_monitor.reset(seed=seed + rank if rank is not None else seed)
            return env_monitor
        except FileNotFoundError:
            print(f"FATAL ERROR (Rank {rank}): Parquet file not found at {parquet_file}. Exiting subprocess.")
            exit(1)
        except ValueError as e:
            print(f"FATAL ERROR (Rank {rank}): Value error creating env: {e}. Exiting subprocess.")
            exit(1)
        except Exception as e:
            print(f"FATAL ERROR (Rank {rank}): Unexpected error creating env: {e}. Exiting subprocess.")
            import traceback
            traceback.print_exc()
            exit(1)
    return _init

if __name__ == "__main__":
    print("--- Starting Training Script ---")
    print(f"Using Training Data: {TRAIN_PARQUET}")
    print(f"Using Evaluation Data: {EVAL_PARQUET}")
    print(f"Feature Columns: {FEATURE_COLS}")
    print(f"Using {N_ENVS} parallel environments.")

    if not os.path.exists(TRAIN_PARQUET):
        print(f"FATAL: Training Parquet file not found at '{TRAIN_PARQUET}'. Run 'create_timeseries_data.py' first.")
        exit(1)
    if not os.path.exists(EVAL_PARQUET):
        print(f"FATAL: Evaluation Parquet file not found at '{EVAL_PARQUET}'. Run 'create_timeseries_data.py' first.")
        exit(1)

    print("\nCreating Training Environments...")
    VecEnvClass = SubprocVecEnv if USE_SUBPROC_VEC_ENV else DummyVecEnv
    try:
        env_fns = []
        for i in range(N_ENVS):
            env_fns.append(make_env(TRAIN_PARQUET, rank=i, seed=SEED, reward_clip=REWARD_CLIP_RANGE))

        train_env = VecEnvClass(env_fns)
        print(f"Successfully created {N_ENVS} training environments ({VecEnvClass.__name__}).")
    except Exception as e:
         print(f"\nFATAL: Failed to create training environments VecEnv: {e}")
         import traceback
         traceback.print_exc()
         print("Check VecEnv setup and dependencies.")
         exit(1)

    print("\nCreating Evaluation Environment...")
    eval_env_instance = None
    try:
        eval_initializer = make_env(EVAL_PARQUET, rank=N_ENVS, seed=SEED, reward_clip=REWARD_CLIP_RANGE)
        eval_env_instance = eval_initializer()
        print("Successfully created evaluation environment.")
    except Exception as e:
         print(f"\nERROR: Failed to create evaluation environment: {e}")
         import traceback
         traceback.print_exc()
         print("Evaluation callback will not function correctly. Continuing without evaluation.")

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
        print("Evaluation callback configured.")
    else:
        print("Skipping evaluation callback setup as eval_env failed creation.")

    print("\nInitializing PPO Agent...")
    device = "auto"
    model = PPO(
        "MlpPolicy", train_env, verbose=1, tensorboard_log=TENSORBOARD_LOG_PATH,
        learning_rate=LEARNING_RATE, n_steps=N_STEPS, batch_size=BATCH_SIZE,
        n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
        ent_coef=ENT_COEF, vf_coef=0.5, max_grad_norm=0.5, seed=SEED, device=device
    )
    print(f"PPO Agent Initialized. Using device: {model.device}")

    print(f"\nStarting Training for {TOTAL_TIMESTEPS} timesteps...")

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=eval_callback,
            log_interval=10,
            tb_log_name="PPO_Run"
        )
        print("\nTraining Finished.")
        final_model_path = f"{MODEL_SAVE_PATH}_final.zip"
        print(f"Saving final model to {final_model_path}")
        model.save(final_model_path)
        print("Final model saved.")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        final_model_path = f"{MODEL_SAVE_PATH}_interrupted.zip"
        print(f"Saving model at interruption point to {final_model_path}")
        model.save(final_model_path)
        print("Model saved.")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        import traceback
        traceback.print_exc()
        final_model_path = f"{MODEL_SAVE_PATH}_error.zip"
        print(f"Attempting to save model before exiting to {final_model_path}...")
        try:
            model.save(final_model_path)
            print("Model saved.")
        except Exception as save_e:
            print(f"Could not save model after error: {save_e}")

    print("\n--- Post-Training Evaluation ---")
    best_model_path = os.path.join(SAVE_DIR, 'best_model_' + MODEL_NAME + '.zip')
    final_model_path_load = locals().get('final_model_path', f"{MODEL_SAVE_PATH}_final.zip")

    eval_model_path = None
    if os.path.exists(best_model_path):
        eval_model_path = best_model_path
        print(f"Found best model saved by EvalCallback: {eval_model_path}")
    elif final_model_path_load and os.path.exists(final_model_path_load):
         eval_model_path = final_model_path_load
         print(f"Best model not found or training didn't finish, using last saved model: {eval_model_path}")
    else:
        interrupted_model_path_load = f"{MODEL_SAVE_PATH}_interrupted.zip"
        error_model_path_load = f"{MODEL_SAVE_PATH}_error.zip"
        if os.path.exists(interrupted_model_path_load): eval_model_path = interrupted_model_path_load
        elif os.path.exists(error_model_path_load): eval_model_path = error_model_path_load
        if eval_model_path: print(f"Using interrupted/error model: {eval_model_path}")
        else: print("No saved model found for evaluation.")

    if eval_model_path and eval_env_instance is not None:
        try:
            print(f"Loading model from {eval_model_path} for evaluation...")
            temp_eval_vec_env = DummyVecEnv([lambda: eval_env_instance])

            eval_model = PPO.load(eval_model_path, env=temp_eval_vec_env, device=device)
            print("Model loaded. Running final evaluation...")

            total_eval_episodes_final = 100
            obs = eval_model.env.reset()
            episodes_done = 0
            cumulative_rewards = []
            actions_taken = {0: 0, 1: 0}
            true_positives, false_positives, true_negatives, false_negatives = 0, 0, 0, 0
            total_steps_eval = 0

            while episodes_done < total_eval_episodes_final:
                action, _states = eval_model.predict(obs, deterministic=True)
                obs, reward, done, infos = eval_model.env.step(action)

                info = infos[0]
                actions_taken[action.item()] += 1
                total_steps_eval += 1

                true_label = info.get("true_label_for_reward", None)
                pred_action = info.get("action_taken", None)
                if true_label is not None and pred_action is not None:
                    if pred_action == 1 and true_label == 1: true_positives += 1
                    elif pred_action == 1 and true_label == 0: false_positives += 1
                    elif pred_action == 0 and true_label == 0: true_negatives += 1
                    elif pred_action == 0 and true_label == 1: false_negatives += 1

                if done:
                    episodes_done += 1
                    ep_rew = info.get('episode', {}).get('r')
                    if ep_rew is not None:
                        cumulative_rewards.append(ep_rew)
                        print(f"  Eval Episode {episodes_done}/{total_eval_episodes_final} | Reward: {ep_rew:.2f}", end='\r')

            print("\n\n--- Final Evaluation Results ---")
            if cumulative_rewards:
                 avg_reward = np.mean(cumulative_rewards); std_reward = np.std(cumulative_rewards)
                 print(f"  Avg Reward ({total_eval_episodes_final} eps): {avg_reward:.4f} +/- {std_reward:.4f}")
            else: print("  No complete episodes during evaluation.")
            print(f"  Total Timesteps Evaluated: {total_steps_eval}")
            print(f"  Action Distribution: Legit: {actions_taken[0]}, Scam: {actions_taken[1]}")
            total_predictions = true_positives + false_positives + true_negatives + false_negatives
            if total_predictions > 0:
                accuracy = (true_positives + true_negatives) / total_predictions
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
                print(f"\n  Classification Metrics (per step):")
                print(f"    Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}, Specificity: {specificity:.4f}")
                print(f"    Conf Matrix (Pred\\True): TN={true_negatives}, FP={false_positives}, FN={false_negatives}, TP={true_positives}")
            else: print("  Could not calculate classification metrics.")

        except Exception as e:
            print(f"An error occurred during final evaluation: {e}")
            import traceback; traceback.print_exc()
    else:
        print("Skipping post-training evaluation (no model or eval env).")

    print("\nClosing environments...")
    try:
        if 'train_env' in locals() and train_env is not None:
            train_env.close()
    except Exception as e:
        print(f"Error closing environments: {e}")

    print("\n--- Script Finished ---")