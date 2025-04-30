import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

try:
    from env_timeseries_v2 import ScamTokenTimeseriesEnvV2 as TokenEnv
except ImportError:
    print("Error: Could not import the Environment.")
    print("Make sure 'env_timeseries_v2.py' exists and contains 'ScamTokenTimeseriesEnvV2'.")
    exit()

MODEL_RUN_TIMESTAMP = "20250430-224733"
BEST_MODEL_DIR = f"./models_timeseries/best_model_ppo_scam_detector_v2_{MODEL_RUN_TIMESTAMP}/"
BEST_MODEL_PATH = os.path.join(BEST_MODEL_DIR, "best_model.zip")
FINAL_MODEL_PATH = f"./models_timeseries/ppo_scam_detector_v2_{MODEL_RUN_TIMESTAMP}_final.zip"

if os.path.exists(BEST_MODEL_PATH):
    MODEL_PATH = BEST_MODEL_PATH
    print(f"Using best model: {MODEL_PATH}")
elif os.path.exists(FINAL_MODEL_PATH):
     MODEL_PATH = FINAL_MODEL_PATH
     print(f"Using final model: {MODEL_PATH}")
else:
    print(f"Error: Neither best model nor final model found for timestamp {MODEL_RUN_TIMESTAMP}")
    print(f"Checked: {BEST_MODEL_PATH}")
    print(f"Checked: {FINAL_MODEL_PATH}")
    exit()

TEST_DATA_FILE = 'test_data_timeseries.parquet'

FEATURE_COLS = [
    'liquidity_lock', 'top5_holder_pct', 'top10_holder_pct',
    'volume_spike_ratio', 'comment_velocity', 'comment_spam_ratio',
    'hype_score', 'bundled_rug_selloff', 'dev_wallet_reputation',
    'wallet_clustering_hhi'
]

TRAIN_MEAN_SCALED_FILE = 'train_mean_timeseries_scaled01.csv'
TRAIN_STD_SCALED_FILE = 'train_std_timeseries_scaled01.csv'

EVAL_OUTPUT_DIR = f"./evaluation_results_{MODEL_RUN_TIMESTAMP}/"
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

def run_evaluation(eval_model, eval_env, num_tokens_to_eval=None):
    all_true_labels = []
    all_predicted_labels = []
    total_steps = 0
    total_raw_reward = 0.0
    total_reward = 0.0

    token_ids_to_eval = eval_env.token_ids
    if num_tokens_to_eval is not None and num_tokens_to_eval < len(token_ids_to_eval):
        token_ids_to_eval = np.random.choice(token_ids_to_eval, num_tokens_to_eval, replace=False)

    num_tokens = len(token_ids_to_eval)
    print(f"\nEvaluating on {num_tokens} tokens...")

    try:
        from tqdm import tqdm
        token_iterator = tqdm(token_ids_to_eval, desc="Evaluating Tokens")
    except ImportError:
        token_iterator = token_ids_to_eval
        print("Install tqdm (`pip install tqdm`) for a progress bar.")

    for token_id in token_iterator:
        obs, info = eval_env.reset(seed=np.random.randint(0, 10000))
        done = False
        episode_steps = 0

        episode_true = []
        episode_pred = []

        while not done:
            action, _ = eval_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated

            true_label = info.get('true_label_for_reward')
            if true_label is not None:
                episode_true.append(true_label)
                episode_pred.append(action.item())

            total_steps += 1
            total_raw_reward += info.get('raw_reward', reward)
            total_reward += reward
            episode_steps +=1

        if episode_true and episode_pred:
             all_true_labels.extend(episode_true)
             all_predicted_labels.extend(episode_pred)

    print(f"Evaluation finished. Total steps: {total_steps}")

    metrics = {
        "total_steps": total_steps,
        "total_raw_reward": total_raw_reward,
        "total_reward": total_reward,
        "mean_raw_reward": total_raw_reward / total_steps if total_steps > 0 else 0,
        "mean_reward": total_reward / total_steps if total_steps > 0 else 0,
        "accuracy": None,
        "classification_report": None,
        "confusion_matrix": None
    }

    if not all_true_labels or not all_predicted_labels:
        print("Warning: No labels collected during evaluation. Cannot calculate classification metrics.")
    elif len(all_true_labels) != len(all_predicted_labels):
        print(f"Warning: Mismatch in collected label counts: True={len(all_true_labels)}, Pred={len(all_predicted_labels)}")
        min_len = min(len(all_true_labels), len(all_predicted_labels))
        all_true_labels = all_true_labels[:min_len]
        all_predicted_labels = all_predicted_labels[:min_len]
        if not all_true_labels:
             print("Warning: Still no labels after trimming.")
             return metrics

    if all_true_labels and all_predicted_labels:
        metrics["accuracy"] = accuracy_score(all_true_labels, all_predicted_labels)
        metrics["classification_report"] = classification_report(
            all_true_labels, all_predicted_labels, target_names=['Legit (0)', 'Scam (1)'], zero_division=0
        )
        metrics["confusion_matrix"] = confusion_matrix(all_true_labels, all_predicted_labels)
        if metrics["confusion_matrix"].shape != (2, 2):
             print("Warning: Confusion matrix is not 2x2. Adjusting.")
             cm = metrics["confusion_matrix"]
             labels = np.unique(all_true_labels + all_predicted_labels)
             new_cm = np.zeros((2, 2), dtype=int)
             for i, true_label in enumerate([0, 1]):
                 for j, pred_label in enumerate([0, 1]):
                     if true_label in labels and pred_label in labels:
                          true_idx = np.where(labels == true_label)[0][0]
                          pred_idx = np.where(labels == pred_label)[0][0]
                          if true_idx < cm.shape[0] and pred_idx < cm.shape[1]:
                              new_cm[i, j] = cm[true_idx, pred_idx]
             metrics["confusion_matrix"] = new_cm

    return metrics

if __name__ == "__main__":
    print("--- Starting Evaluation for Classifier Agent ---")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Test Data: {TEST_DATA_FILE}")
    print(f"Output Dir: {EVAL_OUTPUT_DIR}")

    try:
        train_mean_scaled_df = pd.read_csv(TRAIN_MEAN_SCALED_FILE, index_col=0)
        train_std_scaled_df = pd.read_csv(TRAIN_STD_SCALED_FILE, index_col=0)

        if len(train_mean_scaled_df.columns) == 1:
            train_mean_scaled = train_mean_scaled_df.iloc[:, 0]
        else:
            raise ValueError(f"Expected 1 data column in {TRAIN_MEAN_SCALED_FILE}, found {len(train_mean_scaled_df.columns)}")

        if len(train_std_scaled_df.columns) == 1:
            train_std_scaled = train_std_scaled_df.iloc[:, 0]
        else:
            raise ValueError(f"Expected 1 data column in {TRAIN_STD_SCALED_FILE}, found {len(train_std_scaled_df.columns)}")

        print("Loaded normalization stats (mean/std of scaled training data).")
        missing_mean = [col for col in FEATURE_COLS if col not in train_mean_scaled.index]
        missing_std = [col for col in FEATURE_COLS if col not in train_std_scaled.index]
        if missing_mean or missing_std:
             raise ValueError(f"Mismatch between FEATURE_COLS and loaded stats. Missing mean: {missing_mean}, Missing std: {missing_std}")

    except FileNotFoundError as e:
        print(f"Error: Could not load normalization stats files: {e}")
        print("These files are required to initialize the environment.")
        exit()
    except Exception as e:
        print(f"Error processing normalization stats: {e}")
        exit()

    print("Creating evaluation environment...")
    try:
        eval_env = TokenEnv(
            parquet_file=TEST_DATA_FILE,
            feature_cols=FEATURE_COLS,
            train_mean_scaled=train_mean_scaled,
            train_std_scaled=train_std_scaled,
        )
        print("Evaluation environment created.")
    except Exception as e:
        print(f"Error creating evaluation environment: {e}")
        exit()

    print("Loading trained model...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        exit()
    try:
        if "ppo" in MODEL_PATH.lower():
            model_class = PPO
            print("Detected PPO model type.")

        model = model_class.load(MODEL_PATH, env=None)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        exit()

    NUM_EVAL_TOKENS = None

    eval_metrics = run_evaluation(model, eval_env, num_tokens_to_eval=NUM_EVAL_TOKENS)

    print("\n--- Evaluation Classification Metrics ---")
    if eval_metrics["accuracy"] is not None:
        print(f"Overall Accuracy: {eval_metrics['accuracy']:.4f}")
        print("\nClassification Report:")
        print(eval_metrics["classification_report"])
        print("\nConfusion Matrix:")
        print(eval_metrics["confusion_matrix"])
    else:
        print("Classification metrics could not be calculated (likely no labels found).")

    print("\n--- Evaluation Reward Metrics ---")
    print(f"Total Steps Evaluated: {eval_metrics['total_steps']}")
    print(f"Average Raw Reward per Step: {eval_metrics['mean_raw_reward']:.6f}")
    if eval_env.reward_clip_range is not None:
         print(f"Average Clipped Reward per Step: {eval_metrics['mean_reward']:.6f}")

    if eval_metrics["confusion_matrix"] is not None:
        print("\n--- Generating Confusion Matrix Plot ---")
        try:
            cm = eval_metrics["confusion_matrix"]
            display_labels = ['Legit (0)', 'Scam (1)']
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
            fig, ax = plt.subplots(figsize=(6, 5))
            disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax)
            plt.title('Confusion Matrix - Evaluation Run')
            plt.tight_layout()
            plot_filename = os.path.join(EVAL_OUTPUT_DIR, f"evaluation_confusion_matrix.png")
            plt.savefig(plot_filename)
            print(f"Confusion matrix plot saved to {plot_filename}")
            plt.close(fig)
        except Exception as plot_e:
            print(f"Error plotting confusion matrix: {plot_e}")

    print("\nEvaluation script finished.")
    print(f"Results and plots saved in: {EVAL_OUTPUT_DIR}")