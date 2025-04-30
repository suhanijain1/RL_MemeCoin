
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import traceback
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

try:
    from env_timeseries_v4 import ScamTokenTimeseriesEnvV4 as TokenEnv
except ImportError:
    print("Error: Could not import the Environment V4.")
    print("Make sure 'env_timeseries_v4.py' exists.")
    exit()


MODEL_RUN_TIMESTAMP = "20250501-032010"
BEST_MODEL_DIR = f"./models_timeseries_v4/best_model_ppo_scam_detector_v4_{MODEL_RUN_TIMESTAMP}/"
BEST_MODEL_PATH = os.path.join(BEST_MODEL_DIR, "best_model.zip")
FINAL_MODEL_PATH = f"./models_timeseries_v4/ppo_scam_detector_v4_{MODEL_RUN_TIMESTAMP}_final.zip"

if os.path.exists(BEST_MODEL_PATH): MODEL_PATH = BEST_MODEL_PATH; print(f"Using best model: {MODEL_PATH}")
elif os.path.exists(FINAL_MODEL_PATH): MODEL_PATH = FINAL_MODEL_PATH; print(f"Using final model: {MODEL_PATH}")
else:
    print(f"Error: No V4 model found for timestamp {MODEL_RUN_TIMESTAMP}"); exit()

TEST_DATA_FILE = 'test_data_timeseries.parquet'

FEATURE_COLS = [
    'liquidity_lock', 'top5_holder_pct', 'top10_holder_pct',
    'volume_spike_ratio', 'comment_velocity', 'comment_spam_ratio',
    'hype_score', 'bundled_rug_selloff', 'dev_wallet_reputation',
    'wallet_clustering_hhi'
]

TRAIN_MEAN_SCALED_FILE = 'train_mean_timeseries_scaled01.csv'
TRAIN_STD_SCALED_FILE = 'train_std_timeseries_scaled01.csv'

EVAL_OUTPUT_DIR = f"./evaluation_results_v4_{MODEL_RUN_TIMESTAMP}/"
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)


def run_evaluation_v4(eval_model, eval_env_monitored, token_ids_to_eval: list):
    num_tokens = len(token_ids_to_eval)
    print(f"\nEvaluating V4 model specifically for {num_tokens} unique tokens...")

    episode_rewards = []
    final_predictions = []
    final_true_labels = []
    final_actions = []
    episode_lengths = []
    episodes_done = 0

    try:
        from tqdm import tqdm
        token_iterator = tqdm(token_ids_to_eval, desc="Evaluating Tokens")
    except ImportError:
        token_iterator = token_ids_to_eval
        print("Install tqdm for progress bar.")

    for token_id in token_iterator:
        try:
            obs, info = eval_env_monitored.reset(options={'token_id': token_id})
            if info.get('token_id') != token_id:
                 print(f"\nWarning: Env reset to '{info.get('token_id')}' instead of requested '{token_id}'. Skipping.")
                 continue
        except Exception as reset_err:
             print(f"\nError resetting environment for token {token_id}: {reset_err}. Skipping.")
             continue

        done = False
        while not done:
            action, _ = eval_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env_monitored.step(action)
            done = terminated or truncated

        ep_info = info.get('episode', {})
        ep_rew = ep_info.get('r')
        ep_len = ep_info.get('l')

        if ep_rew is None or ep_len is None:
             print(f"\nWarning: Missing episode reward/length from Monitor for token {token_id}. Last info: {info}. Skipping metrics for this token.")
             episodes_done += 1
             continue

        true_label = info.get('true_token_label')

        last_action = -1
        trunc_penalty = eval_env_monitored.env.truncation_penalty
        if ep_rew == 1.0 or ep_rew == -1.0:
            if (ep_rew == 1.0 and true_label == 0) or (ep_rew == -1.0 and true_label == 1): last_action = 1
            elif (ep_rew == 1.0 and true_label == 1) or (ep_rew == -1.0 and true_label == 0): last_action = 2
        elif np.isclose(ep_rew, trunc_penalty):
            last_action = 0
        else:
            print(f"Warning: Unexpected reward {ep_rew} (TrueLabel:{true_label}, TruncPenalty:{trunc_penalty}) for token {token_id}. Cannot determine last action.")

        episode_rewards.append(ep_rew)
        episode_lengths.append(ep_len)
        final_actions.append(last_action)

        if last_action == 1 or last_action == 2:
            predicted_label = 0 if last_action == 1 else 1
            final_predictions.append(predicted_label)
            if true_label is not None:
                 final_true_labels.append(true_label)
            else:
                 print(f"Warning: Missing true label for classified token {token_id}. Classification metrics might be affected.")

        episodes_done += 1

    if isinstance(token_iterator, tqdm): token_iterator.close()

    print(f"\nEvaluation finished. Processed {episodes_done}/{num_tokens} requested tokens.")

    metrics = {
        "num_episodes": episodes_done,
        "mean_episode_reward": np.mean(episode_rewards) if episode_rewards else 0,
        "std_episode_reward": np.std(episode_rewards) if episode_rewards else 0,
        "mean_episode_length": np.mean(episode_lengths) if episode_lengths else 0,
        "std_episode_length": np.std(episode_lengths) if episode_lengths else 0,
        "final_actions_raw": final_actions,
        "accuracy_classified": None,
        "classification_report_classified": None,
        "confusion_matrix_classified": None,
        "num_classified_episodes": len(final_predictions),
        "num_truncated_episodes": final_actions.count(0)
    }

    if final_predictions and final_true_labels and len(final_predictions) == len(final_true_labels):
        metrics["accuracy_classified"] = accuracy_score(final_true_labels, final_predictions)
        unique_labels = np.unique(final_true_labels + final_predictions)
        target_names = ['Legit (0)', 'Scam (1)']
        labels_to_report = [l for l in [0, 1] if l in unique_labels]
        if not labels_to_report:
             metrics["classification_report_classified"] = "No valid classified labels found."
             metrics["confusion_matrix_classified"] = np.zeros((2,2), dtype=int)
        else:
             metrics["classification_report_classified"] = classification_report(
                 final_true_labels, final_predictions, target_names=[target_names[i] for i in labels_to_report],
                 zero_division=0, labels=labels_to_report
             )
             metrics["confusion_matrix_classified"] = confusion_matrix(final_true_labels, final_predictions, labels=[0, 1])

    elif not final_predictions:
         metrics["classification_report_classified"] = "No episodes were classified."
         metrics["confusion_matrix_classified"] = np.zeros((2,2), dtype=int)
    else:
         metrics["classification_report_classified"] = f"Label mismatch: {len(final_predictions)} preds vs {len(final_true_labels)} true."
         metrics["confusion_matrix_classified"] = np.zeros((2,2), dtype=int)

    return metrics


if __name__ == "__main__":

    print("--- Starting Evaluation for Classifier Agent (V4 - Agent Decides Termination) ---")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Test Data: {TEST_DATA_FILE}")
    print(f"Output Dir: {EVAL_OUTPUT_DIR}")

    try:
        train_mean_scaled_df = pd.read_csv(TRAIN_MEAN_SCALED_FILE, index_col=0)
        train_std_scaled_df = pd.read_csv(TRAIN_STD_SCALED_FILE, index_col=0)
        if len(train_mean_scaled_df.columns) == 1: train_mean_scaled = train_mean_scaled_df.iloc[:, 0]
        else: raise ValueError(f"Expected 1 data column in {TRAIN_MEAN_SCALED_FILE}")
        if len(train_std_scaled_df.columns) == 1: train_std_scaled = train_std_scaled_df.iloc[:, 0]
        else: raise ValueError(f"Expected 1 data column in {TRAIN_STD_SCALED_FILE}")
        train_mean_scaled = train_mean_scaled.reindex(FEATURE_COLS)
        train_std_scaled = train_std_scaled.reindex(FEATURE_COLS).replace(0, 1e-6)
        if train_mean_scaled.isnull().any() or train_std_scaled.isnull().any(): raise ValueError("NaN in norm stats after reindex.")
        print("Loaded normalization stats.")
    except Exception as e:
        print(f"Error loading normalization stats: {e}"); exit()

    print("Creating evaluation environment (V4)...")
    eval_env_instance = None
    eval_env_monitored = None
    all_token_ids = []
    try:
        eval_env_instance = TokenEnv(
            parquet_file=TEST_DATA_FILE,
            feature_cols=FEATURE_COLS,
            train_mean_scaled=train_mean_scaled,
            train_std_scaled=train_std_scaled,
        )
        all_token_ids = list(eval_env_instance.token_ids)
        if not all_token_ids:
            raise ValueError(f"No token IDs found in the environment created from {TEST_DATA_FILE}")

        eval_env_monitored = Monitor(eval_env_instance, allow_early_resets=True)
        print(f"Evaluation environment (V4) created for {len(all_token_ids)} unique tokens and wrapped in Monitor.")

    except Exception as e:
        print(f"Error creating evaluation environment or getting token IDs: {e}");
        traceback.print_exc();
        if eval_env_instance: eval_env_instance.close()
        exit()

    print("Loading trained model...")
    if not os.path.exists(MODEL_PATH): print(f"Error: Model file not found at {MODEL_PATH}"); exit()
    model = None
    try:
        model_class = PPO; print("Assuming PPO model type.")
        model = model_class.load(MODEL_PATH, env=None, device='auto')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}"); traceback.print_exc(); exit()


    eval_metrics = {}
    try:
        eval_metrics = run_evaluation_v4(model, eval_env_monitored, token_ids_to_eval=all_token_ids)
    except Exception as eval_err:
        print(f"\nError occurred during run_evaluation_v4: {eval_err}")
        traceback.print_exc();
    finally:
        print("\nClosing evaluation environment...")
        if eval_env_monitored is not None:
            eval_env_monitored.close()
        elif eval_env_instance is not None:
            eval_env_instance.close()

    if not eval_metrics:
        print("\nEvaluation did not complete successfully. Cannot print metrics.")
    else:
        print("\n--- Evaluation Overall Metrics (V4 Env - All Tokens) ---")
        print(f"Total Tokens Processed: {eval_metrics['num_episodes']}")
        print(f"Average Episode Reward: {eval_metrics['mean_episode_reward']:.4f} +/- {eval_metrics['std_episode_reward']:.4f}")
        print(f"Average Episode Length: {eval_metrics['mean_episode_length']:.2f} +/- {eval_metrics['std_episode_length']:.2f} steps")
        total_ended = eval_metrics['num_classified_episodes'] + eval_metrics['num_truncated_episodes']
        if total_ended > 0:
            trunc_pct = (eval_metrics['num_truncated_episodes'] / total_ended) * 100
            class_pct = (eval_metrics['num_classified_episodes'] / total_ended) * 100
            print(f"Tokens Classified (Act 1/2): {eval_metrics['num_classified_episodes']} ({class_pct:.1f}%)")
            print(f"Tokens Truncated (Act 0): {eval_metrics['num_truncated_episodes']} ({trunc_pct:.1f}%)")

        print("\n--- Evaluation Classification Metrics (Classified Tokens Only) ---")
        if eval_metrics["accuracy_classified"] is not None:
            print(f"Accuracy (Classified only): {eval_metrics['accuracy_classified']:.4f}")
            print("\nClassification Report (Classified only):")
            print(eval_metrics["classification_report_classified"])
            print("\nConfusion Matrix (Classified only - Pred\\True):")
            print(eval_metrics["confusion_matrix_classified"])
        else:
            print(eval_metrics.get("classification_report_classified", "Classification metrics could not be calculated."))


        if eval_metrics["confusion_matrix_classified"] is not None and eval_metrics['num_classified_episodes'] > 0:
            print("\n--- Generating Confusion Matrix Plot (Classified Tokens) ---")
            try:
                cm = eval_metrics["confusion_matrix_classified"]
                display_labels = ['Legit (0)', 'Scam (1)']
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
                fig, ax = plt.subplots(figsize=(6, 5))
                disp.plot(cmap=plt.cm.Blues, values_format='d', ax=ax)
                plt.title(f'Confusion Matrix ({eval_metrics["num_classified_episodes"]} Classified Tokens) - V4 Eval')
                plt.tight_layout()
                plot_filename = os.path.join(EVAL_OUTPUT_DIR, f"evaluation_confusion_matrix_classified.png")
                plt.savefig(plot_filename)
                print(f"Confusion matrix plot saved to {plot_filename}")
                plt.close(fig)
            except Exception as plot_e:
                print(f"Error plotting confusion matrix: {plot_e}")
        elif eval_metrics.get('num_classified_episodes', 0) == 0:
             print("\nSkipping confusion matrix plot (no tokens were classified).")


        if eval_metrics.get("final_actions_raw"):
             print("\n--- Generating Final Action Distribution Plot ---")
             try:
                 plt.figure(figsize=(8, 5))
                 action_map = {0: 'Truncated', 1: 'Classified Legit', 2: 'Classified Scam', -1: 'Unknown'}
                 action_labels = [action_map.get(a, 'Unknown') for a in eval_metrics["final_actions_raw"]]
                 sns.countplot(x=action_labels, order=[action_map[i] for i in [0, 1, 2, -1] if action_map[i] in action_labels])

                 plt.title('Distribution of Final Actions per Token')
                 plt.xlabel('Final Action')
                 plt.ylabel('Frequency')
                 plt.xticks(rotation=15, ha='right')
                 plt.grid(axis='y', alpha=0.5)
                 plt.tight_layout()
                 hist_filename = os.path.join(EVAL_OUTPUT_DIR, f"evaluation_final_action_distribution.png")
                 plt.savefig(hist_filename)
                 print(f"Final action distribution plot saved to {hist_filename}")
                 plt.close()
             except Exception as plot_e:
                 print(f"Error plotting final action distribution: {plot_e}")


    print("\nEvaluation script (V4 - All Tokens) finished.")
    print(f"Results and plots saved in: {EVAL_OUTPUT_DIR}")

