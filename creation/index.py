import numpy as np
import pandas as pd

np.random.seed(42)

def generate_synthetic_data(n_samples=1_000_000):
    # --------------------
    # Generate Features
    # --------------------

    liquidity_lock = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])

    top5_holder_pct = np.where(
        liquidity_lock == 0,  # If liquidity unlocked → higher concentration
        np.random.beta(5, 2, size=n_samples),  # Scam-prone
        np.random.beta(2, 5, size=n_samples)   # Safer
    )

    top10_holder_pct = np.where(
        liquidity_lock == 0,
        np.random.beta(4, 2, size=n_samples),
        np.random.beta(2, 4, size=n_samples)
    )

    volume_spike_ratio = np.where(
        liquidity_lock == 0,
        np.random.pareto(3, size=n_samples) + 1,
        np.random.normal(1.2, 0.2, size=n_samples)
    )

    liquidity_drain_pct = np.where(
        liquidity_lock == 0,
        np.random.choice([0, 0.95, 1.0], size=n_samples, p=[0.7, 0.2, 0.1]),
        np.zeros(n_samples)
    )

    comment_velocity = np.where(
        liquidity_lock == 0,
        np.random.poisson(10, size=n_samples),
        np.random.poisson(2, size=n_samples)
    )

    comment_spam_ratio = np.where(
        liquidity_lock == 0,
        np.random.beta(7, 3, size=n_samples),
        np.random.beta(2, 8, size=n_samples)
    )

    hype_score = np.where(
        liquidity_lock == 0,
        np.random.beta(5, 1, size=n_samples),
        np.random.beta(2, 3, size=n_samples)
    )

    price_spike_magnitude = np.where(
        liquidity_lock == 0,
        np.random.lognormal(1.5, 0.5, size=n_samples),
        np.random.lognormal(0.5, 0.3, size=n_samples)
    )

    price_crash_depth = np.where(
        liquidity_lock == 0,
        np.random.uniform(0.9, 1.0, size=n_samples),
        np.random.uniform(0.1, 0.5, size=n_samples)
    )

    time_to_crash_min = np.where(
        liquidity_lock == 0,
        np.random.exponential(30, size=n_samples),
        np.random.exponential(600, size=n_samples)
    )

    bundled_rug_selloff = np.where(
        liquidity_lock == 0,
        np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5]),
        np.zeros(n_samples)
    )

    deployment_batch_size = np.where(
        liquidity_lock == 0,
        np.random.choice([1, 3, 5], size=n_samples, p=[0.8, 0.15, 0.05]),
        np.ones(n_samples)
    )

    dev_wallet_reputation = np.where(
        liquidity_lock == 0,
        np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4]),
        np.zeros(n_samples)
    )

    wallet_clustering_hhi = np.where(
        liquidity_lock == 0,
        np.random.beta(5, 2, size=n_samples),
        np.random.beta(2, 5, size=n_samples)
    )

    # --------------------
    # Probabilistic Scam Labeling
    # --------------------

    scam_probability = (
        (liquidity_lock == 0) * 0.5 + 
        (top10_holder_pct > 0.7) * 0.3 + 
        (volume_spike_ratio > 3) * 0.2 + 
        (liquidity_drain_pct >= 0.95) * 0.3 + 
        (comment_spam_ratio > 0.5) * 0.1 + 
        (price_crash_depth > 0.9) * 0.2 + 
        (dev_wallet_reputation == 1) * 0.4 +
        (wallet_clustering_hhi > 0.7) * 0.2
    )

    scam_probability = np.clip(scam_probability, 0, 1)

    labels = np.random.binomial(1, scam_probability)

    # --------------------
    # Combine All Features
    # --------------------

    df = pd.DataFrame({
        'liquidity_lock': liquidity_lock,
        'top5_holder_pct': top5_holder_pct,
        'top10_holder_pct': top10_holder_pct,
        'volume_spike_ratio': volume_spike_ratio,
        'liquidity_drain_pct': liquidity_drain_pct,
        'comment_velocity': comment_velocity,
        'comment_spam_ratio': comment_spam_ratio,
        'hype_score': hype_score,
        'price_spike_magnitude': price_spike_magnitude,
        'price_crash_depth': price_crash_depth,
        'time_to_crash_min': time_to_crash_min,
        'bundled_rug_selloff': bundled_rug_selloff,
        'deployment_batch_size': deployment_batch_size,
        'dev_wallet_reputation': dev_wallet_reputation,
        'wallet_clustering_hhi': wallet_clustering_hhi,
        'label': labels
    })

    return df

# Generate dataset
df = generate_synthetic_data(1_000_000)
df.to_parquet('synthetic_scam_token_dataset.parquet')

print(df['label'].value_counts(normalize=True))  # Check scam/legit ratio
