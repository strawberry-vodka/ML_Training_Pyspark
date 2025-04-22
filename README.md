import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

# ---------------------------------------
# Input: test_df must have these columns:
# - user_id
# - weekday
# - click_hour (true label: int from 0 to 23)
# - pred_probs (list or np.array of length 24: model's prediction)
# ---------------------------------------

# Step 1: Compute average predicted distribution per (user_id, weekday)

predicted_grouped = (
    test_df.groupby(['user_id', 'weekday'])['pred_probs']
    .apply(lambda x: np.mean(np.stack(x), axis=0))
    .to_dict()
)

# Step 2: Compute empirical distribution (actuals) per (user_id, weekday)

def compute_empirical_dist(group):
    hist = np.zeros(24)
    for hour in group['click_hour']:
        hist[hour] += 1
    return hist / hist.sum()

actual_grouped = (
    test_df.groupby(['user_id', 'weekday'])
    .apply(compute_empirical_dist)
    .to_dict()
)

# Step 3: Compute JS divergence between actual and predicted distributions

def js_divergence(p, q):
    return jensenshannon(p, q, base=2) ** 2

js_scores = []
total_js = 0
count = 0

for key in actual_grouped:
    if key not in predicted_grouped:
        continue

    actual_dist = actual_grouped[key]
    predicted_dist = predicted_grouped[key]

    # Safety normalization
    actual_dist = actual_dist / actual_dist.sum()
    predicted_dist = predicted_dist / predicted_dist.sum()

    js = js_divergence(actual_dist, predicted_dist)

    js_scores.append({
        'user_id': key[0],
        'weekday': key[1],
        'js_divergence': js
    })

    total_js += js
    count += 1

# Step 4: Report

mean_js = total_js / count if count > 0 else None
print(f"Average JS Divergence: {mean_js:.4f}")

js_df = pd.DataFrame(js_scores)
# Optional: js_df.to_csv("js_divergence_breakdown.csv", index=False)
