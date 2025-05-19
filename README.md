import numpy as np
import pandas as pd

def match_distribution_with_angles(pred_angles, hour_distribution):
    # Sort predicted angles from low to high
    sorted_indices = np.argsort(pred_angles)
    sorted_angles = pred_angles[sorted_indices]

    # Compute how many samples should be assigned to each hour based on distribution
    total = len(pred_angles)
    hour_bins = (hour_distribution * total).round().astype(int)

    # Ensure total count matches (rounding errors)
    diff = total - hour_bins.sum()
    if diff != 0:
        # Adjust the hour with the largest frequency
        max_hour = hour_bins.idxmax()
        hour_bins[max_hour] += diff

    # Assign hours according to bin sizes
    matched_hours = np.empty_like(pred_angles, dtype=int)
    start = 0
    for hour in sorted(hour_bins.index):
        count = hour_bins[hour]
        matched_hours[sorted_indices[start:start+count]] = hour
        start += count

    return matched_hours


# Your predicted degrees array
pred_angles = model_output_degrees  # shape (n_samples,), range [-180, 180]

# Your training hour distribution
hour_distribution = train_df["click_hour"].value_counts(normalize=True).sort_index()

# Get mapped click hours
final_click_hour_preds = match_distribution_with_angles(pred_angles, hour_distribution)
