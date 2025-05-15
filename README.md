def fast_quantile_match(pred_hours, true_bins):
    """
    Adjust predictions using precomputed true_bins.
    
    Args:
        pred_hours: Array of predicted hours (0-23)
        true_bins: Precomputed quantile bins from true distribution
    
    Returns:
        Adjusted hours (0-23)
    """
    # Compute percentiles of predictions in their own distribution
    pred_percentiles = np.array([
        np.sum(pred_hours <= x) / len(pred_hours) * 100  # Faster than percentileofscore
        for x in pred_hours
    ])
    
    # Map to nearest true bin (vectorized)
    bin_indices = np.clip(
        (pred_percentiles / 100 * (len(true_bins) - 1)).astype(int),
        0, len(true_bins) - 1
    )
    
    return np.round(true_bins[bin_indices]) % 24
