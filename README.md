def circular_loss(y_true, y_pred):
    """Custom loss function for circular regression"""
    # Split predictions into sin and cos components
    pred_sin = y_pred[:, 0]
    pred_cos = y_pred[:, 1]
    
    # Convert to angles
    pred_angle = np.arctan2(pred_sin, pred_cos)
    true_angle = np.arctan2(y_true[:, 0], y_true[:, 1])
    
    # Calculate circular distance
    diff = np.abs(pred_angle - true_angle)
    circular_diff = np.minimum(diff, 2*np.pi - diff)
    
    return np.mean(circular_diff**2)  # MSE of circular difference

def lgb_circular_loss(preds, train_data):
    """LightGBM compatible circular loss"""
    y_true = train_data.get_label()
    y_true = y_true.reshape(-1, 2)  # Reshape to (n_samples, 2)
    
    preds = preds.reshape(-1, 2)  # Reshape predictions
    
    grad = np.zeros_like(preds)
    hess = np.zeros_like(preds)
    
    # Calculate gradients and hessians (simplified version)
    pred_angle = np.arctan2(preds[:, 0], preds[:, 1])
    true_angle = np.arctan2(y_true[:, 0], y_true[:, 1])
    
    diff = pred_angle - true_angle
    circular_diff = np.minimum(np.abs(diff), 2*np.pi - np.abs(diff))
    
    grad[:, 0] = 2 * circular_diff * (-np.sin(pred_angle))
    grad[:, 1] = 2 * circular_diff * (np.cos(pred_angle))
    
    hess[:, 0] = 2 * np.sin(pred_angle)**2
    hess[:, 1] = 2 * np.cos(pred_angle)**2
    
    return grad.flatten(), hess.flatten()
