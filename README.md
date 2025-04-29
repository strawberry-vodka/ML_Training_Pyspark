def lgb_circular_loss(preds, train_data):
    """Fixed LightGBM custom loss function"""
    # Reshape predictions to (n_samples, 2)
    preds = preds.reshape(-1, 2)
    y_true = train_data.get_label().reshape(-1, 2)
    
    # Calculate angles
    pred_angles = np.arctan2(preds[:, 0], preds[:, 1])
    true_angles = np.arctan2(y_true[:, 0], y_true[:, 1])
    
    # Circular difference
    diff = pred_angles - true_angles
    circular_diff = np.minimum(np.abs(diff), 2*np.pi - np.abs(diff))
    
    # Gradients
    grad_sin = 2 * circular_diff * (-np.sin(pred_angles))
    grad_cos = 2 * circular_diff * np.cos(pred_angles))
    
    # Hessians (simplified)
    hess_sin = 2 * np.ones_like(grad_sin)
    hess_cos = 2 * np.ones_like(grad_cos)
    
    # Concatenate and flatten
    grad = np.column_stack((grad_sin, grad_cos)).ravel()
    hess = np.column_stack((hess_sin, hess_cos)).ravel()
    
    return grad, hess


def train_with_validation(batch_paths):
    params = {
        'objective': 'custom',
        'verbosity': -1,
        'seed': 42,
        'num_leaves': 31,
        'feature_fraction': 0.9
    }
    
    model = None
    
    for i, path in enumerate(batch_paths):
        print(f"Processing batch {i+1}/{len(batch_paths)}")
        df = pd.read_parquet(path)
        X, y_sin, y_cos = prepare_data(df)
        y = np.column_stack((y_sin, y_cos))
        
        # Temporal split (newest 20% as validation)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, shuffle=False  # Critical: no shuffle for time series
        )
        
        train_data = lgb.Dataset(
            X_train, 
            label=y_train.ravel(),  # Flatten to (n_samples*2,)
            free_raw_data=False
        )
        val_data = lgb.Dataset(
            X_val,
            label=y_val.ravel(),
            reference=train_data
        )
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            early_stopping_rounds=50,
            fobj=lgb_circular_loss,
            callbacks=[
                lgb.log_evaluation(50),
                lgb.record_evaluation({})
            ],
            init_model=model,
            keep_training_booster=True
        )
        
        # Calculate validation metrics
        val_hours = X_val['click_hour']
        pred_hours = predict_hour(model, X_val.drop(columns='click_hour'))
        accuracy = cyclic_accuracy(val_hours, pred_hours)
        print(f"Batch {i+1} Validation Accuracy (±1h): {accuracy:.2%}")
    
    return model
