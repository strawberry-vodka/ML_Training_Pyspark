def train_circular_lgbm(batch_paths):
    params = {
        'objective': 'custom',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }
    
    model = None
    
    for i, path in enumerate(batch_paths):
        print(f"Processing batch {i+1}/{len(batch_paths)}")
        df = pd.read_parquet(path)
        X, y_sin, y_cos = prepare_data(df)
        
        # Combine targets into 2D array
        y = np.column_stack((y_sin, y_cos))
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train.reshape(-1))
        val_data = lgb.Dataset(X_val, label=y_val.reshape(-1), reference=train_data)
        
        if model is None:
            # First batch
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                early_stopping_rounds=50,
                fobj=lgb_circular_loss,
                feval=lambda p, d: ('circular_loss', circular_loss(p.reshape(-1, 2), d.get_label().reshape(-1, 2)), False),
                callbacks=[lgb.log_evaluation(50)]
            )
        else:
            # Subsequent batches
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=1000,
                early_stopping_rounds=50,
                init_model=model,
                fobj=lgb_circular_loss,
                feval=lambda p, d: ('circular_loss', circular_loss(p.reshape(-1, 2), d.get_label().reshape(-1, 2)), False),
                callbacks=[lgb.log_evaluation(50)],
                keep_training_booster=True
            )
    
    return model
