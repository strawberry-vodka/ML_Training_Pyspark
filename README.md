def train_lightgbm(df_list):
    # Initialize model parameters
    params = {
        'objective': 'multiclass',
        'num_class': 24,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'num_threads': 4,
        'max_depth': -1,
        'min_data_in_leaf': 20
    }
    
    models = []
    eval_results = []
    
    for i, df in enumerate(df_list):
        print(f"Training on batch {i+1}/{len(df_list)}")
        
        # Split features and target
        X = df.drop(['user_id', 'click_timestamp', 'click_hour'], axis=1)
        y = df['click_hour']
        
        # Train/test split for this batch
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model on this batch
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            early_stopping_rounds=50,
            verbose_eval=50
        )
        
        models.append(model)
        eval_results.append(model.best_score)
    
    return models, eval_results

# Example usage:
# models, eval_results = train_lightgbm(df_list)
