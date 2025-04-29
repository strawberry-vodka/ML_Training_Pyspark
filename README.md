def prepare_data(df, target_col='click_hour'):
    """Safe feature/target separation"""
    df[target_col] = df['click_timestamp'].dt.hour

    # Drop unused columns
    features = df.drop(['user_id', 'click_timestamp', target_col], axis=1)

    # Target to radians
    hour_rad = df[target_col] * (2 * np.pi / 24)
    y_sin = np.sin(hour_rad)
    y_cos = np.cos(hour_rad)

    return features, y_sin, y_cos

def temporal_train_test_split(X, y, test_size=0.2):
    """Time-based split"""
    split_idx = int(len(X) * (1 - test_size))
    return X.iloc[:split_idx], X.iloc[split_idx:], y[:split_idx], y[split_idx:]

def predict_hour(pred_sin, pred_cos):
    """Convert sin/cos predictions to hour"""
    norms = np.sqrt(pred_sin**2 + pred_cos**2)
    valid = norms > 1e-6
    hours = np.zeros(len(pred_sin))
    hours[valid] = (np.arctan2(pred_sin[valid], pred_cos[valid]) * 24 / (2 * np.pi)) % 24
    return np.round(hours).astype(int)

def train_single_target(batch_paths, target='sin'):
    model = None
    metrics = []

    params = {
        'objective': 'regression',
        'verbosity': -1,
        'seed': 42,
        'num_leaves': 31,
        'feature_fraction': 0.9,
        'learning_rate': 0.05
    }

    for i, path in enumerate(batch_paths):
        df = pd.read_parquet(path).sort_values('click_timestamp')
        X, y_sin, y_cos = prepare_data(df)
        y = y_sin if target == 'sin' else y_cos

        X_train, X_val, y_train, y_val = temporal_train_test_split(X, y)

        train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            early_stopping_rounds=50,
            init_model=model,
            keep_training_booster=True,
            callbacks=[
                lgb.log_evaluation(50),
                lgb.reset_parameter(
                    learning_rate=lambda x: max(0.01, 0.05 * (0.9 ** x))
                )
            ]
        )

        # For sin/cos separately, no accuracy calculation here
        metrics.append(model.best_score['valid_0']['l2'])

    return model, metrics

def train_combined_and_evaluate(batch_paths):
    model_sin, _ = train_single_target(batch_paths, target='sin')
    model_cos, _ = train_single_target(batch_paths, target='cos')

    df = pd.concat([pd.read_parquet(p) for p in batch_paths]).sort_values('click_timestamp')
    X, y_sin, y_cos = prepare_data(df)
    y_true_hours = df['click_hour']

    pred_sin = model_sin.predict(X)
    pred_cos = model_cos.predict(X)

    pred_hours = predict_hour(pred_sin, pred_cos)
    exact_match = np.mean(pred_hours == y_true_hours)
    plus_minus_1 = np.mean(np.abs((pred_hours - y_true_hours + 12) % 24 - 12) <= 1)

    print(f"Exact Match Accuracy: {exact_match:.2%}")
    print(f"±1 Hour Accuracy: {plus_minus_1:.2%}")

    return model_sin, model_cos
