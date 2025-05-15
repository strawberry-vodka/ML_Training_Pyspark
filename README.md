def train_model(self, train_data, target_column, region_column, time_column='click_date', 
               batch_size=32768, epochs=100, validation_split=0.2):
    """Fixed version with sample weights in generator"""
    
    # 1. Calculate sample weights (same as before)
    region_counts = train_data[region_column].value_counts()
    region_weights = 1.0 / region_counts
    sample_weights = train_data[region_column].map(region_weights).values
    sample_weights = sample_weights * (len(train_data) / np.sum(sample_weights)
    
    # 2. Temporal split (fixed version)
    train_data = train_data.sort_values(time_column)
    split_idx = int(len(train_data) * (1 - validation_split))
    train_split = train_data.iloc[:split_idx]
    val_split = train_data.iloc[split_idx:]
    
    # 3. Preprocess and prepare weights
    self.X_train_full = self.preprocess_data(train_split, is_train=True)
    y_train_full = np.column_stack([
        train_split[f'{target_column}_sin'].values,
        train_split[f'{target_column}_cos'].values
    ])
    train_weights = sample_weights[:split_idx]  # Align with train_split
    
    # 4. Modified generator
    def batch_generator():
        while True:
            for i in range(0, len(train_split), batch_size):
                X_batch = [x[i:i+batch_size] for x in self.X_train_full]
                y_batch = y_train_full[i:i+batch_size]
                w_batch = train_weights[i:i+batch_size]
                yield dict(zip(input_names, X_batch)), y_batch, w_batch
    
    # 5. Train with generator
    history = self.model.fit(
        x=batch_generator(),
        steps_per_epoch=int(np.ceil(len(train_split) / batch_size)),
        epochs=epochs,
        validation_data=(X_val_dict, y_val),
        callbacks=[...],
        verbose=2
    )
