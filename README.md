class CyclicOrdinalRegressor:
    # ... [Keep all previous methods unchanged until train_model] ...

    def train_model(self, full_train_data, target_column, time_column='click_date',
                   batch_size=32768, epochs=100, validation_split=0.2):
        """Batch-wise training with temporal validation split"""
        # 1. Sort by time and split
        full_train_data = full_train_data.sort_values(time_column)
        split_idx = int(len(full_train_data) * (1 - validation_split))
        train_data = full_train_data.iloc[:split_idx]
        val_data = full_train_data.iloc[split_idx:]
        
        print(f"Training period: {train_data[time_column].min()} to {train_data[time_column].max()}")
        print(f"Validation period: {val_data[time_column].min()} to {val_data[time_column].max()}")

        # 2. One-time preprocessing (before any batches)
        print("Preprocessing training data...")
        X_train_full = self.preprocess_data(train_data, is_train=True)
        y_train_full = np.column_stack([
            train_data[f'{target_column}_sin'].values,
            train_data[f'{target_column}_cos'].values
        ])
        
        print("Preprocessing validation data...")
        X_val = self.preprocess_data(val_data, is_train=False)
        y_val = np.column_stack([
            val_data[f'{target_column}_sin'].values,
            val_data[f'{target_column}_cos'].values
        ])

        # 3. Convert to dictionary format for multi-input models
        input_names = [layer.name for layer in self.model.inputs]
        X_val_dict = dict(zip(input_names, X_val))
        
        # 4. Create batch generator
        def batch_generator():
            while True:  # Infinite generator for multiple epochs
                # Process in temporal batches
                for i in range(0, len(train_data), batch_size):
                    batch_data = train_data.iloc[i:i+batch_size]
                    
                    # Use preprocessed data directly
                    X_batch = [x[i:i+batch_size] for x in X_train_full]
                    y_batch = y_train_full[i:i+batch_size]
                    
                    yield dict(zip(input_names, X_batch)), y_batch

        # 5. Calculate steps per epoch
        steps_per_epoch = int(np.ceil(len(train_data) / batch_size))
        
        # 6. Train with batch generator
        history = self.model.fit(
            x=batch_generator(),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=(X_val_dict, y_val),
            callbacks=[
                EarlyStopping(patience=5, restore_best_weights=True),
                ModelCheckpoint('best_model.h5', save_best_only=True)
            ],
            verbose=2
        )
        
        self._plot_training(history)
        self.model.save(self.model_name)
        return history

    # ... [Keep remaining methods unchanged] ...
