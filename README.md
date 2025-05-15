def train_model(self, train_data, target_column, region_column, time_column='click_date', batch_size=32768, epochs=100, validation_split=0.2):
        """Batch-wise training with temporal validation split"""

        region_counts = train_data[region_column].value_counts()
        region_weights = 1.0/region_counts
        sample_weights = train_data[region_column].map(region_weights).values

        sample_weights = sample_weights * (len(train_data)/np.sum(sample_weights))
        
        split_idx = int(len(train_data) * (1 - validation_split))
        date_part = train_data.iloc[split_idx][time_column]
        val_data = train_data[train_data[time_column]>=date_part]
        train_data = train_data[train_data[time_column]<date_part]
        
        print(f"Training period: {train_data[time_column].min()} to {train_data[time_column].max()}")
        print(f"Validation period: {val_data[time_column].min()} to {val_data[time_column].max()}")

        # 2. One-time preprocessing (before any batches)
        print("Preprocessing training data...")
        self.X_train_full = self.preprocess_data(train_data, is_train=True)
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
                    X_batch = [x[i:i+batch_size] for x in self.X_train_full]
                    y_batch = y_train_full[i:i+batch_size]
                    
                    yield dict(zip(input_names, X_batch)), y_batch

        # 5. Calculate steps per epoch
        steps_per_epoch = int(np.ceil(len(train_data) / batch_size))

        train_sample_weights = sample_weights[:split_idx]
        
        # 6. Train with batch generator
        history = self.model.fit(
            x=batch_generator(),
            steps_per_epoch=steps_per_epoch,
            sample_weight = train_sample_weights,
            epochs=epochs,
            validation_data=(X_val_dict, y_val),
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ModelCheckpoint('best_model.h5', save_best_only=True)
            ],
            verbose=2
        )
        
        self._plot_training(history)
        self.model.save(self.model_name)
        return history
