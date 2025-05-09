from tensorflow.keras.utils import Sequence

class CyclicBatchGenerator(Sequence):
    def __init__(self, data, model, target_column, batch_size=32768, is_train=True):
        self.data = data
        self.model = model
        self.batch_size = batch_size
        self.is_train = is_train
        self.target_column = target_column
        self.indices = np.arange(len(data))
        
    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))
    
    def __getitem__(self, index):
        batch_idx = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_data = self.data.iloc[batch_idx]
        
        X = self.model.preprocess_data(batch_data, is_train=self.is_train)
        y = np.column_stack([
            batch_data[f'{self.target_column}_sin'].values,
            batch_data[f'{self.target_column}_cos'].values
        ])
        return X, y


def train_model(self, train_data, target_column, epochs=100, batch_size=32768, validation_split=0.05):
    """Train with generators for large datasets"""
    
    split_idx = int(len(train_data) * (1 - validation_split))
    val_data = train_data.iloc[split_idx:]
    train_data = train_data.iloc[:split_idx]
    
    print("Training Shape:", train_data.shape, train_data["click_date"].min(), train_data["click_date"].max())
    print("Validation Shape:", val_data.shape, val_data["click_date"].min(), val_data["click_date"].max())

    # Shuffle training indices for generator
    train_gen = CyclicBatchGenerator(train_data, self, target_column, batch_size=batch_size, is_train=True)
    val_gen = CyclicBatchGenerator(val_data, self, target_column, batch_size=batch_size, is_train=False)
    
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
    ]
    
    history = self.model.fit(
        x=train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2
    )
    
    self._plot_training(history)
    self.model.save(self.model_name)
    return history
