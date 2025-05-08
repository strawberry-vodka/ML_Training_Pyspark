from sklearn.preprocessing import LabelEncoder, StandardScaler

class CyclicOrdinalRegressor:
    
    def __init__(self, numerical_cyclical_vars, numerical_vars, categorical_vars, name):
        self.categorical_vars = categorical_vars
        self.numerical_cyclical_vars = numerical_cyclical_vars
        self.numerical_vars = numerical_vars
        self.model_name = name + '.keras'
        self.embedding_layers = []
        self.input_layers = []
        self.embedding_sizes = {}
        self.model = None
        self.label_encoders = {}
        self.max_categorical_values = {}  # NEW: Track max values

    def calculate_embedding_sizes(self, data):
        """Calculate embedding dimensions with buffer"""
        for col in self.categorical_vars:
            # Find maximum value across all data
            self.max_categorical_values[col] = int(data[col].nunique())  # NEW
            
            # Calculate embedding size with 20% buffer
            unique_vals = len(data[col].unique())
            self.embedding_sizes[col] = max(2, min(50, round(1.6 * np.log2(unique_vals))))
            
            print(f"{col}: Max value={self.max_categorical_values[col]}, Embedding size={self.embedding_sizes[col]}")

    def build_model(self):
        """Build model architecture for cyclic regression"""
        # 1. Categorical inputs
        self._build_categorical_inputs()
        
        # 2. Numerical inputs (already in sin/cos form)
        numerical_cyclical_input = Input(shape=(len(self.numerical_cyclical_vars),), name='numerical_cyclical_input')
        self.input_layers.append(numerical_cyclical_input)

        # 3. Numerical features for non-cyclic-features
        numerical_input = Input(shape=(len(self.numerical_vars),), name='numerical_input')
        self.input_layers.append(numerical__input)
        
        # 4. Concatenate all inputs
        concatenated = Concatenate()(self.embedding_layers + [numerical_cyclical_input])
        
        # 4. Hidden layers
        x = Dense(128, activation='relu')(concatenated)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(32, activation='relu')(x)
        
        # 5. Output layer (2 units for sin/cos)
        output = Dense(2, activation='linear')(x)
        
        self.model = Model(inputs=self.input_layers, outputs=output)
        print(self.model.summary())

    def _build_categorical_inputs(self):
        """Create embedding layers with safe dimensions"""
        for col in self.categorical_vars:
            input_layer = Input(shape=(1,), name=col)
            # NEW: Use max_value + buffer for embedding dimension
            embedding_dim = self.max_categorical_values[col] + 1000  # Safety buffer
            embedding = Embedding(
                input_dim=embedding_dim,  # Now large enough
                output_dim=self.embedding_sizes[col],
                name=f'embed_{col}'
            )(input_layer)
            flattened = Flatten()(embedding)
            
            self.input_layers.append(input_layer)
            self.embedding_layers.append(flattened)
    
    def preprocess_data(self, data, is_train=True):
        """Handle unseen categories safely"""
        X_categorical = []
        
        for col in self.categorical_vars:
            if is_train:
                # NEW: No label encoding needed for numeric IDs
                # Just ensure they're integers and track max
                le = LabelEncoder()
                data[f'{col}_enc']  = le.fit_transform(data[col])
                data[f'{col}_enc'] = data[f'{col}_enc'].astype(int)
                X_categorical.append(data[f'{col}_enc'].values)
                self.label_encoders[col] = le
            else:
                # NEW: Clip to max seen during training
                encoder = self.label_encoders[col]
                rep = {label:i for i, label in enumerate(encoder.classes_)}
                data[f'{col}_enc'] = [rep.get(label,self.max_categorical_values[col]+1) for label in data[col]]
                X_categorical.append(data[f'{col}_enc'].values)
        
        X_numerical = data[self.numerical_cyclical_vars].values
        
        return X_categorical + [X_numerical]

    
    def compile_model(self):
        """Compile with cyclic-aware loss"""
        def cyclic_loss(y_true, y_pred):
            # Calculate angular difference
            true_angle = tf.atan2(y_true[:, 0], y_true[:, 1])
            pred_angle = tf.atan2(y_pred[:, 0], y_pred[:, 1])
            diff = tf.abs(true_angle - pred_angle)
            circular_diff = tf.minimum(diff, 2*np.pi - diff)
            return tf.reduce_mean(tf.square(circular_diff))
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss=cyclic_loss,
            metrics=['mae']
        )
    
    def train_model(self, X, y, epochs=100, batch_size=32768, validation_split=0.2):
        """Train with early stopping"""
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
        ]
        
        history = self.model.fit(
            x=X,
            y=y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=2
        )
        
        self._plot_training(history)
        self.model.save(self.model_name)
        return history
    
    def _plot_training(self, history):
        """Plot training metrics"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Cyclic Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Train MAE')
        plt.plot(history.history['val_mae'], label='Val MAE')
        plt.title('Mean Angular Error')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.model_name[:-6]}_training.png')
        plt.show()
    
    def predict_hours(self, X):
        """Predict hours from model output"""
        preds = self.model.predict(X)
        y_sin = preds[:, 0]
        y_cos = preds[:, 1]
        radians = np.arctan2(y_sin, y_cos)
        hour  =  (radians * 24 /(2*np.pi)) % 24
        return np.round(hour)%24
