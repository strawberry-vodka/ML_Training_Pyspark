import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

class CyclicOrdinalRegressor:
    
    def __init__(self, numerical_cyclical_vars, numerical_vars, categorical_vars, name):
        self.categorical_vars = categorical_vars
        self.numerical_cyclical_vars = numerical_cyclical_vars  # Already in sin/cos form
        self.numerical_vars = numerical_vars  # Non-cyclic features to be scaled
        self.model_name = name + '.keras'
        self.embedding_layers = []
        self.input_layers = []
        self.embedding_sizes = {}
        self.model = None
        self.label_encoders = {}
        self.max_categorical_values = {}
        self.scaler = StandardScaler()  # For non-cyclic numerical features
        
    def calculate_embedding_sizes(self, data):
        """Calculate embedding dimensions with buffer"""
        for col in self.categorical_vars:
            # Find maximum encoded value
            self.max_categorical_values[col] = int(data[col].nunique())
            
            # Calculate embedding size
            unique_vals = len(data[col].unique())
            self.embedding_sizes[col] = max(2, min(50, round(1.6 * np.log2(unique_vals))))
            print(f"{col}: Unique values={unique_vals}, Embedding size={self.embedding_sizes[col]}")

    def build_model(self):
        """Build model architecture with multiple input types"""
        # 1. Categorical inputs
        self._build_categorical_inputs()
        
        # 2. Cyclical numerical inputs (sin/cos)
        numerical_cyclical_input = Input(
            shape=(len(self.numerical_cyclical_vars),), 
            name='numerical_cyclical_input'
        )
        self.input_layers.append(numerical_cyclical_input)
        
        # 3. Regular numerical inputs (will be scaled)
        numerical_input = Input(
            shape=(len(self.numerical_vars),), 
            name='numerical_input'
        )
        self.input_layers.append(numerical_input)
        
        # 4. Concatenate all inputs
        concatenated = Concatenate()(
            self.embedding_layers + 
            [numerical_cyclical_input, numerical_input]
        )
        
        # 5. Hidden layers
        x = Dense(128, activation='relu')(concatenated)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        x = Dense(64, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(32, activation='relu')(x)
        
        # 6. Output layer (sin/cos components)
        output = Dense(2, activation='linear')(x)
        
        self.model = Model(inputs=self.input_layers, outputs=output)
        print(self.model.summary())

    def _build_categorical_inputs(self):
        """Create embedding layers with safe dimensions"""
        for col in self.categorical_vars:
            input_layer = Input(shape=(1,), name=col)
            embedding = Embedding(
                input_dim=self.max_categorical_values[col] + 1000,  # Buffer
                output_dim=self.embedding_sizes[col],
                name=f'embed_{col}'
            )(input_layer)
            flattened = Flatten()(embedding)
            
            self.input_layers.append(input_layer)
            self.embedding_layers.append(flattened)
    
    def preprocess_data(self, data, is_train=True):
        """Prepare all input types with proper transformations"""
        X_categorical = []
        
        # 1. Process categorical columns
        for col in self.categorical_vars:
            if is_train:
                le = LabelEncoder()
                data[f'{col}_enc'] = le.fit_transform(data[col])
                self.label_encoders[col] = le
            else:
                le = self.label_encoders[col]
                # Handle unseen categories by assigning to a new category
                data[f'{col}_enc'] = data[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else self.max_categorical_values[col] + 1
                )
            X_categorical.append(data[f'{col}_enc'].values.reshape(-1, 1))
        
        # 2. Process cyclical numerical features (no scaling needed)
        X_numerical_cyclical = data[self.numerical_cyclical_vars].values
        
        # 3. Process regular numerical features (with scaling)
        if is_train:
            X_numerical_scaled = self.scaler.fit_transform(data[self.numerical_vars])
        else:
            X_numerical_scaled = self.scaler.transform(data[self.numerical_vars])
        
        # Combine all inputs
        return X_categorical + [X_numerical_cyclical, X_numerical_scaled]

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
        angles = np.arctan2(preds[:, 0], preds[:, 1])
        hours = (angles * 24 / (2 * np.pi)) % 24
        return np.round(hours).astype(int)

# Example Usage
if __name__ == "__main__":
    # Sample data
    data = pd.DataFrame({
        'audience_id': np.random.choice(['A','B','C','D'], 1000),
        'hour_sin': np.sin(np.linspace(0, 2*np.pi, 1000)),
        'hour_cos': np.cos(np.linspace(0, 2*np.pi, 1000)),
        'user_age': np.random.randint(18, 70, 1000),
        'purchase_count': np.random.poisson(3, 1000),
        'click_hour': np.random.randint(0, 24, 1000)
    })
    
    # Initialize model
    model = CyclicOrdinalRegressor(
        numerical_cyclical_vars=['hour_sin', 'hour_cos'],
        numerical_vars=['user_age', 'purchase_count'],
        categorical_vars=['audience_id'],
        name='email_time_predictor'
    )
    
    # Prepare data
    model.calculate_embedding_sizes(data)
    X = model.preprocess_data(data, is_train=True)
    y = np.column_stack([
        np.sin(data['click_hour'] * (2*np.pi/24)),
        np.cos(data['click_hour'] * (2*np.pi/24))
    ])
    
    # Build, compile and train
    model.build_model()
    model.compile_model()
    history = model.train_model(X, y, epochs=50, batch_size=256)
    
    # Predict on new data
    test_data = pd.DataFrame({
        'audience_id': np.random.choice(['A','B','E'], 100),  # Note 'E' is new
        'hour_sin': np.sin(np.linspace(0, 2*np.pi, 100)),
        'hour_cos': np.cos(np.linspace(0, 2*np.pi, 100)),
        'user_age': np.random.randint(20, 80, 100),
        'purchase_count': np.random.poisson(5, 100)
    })
    
    X_test = model.preprocess_data(test_data, is_train=False)
    predicted_hours = model.predict_hours(X_test)
    print("Predicted hours:", predicted_hours)
