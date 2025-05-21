import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

class CyclicOrdinalRegressor:
    def __init__(self, numerical_cyclical_vars, numerical_vars, categorical_vars, name, seed=42):
        """Initialize with seed control for all components"""
        self.seed = seed
        self._set_all_seeds(seed)
        
        # Original initialization code
        self.numerical_cyclical_vars = numerical_cyclical_vars
        self.numerical_vars = numerical_vars
        self.categorical_vars = categorical_vars
        self.model_name = f"{name}.keras"
        # ... rest of your existing __init__ code ...

    def _set_all_seeds(self, seed):
        """Set seeds for all random number generators"""
        # Python
        random.seed(seed)
        # Numpy
        np.random.seed(seed)
        # TensorFlow
        tf.random.set_seed(seed)
        # Keras backend
        K.set_floatx('float32')
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['PYTHONHASHSEED'] = str(seed)

def build_model(self):
    """Build model with seed-controlled initialization"""
    # Reset graph and seeds for clean slate
    tf.keras.backend.clear_session()
    self._set_all_seeds(self.seed)
    
    # Original model building code
    # ... (your existing build_model code) ...
    
    # Compile with deterministic ops
    self.model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=self.cyclic_loss,
        metrics=['mae'],
        run_eagerly=False  # For determinism
    )

def train_model(self, train_data, target_column, **kwargs):
    """Training with full seed control"""
    # Set seeds before any random operations
    self._set_all_seeds(self.seed)
    
    # Data shuffling (if any) should use seed
    train_data = train_data.sample(frac=1, random_state=self.seed)  # If shuffling
    
    # Original training code
    # ... (your existing train_model code) ...
    
    # Add seed to fit() if using shuffle=True
    history = self.model.fit(
        x=X_train_dict,
        y=y_train,
        shuffle=False,  # Or use shuffle=True with seed
        # shuffle=True,  # If enabled:
        # shuffle_buffer_size=len(train_data),  # Full shuffle
        # seed=self.seed,
        **kwargs
    )
