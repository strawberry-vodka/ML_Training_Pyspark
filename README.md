import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

def preprocess_data(df):
    """Prepare features and targets with circular encoding"""
    # Extract hour and convert to radians for circular encoding
    df['click_hour_rad'] = df['click_timestamp'].dt.hour * (2 * np.pi / 24)
    
    # Create circular targets
    y_sin = np.sin(df['click_hour_rad'])
    y_cos = np.cos(df['click_hour_rad'])
    y = np.column_stack((y_sin, y_cos))
    
    # Identify numeric columns (excluding IDs and timestamps)
    numeric_cols = [col for col in df.columns if col not in ['audience_id', 'click_timestamp', 'click_hour_rad']]
    
    return df, y, numeric_cols

def temporal_split(df, y, test_size=0.2):
    """Time-based split maintaining temporal order"""
    split_idx = int(len(df) * (1 - test_size))
    return (df.iloc[:split_idx], df.iloc[split_idx:], 
            y[:split_idx], y[split_idx:])

def build_model(n_numeric_features, max_audience_id, embedding_dim=16):
    """Build neural network with audience embeddings"""
    # Input layers
    audience_input = Input(shape=(1,), name='audience_id')
    numeric_input = Input(shape=(n_numeric_features,), name='numeric')
    
    # Embedding layer (no label encoding needed)
    embed = Embedding(input_dim=max_audience_id + 1,  # +1 for safety
                     output_dim=embedding_dim)(audience_input)
    embed_flat = Flatten()(embed)
    
    # Combined model
    x = Concatenate()([numeric_input, embed_flat])
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(2, activation='linear')(x)  # sin and cos components
    
    model = Model(inputs=[audience_input, numeric_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def train_cyclic_nn(train_df, test_df=None):
    """Complete training pipeline"""
    # Preprocess training data
    train_df, y_train, numeric_cols = preprocess_data(train_df)
    
    # Get max audience ID (assuming it's already numeric)
    max_audience_id = int(train_df['audience_id'].max())
    
    # Temporal split if no test_df provided
    if test_df is None:
        X_train, X_val, y_train, y_val = temporal_split(train_df, y_train)
    else:
        # Preprocess test data
        test_df, y_val, _ = preprocess_data(test_df)
        X_train, X_val = train_df, test_df
    
    # Build model
    model = build_model(
        n_numeric_features=len(numeric_cols),
        max_audience_id=max_audience_id,
        embedding_dim=16
    )
    
    # Train model
    history = model.fit(
        x={
            'audience_id': X_train['audience_id'].values,
            'numeric': X_train[numeric_cols].values
        },
        y=y_train,
        validation_data=(
            {
                'audience_id': X_val['audience_id'].values,
                'numeric': X_val[numeric_cols].values
            },
            y_val
        ),
        epochs=50,
        batch_size=256,
        callbacks=[
            EarlyStopping(patience=5, restore_best_weights=True)
        ],
        verbose=2
    )
    
    return model, history, X_val, y_val

def predict_hours(model, df, numeric_cols):
    """Predict hours from trained model"""
    # Get predictions (sin and cos values)
    preds = model.predict({
        'audience_id': df['audience_id'].values,
        'numeric': df[numeric_cols].values
    })
    
    # Convert to angles then hours
    angles = np.arctan2(preds[:, 0], preds[:, 1])
    hours = (angles * 24 / (2 * np.pi)) % 24
    return np.round(hours).astype(int)

def evaluate_predictions(true_hours, pred_hours):
    """Calculate cyclic accuracy metrics"""
    # Calculate circular difference
    diff = np.abs(true_hours - pred_hours)
    circular_diff = np.minimum(diff, 24 - diff)
    
    # Metrics
    accuracy_1h = np.mean(circular_diff <= 1)
    accuracy_2h = np.mean(circular_diff <= 2)
    mean_diff = np.mean(circular_diff)
    
    return {
        'accuracy_within_1h': accuracy_1h,
        'accuracy_within_2h': accuracy_2h,
        'mean_circular_diff': mean_diff
    }

# Example usage:
# Assuming you have train_df and test_df DataFrames with:
# - audience_id (numeric)
# - click_timestamp
# - other numeric features

# Train model
model, history, X_val, y_val = train_cyclic_nn(train_df, test_df)

# Get numeric columns (same as used in training)
_, _, numeric_cols = preprocess_data(train_df)

# Predict on validation/test set
pred_hours = predict_hours(model, X_val, numeric_cols)

# Get true hours from test data
true_hours = (X_val['click_timestamp'].dt.hour).values

# Evaluate
metrics = evaluate_predictions(true_hours, pred_hours)
print(f"Validation Metrics: {metrics}")
