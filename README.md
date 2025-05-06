import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, Concatenate, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import datetime

# ---------------------
# Prepare your data
# ---------------------
def encode_cyclic_column(series, max_val):
    radians = 2 * np.pi * series / max_val
    return np.sin(radians), np.cos(radians)

def preprocess_data(df):
    df = df.sort_values('click_timestamp')  # Important for temporal split

    # Target
    df['click_hour'] = df['click_timestamp'].dt.hour
    hour_rad = 2 * np.pi * df['click_hour'] / 24
    y_sin = np.sin(hour_rad)
    y_cos = np.cos(hour_rad)
    
    # Cyclic features
    df['day_of_week'] = df['click_timestamp'].dt.dayofweek
    df['hour'] = df['click_timestamp'].dt.hour
    df['sin_hour'], df['cos_hour'] = encode_cyclic_column(df['hour'], 24)
    df['sin_dow'], df['cos_dow'] = encode_cyclic_column(df['day_of_week'], 7)

    # Drop raw cyclic cols
    df = df.drop(['click_timestamp', 'hour', 'day_of_week'], axis=1)

    # Feature scaling
    numeric_features = df.drop(['audience_id', 'click_hour'], axis=1).columns
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    return df, np.column_stack((y_sin, y_cos)), numeric_features

# ---------------------
# Temporal split
# ---------------------
def temporal_split(df, y, split_ratio=0.8):
    split_idx = int(len(df) * split_ratio)
    X_train = df.iloc[:split_idx]
    X_val = df.iloc[split_idx:]
    y_train = y[:split_idx]
    y_val = y[split_idx:]
    return X_train, X_val, y_train, y_val

# ---------------------
# Neural Network model
# ---------------------
def build_model(n_numeric_features, n_audience_ids, embedding_dim=16):
    # Inputs
    audience_input = Input(shape=(1,), name='audience_id')
    numeric_input = Input(shape=(n_numeric_features,), name='numeric')

    # Embedding for audience
    embed = Embedding(input_dim=n_audience_ids+1, output_dim=embedding_dim)(audience_input)
    embed_flat = Flatten()(embed)

    # Concatenate and build model
    x = Concatenate()([numeric_input, embed_flat])
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(2, activation='linear')(x)  # Output: sin and cos

    model = Model(inputs=[audience_input, numeric_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ---------------------
# Training pipeline
# ---------------------
def train_cyclic_nn(df):
    df, y, numeric_cols = preprocess_data(df)

    # Audience ID encoding
    df['audience_id'] = df['audience_id'].astype('category').cat.codes
    n_audience_ids = df['audience_id'].nunique()

    # Temporal split
    X_train, X_val, y_train, y_val = temporal_split(df, y)

    # Model
    model = build_model(
        n_numeric_features=len(numeric_cols),
        n_audience_ids=n_audience_ids,
        embedding_dim=16
    )

    # Fit model
    model.fit(
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
    
    return model, X_val, y_val

# ---------------------
# Convert predictions to hour
# ---------------------
def predict_hour(sin_pred, cos_pred):
    radians = np.arctan2(sin_pred, cos_pred)
    hours = (radians * 24 / (2 * np.pi)) % 24
    return np.round(hours).astype(int)
