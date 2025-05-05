import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# 1. Preprocessing
def prepare_data(df):
    # Convert click hour to radians
    hour_rad = df['click_hour'] * 2 * np.pi / 24
    df['target_sin'] = np.sin(hour_rad)
    df['target_cos'] = np.cos(hour_rad)

    # Encode audience_id
    le = LabelEncoder()
    df['audience_idx'] = le.fit_transform(df['audienceid'])

    # Scale features (exclude ID and target)
    feature_cols = [col for col in df.columns if col.startswith(('feat_90d_', 'feat_30d_', 'feat_7d_'))]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df, feature_cols, le

# 2. Build Model
def build_model(num_features, num_audience_ids, embed_dim=16):
    audience_input = tf.keras.Input(shape=(1,), name="audience_idx", dtype="int32")
    feature_input = tf.keras.Input(shape=(num_features,), name="features")

    x_embed = tf.keras.layers.Embedding(input_dim=num_audience_ids, output_dim=embed_dim)(audience_input)
    x_embed = tf.keras.layers.Flatten()(x_embed)

    x = tf.keras.layers.Concatenate()([x_embed, feature_input])
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    output = tf.keras.layers.Dense(2, activation='linear', name='angle_output')(x)

    model = tf.keras.Model(inputs=[audience_input, feature_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# 3. Training
def train_model(df):
    df, feature_cols, le = prepare_data(df)

    X_audience = df['audience_idx'].values
    X_features = df[feature_cols].values
    y = df[['target_sin', 'target_cos']].values

    X_train_aud, X_val_aud, X_train_feat, X_val_feat, y_train, y_val = train_test_split(
        X_audience, X_features, y, test_size=0.2, shuffle=False
    )

    model = build_model(num_features=len(feature_cols), num_audience_ids=df['audience_idx'].nunique())

    model.fit(
        {"audience_idx": X_train_aud, "features": X_train_feat},
        y_train,
        validation_data=({"audience_idx": X_val_aud, "features": X_val_feat}, y_val),
        epochs=20,
        batch_size=512
    )

    return model, le, feature_cols

# 4. Prediction Conversion
def predict_hour_from_sin_cos(y_pred):
    sin_val = y_pred[:, 0]
    cos_val = y_pred[:, 1]
    radians = np.arctan2(sin_val, cos_val)
    hours = (radians * 24 / (2 * np.pi)) % 24
    return np.round(hours).astype(int)
