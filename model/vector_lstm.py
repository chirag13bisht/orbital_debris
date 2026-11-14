"""
model/vector_lstm.py
Stacked two-layer LSTM trained in the continuous ℝ³ vector space.
Also defines the Angular-LSTM baseline for ablation.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import config


def build_vector_lstm(input_shape=None, name="VectorLSTM") -> keras.Model:
    """
    Proposed model: stacked LSTM trained on orbit normal vectors.

    Architecture:
      Input (W, 7) → LSTM(128) → LSTM(64) → Dense(32, ReLU) → Output(3)

    Loss is MSE computed in ℝ³ vector space — by construction avoids
    angular gradient pathology.
    """
    if input_shape is None:
        input_shape = (config.LOOKBACK_WINDOW, config.INPUT_DIM)

    inp = keras.Input(shape=input_shape, name="input")

    x = layers.LSTM(
        config.LSTM_UNITS_L1,
        return_sequences=True,
        dropout=config.DROPOUT_KERNEL,
        recurrent_dropout=config.DROPOUT_RECURRENT,
        kernel_regularizer=regularizers.l2(config.L2_LAMBDA),
        name="lstm_1"
    )(inp)

    x = layers.LSTM(
        config.LSTM_UNITS_L2,
        return_sequences=False,
        dropout=config.DROPOUT_KERNEL,
        recurrent_dropout=config.DROPOUT_RECURRENT,
        kernel_regularizer=regularizers.l2(config.L2_LAMBDA),
        name="lstm_2"
    )(x)

    x = layers.Dense(
        config.DENSE_UNITS,
        activation="relu",
        kernel_regularizer=regularizers.l2(config.L2_LAMBDA),
        name="dense"
    )(x)

    out = layers.Dense(config.OUTPUT_DIM, activation="linear", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name=name)
    return model


def build_angular_lstm(input_shape=None, name="AngularLSTM") -> keras.Model:
    """
    Baseline: identical architecture but trained on raw angular elements.
    Input features: [inc, raan, a, e, n_motion, M_cos, M_sin] (7-dim)
    Output: [inc, raan, a] at t+1

    This is the comparison model — it suffers from wrap-around
    discontinuities in the RAAN channel.
    """
    if input_shape is None:
        input_shape = (config.LOOKBACK_WINDOW, config.INPUT_DIM)

    inp = keras.Input(shape=input_shape, name="input")

    x = layers.LSTM(
        config.LSTM_UNITS_L1,
        return_sequences=True,
        dropout=config.DROPOUT_KERNEL,
        recurrent_dropout=config.DROPOUT_RECURRENT,
        name="lstm_1"
    )(inp)

    x = layers.LSTM(
        config.LSTM_UNITS_L2,
        return_sequences=False,
        dropout=config.DROPOUT_KERNEL,
        recurrent_dropout=config.DROPOUT_RECURRENT,
        name="lstm_2"
    )(x)

    x = layers.Dense(config.DENSE_UNITS, activation="relu", name="dense")(x)

    out = layers.Dense(config.OUTPUT_DIM, activation="linear", name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name=name)
    return model


def get_callbacks(model_name: str) -> list:
    """Standard callbacks: early stopping + LR reduction + checkpointing."""
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOP_PAT,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.LR_REDUCE_FACTOR,
            patience=config.LR_REDUCE_PAT,
            min_lr=config.LR_MIN,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=f"outputs/{model_name}_best.keras",
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
        keras.callbacks.CSVLogger(f"outputs/{model_name}_history.csv"),
    ]


def compile_model(model: keras.Model) -> keras.Model:
    """Compile with Adam and MSE loss."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LR_INITIAL),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train_model(model: keras.Model, X_train, y_train,
                X_val, y_val) -> keras.callbacks.History:
    """Train one model variant."""
    tf.random.set_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.MAX_EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=get_callbacks(model.name),
        verbose=1,
    )
    return history


if __name__ == "__main__":
    model = build_vector_lstm()
    model = compile_model(model)
    model.summary()
    print(f"\nTotal trainable parameters: {model.count_params():,}")
