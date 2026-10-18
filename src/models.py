"""
Model definitions, tf.data pipeline builders, and architecture factory.
Implements Scikit-Learn algorithms and TensorFlow Keras Functional API architectures.
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier


def create_tf_dataset(X, y=None, batch_size=256, shuffle=False, buffer_size=10000):
    """
    Constructs high-performance tf.data.Dataset pipeline with shuffling, batching, and prefetching.
    """
    if y is not None:
        ds = tf.data.Dataset.from_tensor_slices((X.values.astype("float32"), y.values.astype("float32")))
    else:
        ds = tf.data.Dataset.from_tensor_slices(X.values.astype("float32"))

    if shuffle:
        ds = ds.shuffle(buffer_size=buffer_size, seed=42)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def get_classical_model(model_name: str, **kwargs):
    """
    Factory for Scikit-Learn classification algorithms.
    """
    if model_name == "logistic_regression":
        return LogisticRegression(
            C=kwargs.get("C", 1.0),
            penalty="l2",
            solver="lbfgs",
            max_iter=kwargs.get("max_iter", 1000),
            class_weight=kwargs.get("class_weight", None),
            random_state=42
        )
    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=kwargs.get("n_estimators", 100),
            max_depth=kwargs.get("max_depth", 12),
            min_samples_split=kwargs.get("min_samples_split", 20),
            class_weight=kwargs.get("class_weight", None),
            random_state=42,
            n_jobs=-1
        )
    elif model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=kwargs.get("learning_rate", 0.08),
            max_iter=kwargs.get("max_iter", 150),
            max_leaf_nodes=kwargs.get("max_leaf_nodes", 31),
            class_weight=kwargs.get("class_weight", None),
            random_state=42
        )
    else:
        raise ValueError(f"Unknown classical model: {model_name}")


def build_baseline_mlp(input_dim: int, learning_rate: float = 1e-3) -> keras.Model:
    """
    Experiment 5: Baseline Feedforward Multi-Layer Perceptron (Keras Functional API).
    Input -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(1, Sigmoid)
    """
    inputs = keras.Input(shape=(input_dim,), name="clinical_features_input")
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = layers.Dense(32, activation="relu", name="dense_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="diabetes_probability")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Baseline_MLP")
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc_roc", curve="ROC"),
            keras.metrics.AUC(name="auc_pr", curve="PR")
        ]
    )
    return model


def build_regularized_mlp(
    input_dim: int, 
    learning_rate: float = 1e-3, 
    use_lr_schedule: bool = False,
    l2_reg: float = 1e-4
) -> keras.Model:
    """
    Experiments 6 & 7: Deep Regularized MLP with Batch Normalization, Dropout, and L2 Regularization.
    """
    inputs = keras.Input(shape=(input_dim,), name="clinical_features_input")
    
    # Layer block 1
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="dense_1")(inputs)
    x = layers.BatchNormalization(name="batch_norm_1")(x)
    x = layers.Dropout(0.3, name="dropout_1")(x)

    # Layer block 2
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="dense_2")(x)
    x = layers.BatchNormalization(name="batch_norm_2")(x)
    x = layers.Dropout(0.2, name="dropout_2")(x)

    # Layer block 3
    x = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2_reg), name="dense_3")(x)
    x = layers.BatchNormalization(name="batch_norm_3")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="diabetes_probability")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="Regularized_Deep_MLP")

    if use_lr_schedule:
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=1000,
            decay_rate=0.90,
            staircase=True
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    else:
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc_roc", curve="ROC"),
            keras.metrics.AUC(name="auc_pr", curve="PR")
        ]
    )
    return model
