"""
Data preprocessing, feature engineering, and stratified partitioning module.
Dataset: CDC Diabetes Health Indicators (UCI ID: 891)
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_raw_data(filepath: str = "data/raw/diabetes_binary_health_indicators_BRFSS2015.csv") -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at {filepath}")
    return pd.read_csv(filepath)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs domain-grounded clinical and behavioral risk features:
    1. Cardiometabolic Risk Score: Cumulative score of HighBP + HighChol + Obese BMI.
    2. Total Unhealthy Days: Composite score of MentHlth + PhysHlth (capped at 30).
    3. Healthcare Access Barrier: Indicator if patient had no doc due to cost or lacks healthcare.
    4. Lifestyle Risk Index: Smoker (1) + Heavy Alcohol (1) - Physical Activity (1).
    5. High-Risk Senior Indicator: Age group >= 9 (age 60+) combined with HighBP or HighChol.
    """
    df = df.copy()

    # 1. Cardiometabolic Risk Score
    is_obese = (df["BMI"] >= 30).astype(int)
    df["Cardiometabolic_Risk"] = df["HighBP"] + df["HighChol"] + is_obese

    # 2. Total Unhealthy Days (Physical + Mental health burden)
    df["Total_Unhealthy_Days"] = np.clip(df["MentHlth"] + df["PhysHlth"], 0, 30)

    # 3. Healthcare Access Barrier
    df["Healthcare_Barrier"] = ((df["NoDocbcCost"] == 1) | (df["AnyHealthcare"] == 0)).astype(int)

    # 4. Lifestyle Risk Index
    df["Lifestyle_Risk_Index"] = df["Smoker"] + df["HvyAlcoholConsump"] - df["PhysActivity"]

    # 5. Senior Vascular Risk (Age >= 9 corresponds to age 60+ in BRFSS)
    df["Senior_Vascular_Risk"] = ((df["Age"] >= 9) & ((df["HighBP"] == 1) | (df["HighChol"] == 1))).astype(int)

    return df


def preprocess_and_split(
    raw_path: str = "data/raw/diabetes_binary_health_indicators_BRFSS2015.csv",
    output_dir: str = "data/processed",
    random_state: int = 42
):
    """
    Loads raw data, performs feature engineering, creates stratified 70/15/15 train/val/test splits,
    fits standard scalers strictly on training data to prevent data leakage, and persists all artifacts.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = load_raw_data(raw_path)
    
    print(f"[INFO] Loaded raw dataset with shape: {df.shape}")
    print("[INFO] Performing domain-grounded feature engineering...")
    df_engineered = engineer_features(df)
    print(f"[INFO] Dataset shape post-engineering: {df_engineered.shape}")

    target_col = "Diabetes_binary"
    feature_cols = [c for c in df_engineered.columns if c != target_col]

    X = df_engineered[feature_cols]
    y = df_engineered[target_col]

    # Stratified split: 70% Train, 30% Temp (Val + Test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )

    # Split Temp into 50/50 -> 15% Validation, 15% Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    print("\n--- Dataset Partition Shapes ---")
    print(f"Training set:   {X_train.shape} (Positives: {y_train.sum():,} | {y_train.mean()*100:.2f}%)")
    print(f"Validation set: {X_val.shape}   (Positives: {y_val.sum():,} | {y_val.mean()*100:.2f}%)")
    print(f"Test set:       {X_test.shape}  (Positives: {y_test.sum():,} | {y_test.mean()*100:.2f}%)")

    # Features to scale (continuous and multi-level ordinal features)
    continuous_features = [
        "BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", 
        "Income", "Cardiometabolic_Risk", "Total_Unhealthy_Days", "Lifestyle_Risk_Index"
    ]
    binary_features = [c for c in feature_cols if c not in continuous_features]

    print(f"\n[INFO] Continuous features to scale ({len(continuous_features)}): {continuous_features}")
    print(f"[INFO] Binary/indicator features preserved ({len(binary_features)}): {binary_features}")

    scaler = StandardScaler()
    scaler.fit(X_train[continuous_features])

    # Transform partitions
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[continuous_features] = scaler.transform(X_train[continuous_features])
    X_val_scaled[continuous_features] = scaler.transform(X_val[continuous_features])
    X_test_scaled[continuous_features] = scaler.transform(X_test[continuous_features])

    # Save scaler artifact
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"[SUCCESS] Saved fitted StandardScaler to {scaler_path}")

    # Save feature names
    feature_meta = {
        "all_features": feature_cols,
        "continuous_features": continuous_features,
        "binary_features": binary_features,
        "target": target_col
    }
    joblib.dump(feature_meta, os.path.join(output_dir, "feature_metadata.joblib"))

    # Save processed CSVs
    train_df = pd.concat([X_train_scaled, y_train], axis=1)
    val_df = pd.concat([X_val_scaled, y_val], axis=1)
    test_df = pd.concat([X_test_scaled, y_test], axis=1)

    train_df.to_csv(os.path.join(output_dir, "train_processed.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val_processed.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test_processed.csv"), index=False)
    print(f"[SUCCESS] Saved processed datasets to {output_dir}/")

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, feature_meta


if __name__ == "__main__":
    preprocess_and_split()
