"""
Data loading and exploratory inspection module for CDC Diabetes Health Indicators.
Dataset: UCI Machine Learning Repository (ID: 891)
Source: Centers for Disease Control and Prevention (CDC) - BRFSS 2015
"""

import os
import zipfile
import pandas as pd
from ucimlrepo import fetch_ucirepo


def download_and_save_raw_data(data_dir: str = "data/raw"):
    """
    Fetches the raw dataset from the UCI ML Repository and saves both
    the uncompressed CSV and a zipped archive (for Google Drive submission).
    """
    os.makedirs(data_dir, exist_ok=True)
    raw_csv_path = os.path.join(data_dir, "diabetes_binary_health_indicators_BRFSS2015.csv")
    raw_zip_path = os.path.join(data_dir, "raw_dataset.zip")

    print("[INFO] Fetching CDC Diabetes Health Indicators (UCI ID: 891)...")
    dataset = fetch_ucirepo(id=891)
    
    # Extract features and targets
    features = dataset.data.features
    targets = dataset.data.targets

    # Combine into single DataFrame
    df = pd.concat([targets, features], axis=1)

    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Target column name: {targets.columns.tolist()}")
    print(f"[INFO] Features ({len(features.columns)}): {features.columns.tolist()}")

    # Save to CSV
    df.to_csv(raw_csv_path, index=False)
    print(f"[SUCCESS] Saved raw dataset to {raw_csv_path}")

    # Create zip archive for Google Drive deliverable
    with zipfile.ZipFile(raw_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(raw_csv_path, arcname="diabetes_binary_health_indicators_BRFSS2015.csv")
    print(f"[SUCCESS] Created zipped dataset archive at {raw_zip_path}")

    # Inspect class balance and summary statistics
    target_col = targets.columns[0]
    counts = df[target_col].value_counts()
    percentages = df[target_col].value_counts(normalize=True) * 100
    
    print("\n--- Target Variable Distribution ---")
    for cls in counts.index:
        print(f"Class {cls}: {counts[cls]:,} samples ({percentages[cls]:.2f}%)")

    missing_vals = df.isnull().sum().sum()
    print(f"\nTotal missing values: {missing_vals}")
    print("\nData Types:")
    print(df.dtypes)

    return df


if __name__ == "__main__":
    download_and_save_raw_data()
