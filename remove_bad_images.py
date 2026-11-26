import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"

bad_df = pd.read_csv(DATASET_DIR / "bad_images.csv")
bad_ids = bad_df["id_code"].astype(str).tolist()

csv_files = [
    "cleaned_data.csv",
    "train_split.csv",
    "val_split.csv",
    "test_split.csv"
]

for file_name in csv_files:
    file_path = DATASET_DIR / file_name

    df = pd.read_csv(file_path)
    original_count = len(df)

    df = df[~df["id_code"].astype(str).isin(bad_ids)]
    df.to_csv(file_path, index=False)

    print(file_name, ":", original_count, "->", len(df))

print("Bad image removed from CSV files.")