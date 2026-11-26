import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Project folders
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
IMAGE_DIR = DATASET_DIR / "train_images"

# Read dataset
csv_path = DATASET_DIR / "train.csv"
df = pd.read_csv(csv_path)

print("Original dataset shape:", df.shape)

# Remove missing values
df = df.dropna(subset=["id_code", "diagnosis"])

# Remove duplicate image IDs
df = df.drop_duplicates(subset=["id_code"])

# Keep only rows whose image files exist
df["image_path"] = df["id_code"].apply(
    lambda x: str(IMAGE_DIR / f"{x}.png")
)

df = df[df["image_path"].apply(lambda x: Path(x).exists())]

# Save cleaned dataset
cleaned_path = DATASET_DIR / "cleaned_data.csv"
df.to_csv(cleaned_path, index=False)

print("Cleaned dataset shape:", df.shape)
print("Cleaned data saved to:", cleaned_path)

# Split into train, validation and test sets
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=42,
    stratify=df["diagnosis"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["diagnosis"]
)

# Save split files
train_df.to_csv(DATASET_DIR / "train_split.csv", index=False)
val_df.to_csv(DATASET_DIR / "val_split.csv", index=False)
test_df.to_csv(DATASET_DIR / "test_split.csv", index=False)

print("Train images:", len(train_df))
print("Validation images:", len(val_df))
print("Test images:", len(test_df))
print("Data preparation completed successfully!")