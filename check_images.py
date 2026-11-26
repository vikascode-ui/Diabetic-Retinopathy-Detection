import pandas as pd
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
IMAGE_DIR = DATASET_DIR / "train_images"

# Check all image IDs from cleaned dataset
df = pd.read_csv(DATASET_DIR / "cleaned_data.csv")

bad_images = []
good_images = 0

for image_id in df["id_code"].astype(str):
    image_path = IMAGE_DIR / f"{image_id}.png"

    try:
        with Image.open(image_path) as img:
            img.verify()
        good_images += 1

    except Exception as error:
        bad_images.append((image_id, str(error)))

print("Total images checked:", len(df))
print("Good images:", good_images)
print("Bad images:", len(bad_images))

# Save list of bad images
bad_df = pd.DataFrame(bad_images, columns=["id_code", "error"])
bad_df.to_csv(DATASET_DIR / "bad_images.csv", index=False)

print("Bad image list saved to dataset/bad_images.csv")