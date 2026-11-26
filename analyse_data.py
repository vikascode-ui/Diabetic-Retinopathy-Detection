import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"

# Load cleaned data
df = pd.read_csv(DATASET_DIR / "cleaned_data.csv")

print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate IDs:", df["id_code"].duplicated().sum())

print("\nDisease grade distribution:")
print(df["diagnosis"].value_counts().sort_index())

# Plot class distribution
df["diagnosis"].value_counts().sort_index().plot(
    kind="bar",
    color="skyblue",
    edgecolor="black"
)

plt.title("Diabetic Retinopathy Class Distribution")
plt.xlabel("Disease Grade (0 to 4)")
plt.ylabel("Number of Images")
plt.xticks(rotation=0)
plt.tight_layout()

# Save graph
output_dir = BASE_DIR / "outputs"
output_dir.mkdir(exist_ok=True)

plt.savefig(output_dir / "class_distribution.png")
plt.show()

print("\nAnalysis completed!")
print("Graph saved in outputs/class_distribution.png")