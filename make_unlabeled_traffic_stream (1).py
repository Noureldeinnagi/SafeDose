import os
import pandas as pd

DATA_DIR = r"D:\Grad\Dataset"
FEATURE_COLS_PATH = r"D:\Grad\derived_dataset\feature_columns.txt"
OUT_DIR = r"D:\Grad\deployment"
OUT_FILE = os.path.join(OUT_DIR, "traffic_stream_labeled_small_unmixed.csv")

ROWS_PER_FILE = 200        # keep small (e.g., 200 or 500)
RANDOM_SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)

# Load training feature order (45 columns)
with open(FEATURE_COLS_PATH, "r", encoding="utf-8") as f:
    feature_cols = [line.strip() for line in f if line.strip()]

def infer_class_name(filename: str) -> str:
    name = filename
    for suffix in ["_test.pcap.csv", "_test.csv", ".pcap.csv", ".csv"]:
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    # remove "_test" if still present
    if name.lower().endswith("_test"):
        name = name[:-5]
    return name

all_parts = []
loaded_files = []

# Keep file order deterministic (optional but recommended)
test_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv") and "_test" in f.lower()]
test_files.sort()

for file in test_files:
    path = os.path.join(DATA_DIR, file)
    df = pd.read_csv(path)

    # Ensure label exists (create it from filename)
    label = infer_class_name(file)
    df["Label"] = label

    # Check required feature columns exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"File {file} missing columns: {missing[:10]} ... total missing={len(missing)}"
        )

    # Keep features + label (Label kept ONLY for evaluation)
    df = df[feature_cols + ["Label"]]

    # Make it small (sampling within the same file)
    if len(df) > ROWS_PER_FILE:
        df = df.sample(ROWS_PER_FILE, random_state=RANDOM_SEED)

    # IMPORTANT: do NOT shuffle after concatenation (unmixed)
    all_parts.append(df)
    loaded_files.append((file, len(df)))

if not all_parts:
    raise SystemExit("No test CSV files found in D:\\Grad\\Dataset.")

# Concatenate in file order; rows inside each part keep their sampled order
stream_df = pd.concat(all_parts, ignore_index=True)

stream_df.to_csv(OUT_FILE, index=False)

print("✅ Created LABELED traffic stream (UNMIXED):", OUT_FILE)
print("Total rows:", len(stream_df))
print("Columns:", stream_df.shape[1], "(45 features + Label)")
print("\nFiles included (in order):")
for name, n in loaded_files:
    print(f"- {name}: {n} rows")

print("\nLabel distribution:")
print(stream_df["Label"].value_counts())
