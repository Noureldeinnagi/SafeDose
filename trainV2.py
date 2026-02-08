import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# =========================
# CONFIG (EDIT HERE ONLY)
# =========================
BASE_TRAIN_CSV = r"D:\Grad\derived_dataset\train_balanced_clean.csv"
LIVE_STREAM_CSV = r"D:\Grad\deployment\traffic_stream_labeled_small_unmixed.csv"

# IMPORTANT: use the ORIGINAL training artifacts (do NOT refit these)
FEATURE_COLS_PATH = r"D:\Grad\derived_dataset\feature_columns.txt"
SCALER_PATH = r"D:\Grad\derived_dataset\scaler.joblib"
LABEL_MAP_PATH = r"D:\Grad\derived_dataset\label_mapping.csv"

# Fine-tune from existing model weights
BASE_MODEL_PATH = r"D:\Grad\model_output\cnn_lstm_final.keras"
OUT_MODEL_PATH  = r"D:\Grad\model_output\cnn_lstm_finetuned.keras"
OUT_HISTORY_CSV = r"D:\Grad\model_output\finetune_history.csv"

# Mix ratio: keep base data dominant
LIVE_FRACTION = 0.15   # 15% live, 85% base (good starting point)
RANDOM_SEED = 42

# Sequence settings MUST match your model training
WINDOW_SIZE = 20
STRIDE = 5

# Fine-tune settings
LEARNING_RATE = 1e-4
EPOCHS = 10
BATCH_SIZE = 128
PATIENCE = 3

# =========================
# Helpers
# =========================
def normalize_label(s: str) -> str:
    s = str(s)
    while s.lower().endswith(".csv"):
        s = s[:-4]
    if s.lower().endswith("_train"):
        s = s[:-6]
    if s.lower().endswith("_test"):
        s = s[:-5]
    if s.lower().startswith("benign"):
        return "Benign"
    return s

def majority_vote_int(arr):
    # arr is 1D np array of ints
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[np.argmax(counts)])

# =========================
# Load feature columns (fixed order)
# =========================
with open(FEATURE_COLS_PATH, "r", encoding="utf-8") as f:
    feature_cols = [line.strip() for line in f if line.strip()]
if not feature_cols:
    raise SystemExit("feature_columns.txt is empty.")
print("✅ Feature cols:", len(feature_cols))

# =========================
# Load scaler + label mapping (OLD, frozen)
# =========================
scaler = joblib.load(SCALER_PATH)

label_map_df = pd.read_csv(LABEL_MAP_PATH)
# label_map_df columns: Label, Encoded
label_map_df["Label_norm"] = label_map_df["Label"].map(normalize_label)
# Build mapping: normalized label -> encoded id
label_to_id = {}
for _, r in label_map_df.iterrows():
    label_to_id[r["Label_norm"]] = int(r["Encoded"])

if "Benign" not in label_to_id:
    raise ValueError("Benign not found in label_mapping.csv after normalization.")

print("✅ Classes in mapping:", len(label_to_id))

# =========================
# Load base train + live stream
# =========================
base_df = pd.read_csv(BASE_TRAIN_CSV)
live_df = pd.read_csv(LIVE_STREAM_CSV)

if "Label" not in base_df.columns:
    raise ValueError("Base train CSV must contain Label column.")
if "Label" not in live_df.columns:
    raise ValueError("Live stream CSV must contain Label column (for fine-tuning).")

# Validate features
missing_base = [c for c in feature_cols if c not in base_df.columns]
missing_live = [c for c in feature_cols if c not in live_df.columns]
if missing_base:
    raise ValueError(f"Base train missing features: {missing_base[:10]} ... total={len(missing_base)}")
if missing_live:
    raise ValueError(f"Live stream missing features: {missing_live[:10]} ... total={len(missing_live)}")

# Normalize labels
base_df["Label_norm"] = base_df["Label"].astype(str).map(normalize_label)
live_df["Label_norm"] = live_df["Label"].astype(str).map(normalize_label)

# Keep only rows whose labels exist in old mapping
base_keep = base_df["Label_norm"].isin(label_to_id.keys())
live_keep = live_df["Label_norm"].isin(label_to_id.keys())

dropped_base = int((~base_keep).sum())
dropped_live = int((~live_keep).sum())

if dropped_base > 0:
    print(f"⚠️ Dropped {dropped_base} base rows with labels not in mapping.")
if dropped_live > 0:
    print(f"⚠️ Dropped {dropped_live} live rows with labels not in mapping.")

base_df = base_df.loc[base_keep, feature_cols + ["Label_norm"]].copy()
live_df = live_df.loc[live_keep, feature_cols + ["Label_norm"]].copy()

if len(live_df) == 0:
    raise SystemExit("Live stream has 0 rows after filtering by known labels.")
if len(base_df) == 0:
    raise SystemExit("Base train has 0 rows after filtering by known labels.")

# Sample live portion (so it doesn't dominate)
live_n = int(round(len(base_df) * LIVE_FRACTION))
live_n = max(1, min(live_n, len(live_df)))

live_sample = live_df.sample(n=live_n, random_state=RANDOM_SEED).reset_index(drop=True)
base_shuf   = base_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

combined_df = pd.concat([base_shuf, live_sample], ignore_index=True)
combined_df = combined_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

print("\n✅ Combined fine-tune dataset")
print("Base rows:", len(base_df))
print("Live rows used:", len(live_sample))
print("Total:", len(combined_df))
print("Label dist:\n", combined_df["Label_norm"].value_counts().head(10))

# =========================
# Encode + scale (NO refit)
# =========================
X = combined_df[feature_cols].values
y = combined_df["Label_norm"].map(label_to_id).astype(int).values

X_scaled = scaler.transform(X).astype(np.float32)

# =========================
# Create sequences
# =========================
X_seq = []
y_seq = []

num_samples = X_scaled.shape[0]
for start in range(0, num_samples - WINDOW_SIZE + 1, STRIDE):
    end = start + WINDOW_SIZE
    window_X = X_scaled[start:end]
    window_y = y[start:end]
    X_seq.append(window_X)
    y_seq.append(majority_vote_int(window_y))

X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq, dtype=np.int64)

print("\n✅ Sequences created")
print("X_seq:", X_seq.shape)
print("y_seq:", y_seq.shape)

# =========================
# Train/Val split + class weights
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    X_seq, y_seq, test_size=0.2, random_state=RANDOM_SEED, stratify=y_seq
)

classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight = {int(c): float(w) for c, w in zip(classes, weights)}
print("\nClass weights:", list(class_weight.items())[:10], " ...")

# =========================
# Load model + fine-tune
# =========================
print("\nLoading base model:", BASE_MODEL_PATH)
model = tf.keras.models.load_model(BASE_MODEL_PATH)

# IMPORTANT: lower LR for fine-tuning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=OUT_MODEL_PATH,
        monitor="val_loss",
        save_best_only=True
    )
]

print("\n🚀 Fine-tuning...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

# Save history
pd.DataFrame(history.history).to_csv(OUT_HISTORY_CSV, index=False)

print("\n✅ Fine-tuned model saved to:", OUT_MODEL_PATH)
print("✅ History saved to:", OUT_HISTORY_CSV)
