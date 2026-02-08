import os
import random
import pandas as pd

DATA_DIR = r"D:\Grad\Dataset"
FEATURE_COLS_PATH = r"D:\Grad\derived_dataset\feature_columns.txt"
OUT_DIR = r"D:\Grad\deployment"
OUT_FILE = os.path.join(OUT_DIR, "traffic_stream_500_one_attack_twice.csv")

TOTAL_ROWS = 500
ATTACK_REPEAT = 2
BENIGN_ROWS = TOTAL_ROWS - ATTACK_REPEAT
RANDOM_SEED = 42

# Make it chunky: Benign -> Attack -> Benign -> Attack -> Benign
# These control where the 2 attack rows appear (randomly, but not near edges)
EDGE_PADDING = max(1, min(50, TOTAL_ROWS // 10))   # for 500 => 50
MIN_GAP = max(1, min(50, TOTAL_ROWS // 10))        # for 500 => 50

os.makedirs(OUT_DIR, exist_ok=True)

with open(FEATURE_COLS_PATH, "r", encoding="utf-8") as f:
    feature_cols = [line.strip() for line in f if line.strip()]

def infer_class_name(filename: str) -> str:
    name = filename
    for suffix in ["_test.pcap.csv", "_test.csv", ".pcap.csv", ".csv"]:
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    if name.lower().endswith("_test"):
        name = name[:-5]
    if name.lower().startswith("benign"):
        return "Benign"
    return name

rng = random.Random(RANDOM_SEED)

# -------------------------
# Collect *_test*.csv files
# -------------------------
test_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv") and "_test" in f.lower()]
if not test_files:
    raise SystemExit("No *_test*.csv files found in DATA_DIR.")

benign_files = [f for f in test_files if infer_class_name(f) == "Benign"]
attack_files  = [f for f in test_files if infer_class_name(f) != "Benign"]

if not benign_files:
    raise SystemExit("No Benign *_test*.csv file found.")
if not attack_files:
    raise SystemExit("No attack *_test*.csv files found.")

# -------------------------
# Load benign pool (need 498 rows)
# -------------------------
benign_path = os.path.join(DATA_DIR, benign_files[0])
benign_df = pd.read_csv(benign_path)

missing = [c for c in feature_cols if c not in benign_df.columns]
if missing:
    raise ValueError(f"Benign file missing columns: {missing[:10]} ... total missing={len(missing)}")

benign_df = benign_df[feature_cols].copy()
benign_df["Label"] = "Benign"

if len(benign_df) < BENIGN_ROWS:
    raise SystemExit(f"Benign file has only {len(benign_df)} rows, need {BENIGN_ROWS}.")

benign_pool = benign_df.sample(n=BENIGN_ROWS, random_state=RANDOM_SEED).reset_index(drop=True)

# -------------------------
# Choose ONE attack label and sample 2 rows from it
# -------------------------
attack_file = rng.choice(sorted(attack_files))
attack_label = infer_class_name(attack_file)

attack_path = os.path.join(DATA_DIR, attack_file)
attack_df = pd.read_csv(attack_path)

missing = [c for c in feature_cols if c not in attack_df.columns]
if missing:
    raise ValueError(f"Attack file {attack_file} missing columns: {missing[:10]} ... total missing={len(missing)}")

attack_df = attack_df[feature_cols].copy()
attack_df["Label"] = attack_label

if len(attack_df) < ATTACK_REPEAT:
    raise SystemExit(f"Attack file {attack_file} has only {len(attack_df)} rows, need {ATTACK_REPEAT}.")

attack_pool = attack_df.sample(n=ATTACK_REPEAT, random_state=RANDOM_SEED + 7).reset_index(drop=True)

# -------------------------
# Pick 2 attack positions (chunky)
# stream indices are 0..TOTAL_ROWS-1
# attacks are at p1 and p2 (p2 > p1 + MIN_GAP)
# -------------------------
if TOTAL_ROWS < (2 * EDGE_PADDING + MIN_GAP + 2):
    raise SystemExit("TOTAL_ROWS too small for chosen EDGE_PADDING/MIN_GAP. Reduce them or increase TOTAL_ROWS.")

p1_low = EDGE_PADDING
p1_high = TOTAL_ROWS - EDGE_PADDING - MIN_GAP - 2
p1 = rng.randint(p1_low, p1_high)

p2_low = p1 + MIN_GAP + 1
p2_high = TOTAL_ROWS - EDGE_PADDING - 1
p2 = rng.randint(p2_low, p2_high)

# segment benign sizes
b1 = p1
b2 = (p2 - p1 - 1)
b3 = (TOTAL_ROWS - p2 - 1)

# b1+b2+b3 must equal BENIGN_ROWS
if (b1 + b2 + b3) != BENIGN_ROWS:
    raise SystemExit(f"Segment mismatch: b1+b2+b3={b1+b2+b3} but BENIGN_ROWS={BENIGN_ROWS}")

# -------------------------
# Build stream: Benign(b1) -> Attack -> Benign(b2) -> Attack -> Benign(b3)
# -------------------------
parts = []
parts.append(benign_pool.iloc[0:b1])
parts.append(attack_pool.iloc[0:1])
parts.append(benign_pool.iloc[b1:b1+b2])
parts.append(attack_pool.iloc[1:2])
parts.append(benign_pool.iloc[b1+b2:b1+b2+b3])

stream_df = pd.concat(parts, ignore_index=True)

# Safety checks
attack_count = int((stream_df["Label"] == attack_label).sum())
benign_count = int((stream_df["Label"] == "Benign").sum())

if len(stream_df) != TOTAL_ROWS or attack_count != ATTACK_REPEAT or benign_count != BENIGN_ROWS:
    raise SystemExit(
        f"Build check failed: rows={len(stream_df)} benign={benign_count} {attack_label}={attack_count}"
    )

stream_df.to_csv(OUT_FILE, index=False)

print(f"✅ Created {TOTAL_ROWS}-row stream: {OUT_FILE}")
print(f"Attack chosen: {attack_label} (appears {attack_count} times)")
print(f"Attack positions (0-based): p1={p1}, p2={p2}")
print("Label distribution:")
print(stream_df["Label"].value_counts())
