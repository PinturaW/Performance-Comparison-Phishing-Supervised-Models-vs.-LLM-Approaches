import os
import re
import glob
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

# ---- Import guards ----
try:
    import transformers
except ModuleNotFoundError as e:
    raise SystemExit(
        "❌ Missing 'transformers'. ติดตั้งก่อนด้วย:\n"
        "/opt/homebrew/bin/python3.11 -m pip install transformers"
    ) from e

try:
    import torch  # noqa
except ModuleNotFoundError as e:
    raise SystemExit(
        "❌ Missing 'torch'. ติดตั้งก่อนด้วย:\n"
        "/opt/homebrew/bin/python3.11 -m pip install torch torchvision torchaudio"
    ) from e

try:
    from datasets import Dataset
except ModuleNotFoundError as e:
    raise SystemExit(
        "❌ Missing 'datasets'. ติดตั้งก่อนด้วย:\n"
        "/opt/homebrew/bin/python3.11 -m pip install datasets"
    ) from e

try:
    from sklearn.metrics import accuracy_score, classification_report
except ModuleNotFoundError as e:
    raise SystemExit(
        "❌ Missing 'scikit-learn'. ติดตั้งก่อนด้วย:\n"
        "/opt/homebrew/bin/python3.11 -m pip install scikit-learn"
    ) from e

from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    TrainingArguments,
    Trainer,
)

print(f"Transformers version: {transformers.__version__}")

# ====== 0. Configs ======
TRAIN_START, TRAIN_END = 1, 350
TEST_START,  TEST_END  = 351, 400

PATH_WHITE = 'csv_train_new/white'
PATH_BLACK = 'csv_train_new/black'

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# ====== 1. Clean text ======
def clean_text(text: str) -> str:
    text = re.sub(r'<.*?>', '', text)                 # Remove HTML tags
    text = re.sub(r'http\S+|www\S+', '', text)        # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', ' ', text)   # Keep basic chars
    text = re.sub(r'\s+', ' ', text)                  # Normalize whitespace
    return text.lower().strip()

# ====== 2. Utilities for row ordering ======
_row_pat = re.compile(r'row[^0-9]*?(\d+)', re.IGNORECASE)

def extract_row_index(path: str) -> int:
    """
    ดึงเลข row จากชื่อไฟล์ เช่น 'row_123_domain_analyzed.csv' -> 123
    ถ้าไม่เจอ ให้เป็นค่ามาก ๆ เพื่อดันไปท้าย (เช่น 10^9)
    """
    base = os.path.basename(path)
    m = _row_pat.search(base)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return 10**9

def list_files_sorted_by_row(folder: str) -> List[str]:
    files = glob.glob(os.path.join(folder, '*.csv'))
    files.sort(key=lambda p: (extract_row_index(p), os.path.basename(p).lower()))
    return files

def debug_missing_rows(loaded_files: List[str], start_row: int, end_row: int) -> List[int]:
    present = {extract_row_index(fp) for fp in loaded_files}
    target = set(range(start_row, end_row + 1))
    missing = sorted([r for r in target if r not in present])
    return missing

# ====== 3. Load data by row range ======
def read_data_string_csv(filename: str) -> Optional[str]:
    """
    อ่านคอลัมน์ data_string แถวแรกจากไฟล์ CSV (1 แถว/ไฟล์)
    """
    try:
        df = pd.read_csv(filename, encoding='utf-8')
        if 'data_string' in df.columns and len(df) > 0:
            return str(df.loc[0, 'data_string'])
        # fallback (กันไว้)
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")
    return None

def load_data_by_row_range(path: str, label: int, start_row: int, end_row: int
                           ) -> Tuple[List[Dict], List[str]]:
    """
    โหลดข้อมูลจากไฟล์ที่เรียงตามเลข row แล้วเลือกเฉพาะช่วง [start_row, end_row]
    คืนค่า:
      - data: list[{'text': cleaned_text, 'label': label}]
      - used_files: รายชื่อไฟล์ที่ 'ถูกใช้จริง' (มีเนื้อหา)
    """
    files = list_files_sorted_by_row(path)
    # เฉพาะไฟล์ที่อยู่ในช่วง row
    target_files = [fp for fp in files if start_row <= extract_row_index(fp) <= end_row]

    data: List[Dict] = []
    used_files: List[str] = []

    for fp in target_files:
        txt = read_data_string_csv(fp)
        if not txt:
            continue
        cleaned = clean_text(txt)
        if not cleaned:
            continue
        data.append({'text': cleaned, 'label': label})
        used_files.append(fp)

    print(f"✅ Loaded {len(data)} items from {path} (rows {start_row}-{end_row})")
    if used_files:
        print(f"   e.g. first 3: {[os.path.basename(x) for x in used_files[:3]]}")
    return data, used_files

def write_manifest(split: str, label_name: str, files: List[str]):
    """
    เขียนรายชื่อไฟล์ที่ใช้จริงลง logs/ และต่อรวมใน manifest รวม
    """
    out_txt = os.path.join(LOG_DIR, f"{split}_{label_name}_files.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        for fp in sorted(files, key=lambda p: (extract_row_index(p), os.path.basename(p).lower())):
            f.write(f"{os.path.basename(fp)}\n")
    print(f"📝 Wrote {len(files)} file names → {out_txt}")

# ====== 4. Load train/test deterministically by row ranges (+manifests) ======
print("🟢 Loading whitelist TRAIN rows 1–350 ...")
white_train, white_train_files = load_data_by_row_range(PATH_WHITE, label=0, start_row=TRAIN_START, end_row=TRAIN_END)
write_manifest("train", "white", white_train_files)
miss_white_train = debug_missing_rows(white_train_files, TRAIN_START, TRAIN_END)
if miss_white_train:
    print(f"⚠️ Missing white TRAIN rows ({TRAIN_START}-{TRAIN_END}): {miss_white_train[:40]}{' ...' if len(miss_white_train)>40 else ''}")
else:
    print("✅ No missing white TRAIN rows")

print("🔴 Loading blacklist TRAIN rows 1–350 ...")
black_train, black_train_files = load_data_by_row_range(PATH_BLACK, label=1, start_row=TRAIN_START, end_row=TRAIN_END)
write_manifest("train", "black", black_train_files)
miss_black_train = debug_missing_rows(black_train_files, TRAIN_START, TRAIN_END)
if miss_black_train:
    print(f"⚠️ Missing black TRAIN rows ({TRAIN_START}-{TRAIN_END}): {miss_black_train[:40]}{' ...' if len(miss_black_train)>40 else ''}")
else:
    print("✅ No missing black TRAIN rows")

print("\n🧪 Loading whitelist TEST rows 351–400 ...")
white_test, white_test_files = load_data_by_row_range(PATH_WHITE, label=0, start_row=TEST_START, end_row=TEST_END)
write_manifest("test", "white", white_test_files)
miss_white_test = debug_missing_rows(white_test_files, TEST_START, TEST_END)
if miss_white_test:
    print(f"⚠️ Missing white TEST rows ({TEST_START}-{TEST_END}): {miss_white_test}")
else:
    print("✅ No missing white TEST rows")

print("🧪 Loading blacklist TEST rows 351–400 ...")
black_test, black_test_files = load_data_by_row_range(PATH_BLACK, label=1, start_row=TEST_START, end_row=TEST_END)
write_manifest("test", "black", black_test_files)
miss_black_test = debug_missing_rows(black_test_files, TEST_START, TEST_END)
if miss_black_test:
    print(f"⚠️ Missing black TEST rows ({TEST_START}-{TEST_END}): {miss_black_test}")
else:
    print("✅ No missing black TEST rows")

# รวม manifest ทั้งหมดเป็นไฟล์เดียว (CSV)
manifest_rows = []
def add_to_manifest(split, label_name, files):
    for fp in files:
        manifest_rows.append({
            "split": split,
            "label": label_name,
            "filename": os.path.basename(fp),
            "row": extract_row_index(fp),
        })
add_to_manifest("train", "white", white_train_files)
add_to_manifest("train", "black", black_train_files)
add_to_manifest("test",  "white", white_test_files)
add_to_manifest("test",  "black", black_test_files)

manifest_df = pd.DataFrame(manifest_rows).sort_values(by=["split","label","row","filename"])
manifest_path = os.path.join(LOG_DIR, "manifest_all_splits.csv")
manifest_df.to_csv(manifest_path, index=False, encoding="utf-8")
print(f"📄 Combined manifest → {manifest_path}")

# Combine
train_data = white_train + black_train
test_data  = white_test + black_test

if not train_data:
    raise ValueError("No training data found. Check row ranges and csv_train_new folders.")
if not test_data:
    raise ValueError("No test data found. Check row ranges and csv_train_new folders.")

# ====== 5. Build datasets ======
train_df = pd.DataFrame(train_data)
test_df  = pd.DataFrame(test_data)

print("\n📊 Training Data Distribution:")
print(train_df['label'].value_counts())
print("\n📊 Test Data Distribution:")
print(test_df['label'].value_counts())

train_dataset = Dataset.from_pandas(train_df)
test_dataset  = Dataset.from_pandas(test_df)

print(f"\n✅ Datasets created:")
print(f"   🚀 Training dataset: {len(train_dataset)} examples")
print(f"   🧪 Test dataset: {len(test_dataset)} examples")

# ====== 6. Tokenization & Model ======
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

def preprocess(examples):
    texts = [str(t) if t is not None else "" for t in examples["text"]]
    enc = tokenizer(texts, truncation=True, padding="max_length", max_length=512)
    enc["labels"] = examples["label"]  # สำคัญให้ Trainer เห็น labels
    return enc

print("\n🔄 Tokenizing datasets...")
train_tokenized = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
test_tokenized  = test_dataset.map(preprocess,  batched=True, remove_columns=test_dataset.column_names)

# ====== 7. Training ======
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    eval_strategy="epoch",
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_strategy="epoch",
    warmup_steps=100,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    compute_metrics=compute_metrics
)

print("\n🚀 Starting model training...")
print(f"📊 Training on {len(train_data)} examples, evaluating on {len(test_data)} examples")
trainer.train()
print("✅ Training completed.")

# ====== 8. Evaluation ======
print("\n🧪 Running final evaluation...")
eval_results = trainer.evaluate()
print(f"📊 Final Evaluation Results:")
for key, value in eval_results.items():
    try:
        print(f"   {key}: {float(value):.4f}")
    except Exception:
        print(f"   {key}: {value}")

# ====== 9. Detailed predictions ======
print("\n🔍 Generating detailed predictions...")
predictions = trainer.predict(test_tokenized)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = test_df['label'].values
print("\n📊 Detailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Not Phishing', 'Phishing']))

# ====== 10. Save model ======
save_path = "phishing_train_10"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"\n💾 Model saved to: {save_path}")

# ====== 11. Summary ======
print("\n" + "="*60)
print("📦 FINAL TRAINING REPORT")
print("="*60)
print(f"📊 Training Data:")
print(f"   🟢 Not Phishing (white): {len(white_train_files)} files used")
print(f"   🔴 Phishing (black): {len(black_train_files)} files used")
print(f"   📊 Total training: {len(train_data)} examples")
print(f"\n🧪 Test Data:")
print(f"   🟢 Not Phishing (white): {len(white_test_files)} files used")
print(f"   🔴 Phishing (black): {len(black_test_files)} files used")
print(f"   📊 Total test: {len(test_data)} examples")
if miss_white_train or miss_black_train or miss_white_test or miss_black_test:
    print("\n⚠️ Missing rows summary:")
    if miss_white_train: print(f"   white TRAIN missing rows: {miss_white_train[:40]}{' ...' if len(miss_white_train)>40 else ''}")
    if miss_black_train: print(f"   black TRAIN missing rows: {miss_black_train[:40]}{' ...' if len(miss_black_train)>40 else ''}")
    if miss_white_test:  print(f"   white TEST missing rows: {miss_white_test}")
    if miss_black_test:  print(f"   black TEST missing rows: {miss_black_test}")
print(f"\n📝 See detailed file lists in: {LOG_DIR} (TXT manifests) and {manifest_path}")
print("="*60)
