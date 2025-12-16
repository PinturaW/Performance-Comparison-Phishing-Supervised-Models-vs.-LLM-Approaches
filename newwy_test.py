# predict_351_400.py
import os
import re
import glob
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============== CONFIG ==============
MODEL_PATH = "./phishing_train_3"
WHITE_DIR  = "csv_train_new/white"
BLACK_DIR  = "csv_train_new/black"
START_ROW, END_ROW = 351, 400
OUT_DIR = "predictions"
os.makedirs(OUT_DIR, exist_ok=True)

# ============== DEVICE ==============
device = torch.device("cpu")  # ตามที่ต้องการ
print(f"🖥️  Using device: {device}")

# ============== MODEL ===============
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device).eval()

# ============ HELPERS ==============
_row_pat = re.compile(r'row[^0-9]*?(\d+)', re.IGNORECASE)

def extract_row_index(path):
    base = os.path.basename(path)
    m = _row_pat.search(base)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return 10**9

def list_files_sorted_by_row(folder):
    files = glob.glob(os.path.join(folder, "*.csv"))
    files.sort(key=lambda p: (extract_row_index(p), os.path.basename(p).lower()))
    return files

def read_data_string_csv(filename):
    try:
        df = pd.read_csv(filename, encoding="utf-8")
        if "data_string" in df.columns and len(df) > 0:
            return str(df.loc[0, "data_string"])
        # fallback เผื่อไฟล์ไม่ตรงรูปแบบ
        with open(filename, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error reading {os.path.basename(filename)}: {e}")
        return ""

def clean_text(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

def extract_url_from_data_string(s):
    m = re.search(r"^URL:\s*(.+)$", s, flags=re.M)
    return m.group(1).strip() if m else ""

@torch.no_grad()
def predict_with_score(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    enc = {k: v.to(device) for k, v in enc.items()}
    outputs = model(**enc)
    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    return pred, float(probs[1]), float(probs[0])  # (pred, phishing_prob, legit_prob)

def files_in_range(folder):
    all_files = list_files_sorted_by_row(folder)
    picked = [fp for fp in all_files if START_ROW <= extract_row_index(fp) <= END_ROW]
    have_rows = {extract_row_index(fp) for fp in picked}
    need_rows = set(range(START_ROW, END_ROW + 1))
    missing = sorted(list(need_rows - have_rows))
    return picked, missing

def run_split(files, label_id, label_name):
    rows = []
    correct = 0
    total = 0
    for fp in files:
        base = os.path.basename(fp)
        rix  = extract_row_index(fp)
        raw  = read_data_string_csv(fp)
        if not raw.strip():
            print(f"⚠️ Empty file: {base}")
            continue
        url  = extract_url_from_data_string(raw)
        text = clean_text(raw)

        pred, phish, legit = predict_with_score(text)
        ok = int(pred == label_id)
        correct += ok
        total += 1

        print(f"🧪 row {rix:>3} | {base}")
        print(f"   ➤ Pred: {'Phishing ⚠️' if pred == 1 else 'Legitimate ✅'} "
              f"(phish={phish:.2%}, legit={legit:.2%}) | Label: {label_name}\n")

        rows.append({
            "filename": base,
            "row": rix,
            "url": url,
            "label": label_id,
            "label_name": label_name,
            "pred": pred,
            "pred_name": "Phishing" if pred == 1 else "Legitimate",
            "phishing_prob": phish,
            "legit_prob": legit,
            "correct": ok,
        })
    df = pd.DataFrame(rows).sort_values(["row", "filename"])
    return df, correct, total

# ============== LOAD LISTS ==============
print(f"📝 Selecting files rows {START_ROW}-{END_ROW}")
white_files, miss_w = files_in_range(WHITE_DIR)
black_files, miss_b = files_in_range(BLACK_DIR)
print(f"   🟢 White: {len(white_files)} files (missing rows: {miss_w if miss_w else 'None'})")
print(f"   🔴 Black: {len(black_files)} files (missing rows: {miss_b if miss_b else 'None'})\n")

# ============== PREDICT ==============
print("🔮 Predict WHITE")
white_df, c_w, t_w = run_split(white_files, 0, "Legitimate")
print("🔮 Predict BLACK")
black_df, c_b, t_b = run_split(black_files, 1, "Phishing")

# ============== SAVE REPORT ==============
white_out = os.path.join(OUT_DIR, f"white_test_{START_ROW}_{END_ROW}.csv")
black_out = os.path.join(OUT_DIR, f"black_test_{START_ROW}_{END_ROW}.csv")
all_out   = os.path.join(OUT_DIR, f"all_test_{START_ROW}_{END_ROW}.csv")

white_df.to_csv(white_out, index=False, encoding="utf-8")
black_df.to_csv(black_out, index=False, encoding="utf-8")
pd.concat([white_df, black_df], ignore_index=True).to_csv(all_out, index=False, encoding="utf-8")

print("\n💾 Saved:")
print(f"   • {white_out}")
print(f"   • {black_out}")
print(f"   • {all_out}")

# ============== SUMMARY ==============
def acc_str(c, t): return f"{(c/t):.2%}" if t else "N/A"
print("\n" + "="*60)
print("📊 TEST SUMMARY")
print("="*60)
print(f"🟢 White  : {t_w:3d} files | Accuracy = {acc_str(c_w, t_w)}")
print(f"🔴 Black  : {t_b:3d} files | Accuracy = {acc_str(c_b, t_b)}")
print(f"🌐 Overall: {t_w + t_b:3d} files | Accuracy = {acc_str(c_w + c_b, t_w + t_b)}")
if miss_w or miss_b:
    print("\n⚠️ Missing rows:")
    if miss_w: print(f"   white: {miss_w}")
    if miss_b: print(f"   black: {miss_b}")
print("="*60)