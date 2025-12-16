#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate phishing_llm_newwy models (_1, _3, _5) on a labeled CSV set
and print precision/recall/F1/support for:
- Non-Phishing
- Phishing
- accuracy
- macro avg
- weighted avg

It scans two folders:
  WHITE_DIR  -> label 0 (Non-Phishing / Legitimate)
  BLACK_DIR  -> label 1 (Phishing)

By default, restricts to rows 351..400 (parsed from filenames like 'row_351_...csv').
Set START_ROW/END_ROW = None to evaluate all files.

Outputs:
- Prints metrics to stdout
- Saves metrics to ./predictions/metrics_model_{id}.csv
- Saves detailed predictions to ./predictions/pred_details_model_{id}.csv
"""

import os
import re
import glob
import torch
import pandas as pd
import numpy as np
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report

# ================== CONFIG ==================
MODEL_TEMPLATE = "./phishing_llm_newwy_{}"      # will evaluate ids [1,3,5]
EVAL_MODEL_IDS = [1, 3, 5]                      # skip 2 as requested
WHITE_DIR  = "csv_train_new/white"              # label 0
BLACK_DIR  = "csv_train_new/black"              # label 1

# Row filtering (inclusive); set to None to disable and use all files
START_ROW = 351
END_ROW   = 400

OUT_DIR = "predictions"
os.makedirs(OUT_DIR, exist_ok=True)

# ================== DEVICE ==================
device = torch.device("cpu")
print(f"🖥️  Using device: {device}")

# =============== HELPERS ====================
_row_pat = re.compile(r'row[^0-9]*?(\d+)', re.IGNORECASE)

def extract_row_index(path: str) -> int:
    base = os.path.basename(path)
    m = _row_pat.search(base)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    # put non-matching to the end
    return 10**9

def list_files_sorted_by_row(folder: str) -> List[str]:
    files = glob.glob(os.path.join(folder, "*.csv"))
    files.sort(key=lambda p: (extract_row_index(p), os.path.basename(p).lower()))
    return files

def files_in_range(folder: str, start: int = None, end: int = None) -> Tuple[List[str], List[int]]:
    all_files = list_files_sorted_by_row(folder)
    if start is None or end is None:
        return all_files, []
    picked = [fp for fp in all_files if start <= extract_row_index(fp) <= end]
    have_rows = {extract_row_index(fp) for fp in picked if extract_row_index(fp) != 10**9}
    need_rows = set(range(start, end + 1))
    missing = sorted(list(need_rows - have_rows))
    return picked, missing

def read_data_string_csv(filename: str) -> str:
    try:
        df = pd.read_csv(filename, encoding="utf-8")
        if "data_string" in df.columns and len(df) > 0:
            return str(df.loc[0, "data_string"])
        # fallback: read whole file text
        with open(filename, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error reading {os.path.basename(filename)}: {e}")
        return ""

def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

@torch.no_grad()
def predict_with_score(model, tokenizer, text: str):
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    enc = {k: v.to(device) for k, v in enc.items()}
    outputs = model(**enc)
    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    return pred, float(probs[1]), float(probs[0])  # (pred, phishing_prob, legit_prob)

def gather_files() -> Tuple[list, list]:
    """Return ([(filepath,label), ...], missing_rows_list)."""
    if START_ROW is not None and END_ROW is not None:
        white_files, miss_w = files_in_range(WHITE_DIR, START_ROW, END_ROW)
        black_files, miss_b = files_in_range(BLACK_DIR, START_ROW, END_ROW)
    else:
        white_files, miss_w = files_in_range(WHITE_DIR)
        black_files, miss_b = files_in_range(BLACK_DIR)
    all_files = [(fp, 0) for fp in white_files] + [(fp, 1) for fp in black_files]
    return all_files, sorted(set(miss_w + miss_b))

def evaluate_model(model_id: int):
    model_path = MODEL_TEMPLATE.format(model_id)
    print("\n" + "="*70)
    print(f"🚀 Evaluating model: {os.path.basename(model_path)}")
    print("="*70)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()

    pairs, missing_rows = gather_files()
    y_true, y_pred = [], []
    detailed_rows = []

    for fp, label in pairs:
        raw = read_data_string_csv(fp)
        if not raw.strip():
            continue
        text = clean_text(raw)
        pred, phish_p, legit_p = predict_with_score(model, tokenizer, text)
        y_true.append(label)
        y_pred.append(pred)
        detailed_rows.append({
            "filename": os.path.basename(fp),
            "row": extract_row_index(fp),
            "label": label,
            "pred": pred,
            "phishing_prob": phish_p,
            "legit_prob": legit_p,
            "correct": int(label == pred),
        })

    target_names = ["Non-Phishing", "Phishing"]
    report = classification_report(
        y_true, y_pred, target_names=target_names, digits=2, output_dict=True
    )

    def pp_row(name, r):
        return f"{name:15s}\t{r.get('precision', float('nan')):.2f}\t{r.get('recall', float('nan')):.2f}\t{r.get('f1-score', float('nan')):.2f}\t{int(r.get('support', 0))}"

    # ---- Print in your requested layout ----
    print("\nprecision\trecall\tf1-score\tsupport")
    print(pp_row("Non-Phishing", report.get("Non-Phishing", {})))
    print(pp_row("Phishing",     report.get("Phishing", {})))
    print(f"accuracy\t\t\t{report.get('accuracy', float('nan')):.2f}\t{int(sum(report[k]['support'] for k in target_names))}")
    print(pp_row("macro avg",    report.get("macro avg", {})))
    print(pp_row("weighted avg", report.get("weighted avg", {})))

    # ---- Save metrics and details to CSVs (optional but handy) ----
    metrics_path = os.path.join(OUT_DIR, f"metrics_model_{model_id}.csv")
    details_path = os.path.join(OUT_DIR, f"pred_details_model_{model_id}.csv")

    rows = []
    for key in ["Non-Phishing", "Phishing", "macro avg", "weighted avg"]:
        r = report.get(key, {})
        rows.append({
            "label": key,
            "precision": r.get("precision", np.nan),
            "recall": r.get("recall", np.nan),
            "f1-score": r.get("f1-score", np.nan),
            "support": r.get("support", 0 if key != "accuracy" else ""),
        })
    rows.append({
        "label": "accuracy",
        "precision": "",
        "recall": "",
        "f1-score": report.get("accuracy", np.nan),
        "support": int(sum(report[k]["support"] for k in target_names)),
    })
    pd.DataFrame(rows).to_csv(metrics_path, index=False, encoding="utf-8")
    pd.DataFrame(detailed_rows).sort_values(["row", "filename"]).to_csv(details_path, index=False, encoding="utf-8")

    print(f"\n💾 Saved: {metrics_path}")
    print(f"💾 Saved: {details_path}")
    if START_ROW is not None and END_ROW is not None:
        print(f"⚠️ Missing rows in range {START_ROW}-{END_ROW}: {missing_rows if missing_rows else 'None'}")

def main():
    # Header preview of what will be evaluated
    if START_ROW is not None and END_ROW is not None:
        white_files, miss_w = files_in_range(WHITE_DIR, START_ROW, END_ROW)
        black_files, miss_b = files_in_range(BLACK_DIR, START_ROW, END_ROW)
        print(f"📝 Evaluating rows {START_ROW}-{END_ROW}")
        print(f"   🟢 White dir: {WHITE_DIR} -> {len(white_files)} files (missing: {miss_w})")
        print(f"   🔴 Black dir: {BLACK_DIR} -> {len(black_files)} files (missing: {miss_b})")
    else:
        white_files, _ = files_in_range(WHITE_DIR)
        black_files, _ = files_in_range(BLACK_DIR)
        print("📝 Evaluating ALL rows")
        print(f"   🟢 White dir: {WHITE_DIR} -> {len(white_files)} files")
        print(f"   🔴 Black dir: {BLACK_DIR} -> {len(black_files)} files")

    for mid in EVAL_MODEL_IDS:
        evaluate_model(mid)

if __name__ == "__main__":
    main()
