# -*- coding: utf-8 -*-
import transformers
import pandas as pd
import glob
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, classification_report
import os
import re
import random
import numpy as np

print(f"Transformers version: {transformers.__version__}")

# =========================
# 0) Fixed test set (25 white + 25 black)
# =========================
test_files = [
    # 🟢 White (label 0)
    {"path": "csv_train/white/row_344_www.vestiairecollective.com__analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_309_www.olacabs.com__analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_288_bohdan-books.com_foreign-rights__srsltid=AfmBOoo0AHa-JgpbohQygIeESem-o1CvgWT46yJ5xnTg1tjzBiNEl8FT_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_73_www.tedsmontanagrill.com__gad_source=1&gad_campaignid=22799456150&gclid=Cj0KCQjwyvfDBhDYARIsAItzbZFc_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_119_www.temu.com__analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_325_www.idealista.com_en__analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_108_global.alipay.com_platform_site_ihome_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_331_wellhub.com_en-us__analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_41_deliveroo.co.uk__srsltid=AfmBOop31pV5I-74puyfh1xXhy4aty1MfULkpITIe88Z5nDBT4LTZfJ8_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_231_www.transartists.org_en_organisation_mann-ivanov-and-ferber_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_326_www.uniplaces.com__srsltid=AfmBOoq8qlcaS-NlDkkqpbh8uDaJGnSLVt3A5T-0wQlnulVRNhBoo9Zb_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_29_www.mcdonalds.co.th__analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_272_starylev.com.ua_about-us_srsltid=AfmBOooKl4ewpSh0cxnKn0BsyqiSgBXzTm2rgCCiRfwSMWrkh2vSFx4o_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_157_www.belpost.by_en_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_280_pocketbook.ch_en-ch_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_240_play.google.com_store_apps_dev_id=8427637182851887164&hl=en_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_43_justeattakeaway.com__analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_333_www.domestika.org_en_courses_popular_utm_source=google&utm_medium=cpc&utm_campaign=01_OTHERS_MIX_NA__analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_382_banco.bradesco_html_classic_index.shtm_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_88_sezzledealsnow.com__source=_2Bsezzle&gad_source=1&gad_campaignid=21705159098&gclid=Cj0KCQjwyvfDBhDYA_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_274_chytomo.com_en_the-ukrainian-old-lion-publishing-house-launches-print-on-demand-project__analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_4_www.thaiairways.com_en-th__analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_67_n26.com_en-eu_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_334_www.vendia.com_use-cases_data-monetization__utm_source=google&utm_medium=paid_search&utm_campaign=gl_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_233_www.mann-ivanov-ferber.com__analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_145_cdek-th.com_en_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_178_5ka.ru__utm_referrer=https_3a_2f_2fwww.google.com_2f_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_192_www.exlibrus.de__q=en_shop_book_list_publisher_Vagrius_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_210_rosman.ru_en__analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_256_www.ebay.com__analyzed.csv", "label": 0},

    # 🔴 Black (label 1)
    {"path": "csv_train/black/row_87_drive.usercontent.google.com_download_id=1nWsqALU5AljotDVZ4LzE3992QiaSSLDZ&export=download_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_246_flowing_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_110_blinq_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_147_sso-itrustcapetal_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_62_aff.redkong89.cc__ref=418zqoc3g7_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_240_bafybeielj_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_319_excduswale_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_312_wirett.weebly_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_107_webproxy_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_337_tech.aton40_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_161_kquinelogin_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_10_simplyconnect.polygonuk.com__analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_27_www.quickexchange.net__analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_221_one_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_283_sidneymccray_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_88_soanywbha.com__analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_105_docs.google_sentform_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_299_fluffy-paprenjak-2518cc_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_385_s.team-yh_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_255_liambiggs201_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_103_att-mail-109008_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_141_bpccsdf_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_288_scribehoe_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_321_eu.jotform_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_35_www.gopumpkin.fun_airdrop_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_190_yuh_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_214_cpcontacts_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_228_westfield_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_250_get_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_265_wk-drilling_analyzed.csv", "label": 1},
]

# =========================
# 1) Reproducibility
# =========================
random.seed(42)
np.random.seed(42)

# =========================
# 2) Cleaner
# =========================
def clean_text(text: str):
    text = re.sub(r'<.*?>', '', text)                  # Remove HTML tags
    text = re.sub(r'http\S+|www\S+', '', text)         # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)     # Remove special chars
    text = re.sub(r'\s+', ' ', text)                   # Normalize whitespace
    return text.lower().strip()

# =========================
# 3) Helpers to load train/test with exclusion
# =========================
def read_csv_as_text(file_path: str) -> str:
    # ลองอ่านด้วย pandas ถ้า fail ค่อย fallback เป็น open().read()
    try:
        df = pd.read_csv(file_path)
        # รวมทุกคอลัมน์เป็นข้อความเดียว (กันกรณีบางไฟล์คอลัมน์แรกว่าง)
        text = " ".join(df.astype(str).fillna("").agg(" ".join, axis=1).tolist())
        return text
    except Exception:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def load_fixed_test_data(test_list):
    test_data = []
    for item in test_list:
        path, label = item["path"], item["label"]
        try:
            raw = read_csv_as_text(path)
            cleaned = clean_text(raw)
            if cleaned:
                test_data.append({"text": cleaned, "label": label, "path": path})
                print(f"🧪 Test: {os.path.basename(path)} ({'white' if label==0 else 'black'})")
            else:
                print(f"⚠️ Empty after clean: {path}")
        except Exception as e:
            print(f"❌ Error reading test file {path}: {e}")
    return test_data

def load_train_excluding_tests(dir_path: str, label: int, exclude_set: set, limit: int):
    """
    โหลดไฟล์จาก dir_path โดย 'ตัด' ไฟล์ใน exclude_set ออก (ไม่เอา test มาปน)
    และจำกัดจำนวนสูงสุดที่ limit (เช่น 300)
    """
    all_files = sorted(glob.glob(os.path.join(dir_path, "*.csv")))
    files = [f for f in all_files if f not in exclude_set]
    random.shuffle(files)

    data_list = []
    for fp in files:
        if len(data_list) >= limit:
            break
        try:
            raw = read_csv_as_text(fp)
            cleaned = clean_text(raw)
            if cleaned:
                data_list.append({"text": cleaned, "label": label, "path": fp})
                print(f"✅ Train: {os.path.basename(fp)} ({'white' if label==0 else 'black'})")
        except Exception as e:
            print(f"❌ Error reading train file {fp}: {e}")

    if len(data_list) < limit:
        print(f"⚠️ Warning: requested {limit} but found only {len(data_list)} in {dir_path} (after excluding tests).")
    return data_list

# =========================
# 4) Paths & test exclusion
# =========================
path_white = "csv_train/white"
path_black = "csv_train/black"

# เซ็ต path เต็มของ test เพื่อตัดออกจาก train
test_full_paths = set(os.path.normpath(t["path"]) for t in test_files)

print("\n🧪 Loading FIXED test data from provided list...")
white_test = [t for t in test_files if t["label"] == 0]
black_test = [t for t in test_files if t["label"] == 1]
test_data_white = load_fixed_test_data(white_test)
test_data_black = load_fixed_test_data(black_test)
test_data = test_data_white + test_data_black

print(f"\n📊 Test Data Summary:")
print(f"   🟢 White test: {len(test_data_white)} files")
print(f"   🔴 Black test: {len(test_data_black)} files")
print(f"   📊 Total test: {len(test_data)} files")

# =========================
# 5) Load TRAIN 300/300 excluding test
# =========================
TRAIN_LIMIT_WHITE = 300
TRAIN_LIMIT_BLACK = 300

print("\n🟢 Loading whitelist TRAIN (300 files, excluding test)...")
train_white = load_train_excluding_tests(path_white, label=0, exclude_set=test_full_paths, limit=TRAIN_LIMIT_WHITE)

print("🔴 Loading blacklist TRAIN (300 files, excluding test)...")
train_black = load_train_excluding_tests(path_black, label=1, exclude_set=test_full_paths, limit=TRAIN_LIMIT_BLACK)

train_data = train_white + train_black

if not train_data:
    raise ValueError("No training data found after excluding tests. Check folders and filenames.")
if not test_data:
    raise ValueError("No test data found from provided test_files list.")

# =========================
# 6) DataFrames & Datasets
# =========================
train_df = pd.DataFrame(train_data)
test_df  = pd.DataFrame(test_data)

print("\n📊 Training Data Distribution:")
print(train_df["label"].value_counts())
print("\n📊 Test Data Distribution:")
print(test_df["label"].value_counts())

train_dataset = Dataset.from_pandas(train_df[["text","label"]])
test_dataset  = Dataset.from_pandas(test_df[["text","label"]])

print(f"\n✅ Datasets created:")
print(f"   🚀 Training dataset: {len(train_dataset)} examples")
print(f"   🧪 Test dataset: {len(test_dataset)} examples")

# =========================
# 7) Tokenization
# =========================
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

def preprocess(examples):
    texts = [str(t) if t is not None else "" for t in examples["text"]]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

print("\n🔄 Tokenizing datasets...")
train_tokenized = train_dataset.map(preprocess, batched=True)
test_tokenized  = test_dataset.map(preprocess, batched=True)

# =========================
# 8) Training
# =========================
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {'accuracy': accuracy_score(labels, predictions)}

args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    evaluation_strategy="epoch",
    learning_rate=1e-5,
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
print(f"📊 Training on {len(train_dataset)} examples (white={len(train_white)}, black={len(train_black)})")
print(f"🧪 Evaluating on {len(test_dataset)} examples (fixed provided list)")
trainer.train()
print("✅ Training completed.")

# =========================
# 9) Final evaluation
# =========================
print("\n🧪 Running final evaluation...")
eval_results = trainer.evaluate()
print(f"📊 Final Evaluation Results:")
for key, value in eval_results.items():
    try:
        print(f"   {key}: {value:.4f}")
    except Exception:
        print(f"   {key}: {value}")

# =========================
# 10) Detailed predictions
# =========================
print("\n🔍 Generating detailed predictions...")
predictions = trainer.predict(test_tokenized)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = test_df['label'].values

print("\n📊 Detailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Legitimate', 'Phishing']))

# =========================
# 11) Save fine-tuned model
# =========================
save_path = "phishing_llm_300_1_1"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"\n💾 Model saved to: {save_path}")

# =========================
# 12) Final summary
# =========================
print("\n" + "="*60)
print("📦 FINAL TRAINING REPORT")
print("="*60)
print(f"📊 Training Data:")
print(f"   🟢 Legitimate (white): {len(train_white)} examples")
print(f"   🔴 Phishing (black): {len(train_black)} examples")
print(f"   📊 Total training: {len(train_data)} examples")
print(f"\n🧪 Test Data (fixed list):")
print(f"   🟢 Legitimate (white): {len(test_data_white)} examples") 
print(f"   🔴 Phishing (black): {len(test_data_black)} examples")
print(f"   📊 Total test: {len(test_data)} examples")
print(f"\n🎯 Model Performance:")
print(f"   📊 Test Accuracy: {eval_results.get('eval_accuracy', 'N/A')}")
print(f"   💾 Model saved to: {save_path}")
print("="*60)
