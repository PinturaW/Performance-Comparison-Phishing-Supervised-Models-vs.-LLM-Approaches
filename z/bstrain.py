import transformers
import pandas as pd
import glob
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import os
import re
from sklearn.model_selection import train_test_split
import torch

device = torch.device("cpu")
model.to(device)

print(f"Transformers version: {transformers.__version__}")

# ====== 1. Clean text function for string format data ======
def clean_text(text):
    """ทำความสะอาด text สำหรับข้อมูล string format"""
    if text is None or text == "":
        return ""
    
    text = str(text)
    
    # แปลง newlines เป็น spaces เพื่อให้ BERT อ่านได้ดีขึ้น
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    
    # ลบ HTML tags หากมี
    text = re.sub(r'<.*?>', '', text)
    
    # ลบ URLs ที่เหลืออยู่
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # เก็บตัวอักษร ตัวเลข และ punctuation ที่สำคัญ
    text = re.sub(r'[^\w\s.,!?:;-]', ' ', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# ====== 2. Load data from CSV files ======
def load_csv_data(file_path):
    """โหลดข้อมูลจาก CSV และดึง data_string column"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        if 'data_string' in df.columns:
            # ดึงข้อมูลจาก data_string column
            return df['data_string'].iloc[0] if len(df) > 0 else ""
        else:
            # ถ้าไม่มี data_string column ให้อ่านทั้งหมดเป็น text
            return df.to_string()
    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")
        return ""

path_white = 'csv/white'
path_black = 'csv/black'

# ตรวจสอบว่า folders มีอยู่หรือไม่
if not os.path.exists(path_white):
    print(f"❌ White folder not found: {path_white}")
if not os.path.exists(path_black):
    print(f"❌ Black folder not found: {path_black}")

all_files_white = glob.glob(os.path.join(path_white, '*.csv'))
all_files_black = glob.glob(os.path.join(path_black, '*.csv'))

data_list = []

print("🟢 Reading whitelist files...")
for filename in all_files_white:
    content = load_csv_data(filename)
    if content:
        cleaned = clean_text(content)
        if len(cleaned) > 10:  # เช็คว่ามีข้อมูลเพียงพอ
            data_list.append({
                'text': cleaned, 
                'label': 0,
                'source': os.path.basename(filename)
            })
            print(f"✅ {os.path.basename(filename)} (white) - {len(cleaned)} chars")
        else:
            print(f"⚠️  {os.path.basename(filename)} (white) - too short, skipped")

print("\n🔴 Reading blacklist files...")
for filename in all_files_black:
    content = load_csv_data(filename)
    if content:
        cleaned = clean_text(content)
        if len(cleaned) > 10:  # เช็คว่ามีข้อมูลเพียงพอ
            data_list.append({
                'text': cleaned, 
                'label': 1,
                'source': os.path.basename(filename)
            })
            print(f"✅ {os.path.basename(filename)} (black) - {len(cleaned)} chars")
        else:
            print(f"⚠️  {os.path.basename(filename)} (black) - too short, skipped")

if not data_list:
    raise ValueError("❌ No valid data found. Check csv folders and data format.")

# สร้าง DataFrame และแสดงสถิติ
combined_df = pd.DataFrame(data_list)

print(f"\n📊 Dataset Statistics:")
print(f"   • Total samples: {len(combined_df)}")
print(f"   • White samples: {len(combined_df[combined_df['label'] == 0])}")
print(f"   • Black samples: {len(combined_df[combined_df['label'] == 1])}")
print(f"   • Average text length: {combined_df['text'].str.len().mean():.0f} chars")
print(f"   • Max text length: {combined_df['text'].str.len().max()} chars")
print(f"   • Min text length: {combined_df['text'].str.len().min()} chars")

# แสดงตัวอย่างข้อมูล
print(f"\n📋 Sample data:")
for i, row in combined_df.head(2).iterrows():
    print(f"   {i+1}. {row['source']} (label: {row['label']})")
    print(f"      Text preview: {row['text'][:100]}...")

# ====== 3. Split data into train/validation ======
train_df, val_df = train_test_split(
    combined_df, 
    test_size=0.2, 
    stratify=combined_df['label'], 
    random_state=42
)

print(f"\n🔄 Data Split:")
print(f"   • Training samples: {len(train_df)}")
print(f"   • Validation samples: {len(val_df)}")

# สร้าง Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
val_dataset = Dataset.from_pandas(val_df[['text', 'label']])

print(f"\n✅ Hugging Face datasets created:")
print(f"   • Train: {train_dataset}")
print(f"   • Validation: {val_dataset}")

# ====== 4. Tokenization ======
model_name = "bert-base-uncased"
print(f"\n🤖 Loading model: {model_name}")

tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

def preprocess(examples):
    """Tokenize text with proper handling for long sequences"""
    texts = [str(t) if t is not None else "" for t in examples["text"]]
    
    # ใช้ max_length ที่เหมาะสมกับข้อมูล
    return tokenizer(
        texts, 
        truncation=True, 
        padding="max_length", 
        max_length=512,  # BERT's max length
        return_tensors=None
    )

print("🔧 Tokenizing datasets...")
train_tokenized = train_dataset.map(preprocess, batched=True)
val_tokenized = val_dataset.map(preprocess, batched=True)

print("✅ Tokenization completed")

# ====== 5. Training Arguments ======
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,  # เพิ่ม batch size
    per_device_eval_batch_size=8,
    num_train_epochs=5,  # เพิ่ม epochs
    evaluation_strategy="steps",  # evaluate ทุก steps
    eval_steps=50,  # evaluate ทุก 50 steps
    save_strategy="steps",
    save_steps=100,
    learning_rate=2e-5,  # learning rate ที่เหมาะสม
    weight_decay=0.01,  # regularization
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    warmup_steps=100,  # warmup
    report_to=None  # ไม่ส่งไป wandb
)

# ====== 6. Trainer ======
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=val_tokenized,
    tokenizer=tokenizer
)

print("\n🚀 Starting model training...")
print("=" * 50)

# เริ่ม training
trainer.train()

print("\n✅ Training completed!")

# ====== 7. Evaluation ======
print("\n📊 Evaluating model...")
eval_results = trainer.evaluate()

print(f"📈 Evaluation Results:")
for key, value in eval_results.items():
    print(f"   • {key}: {value:.4f}")

# ====== 8. Save fine-tuned model ======
save_path = "website_classifier_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"\n💾 Model saved to: {save_path}")

# ====== 9. Test model with sample prediction ======
print("\n🧪 Testing model with sample predictions...")

def predict_website_type(text):
    """ทำนายประเภทเว็บไซต์"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    return "Blacklist" if predicted_class == 1 else "Whitelist", confidence

# ทดสอบกับข้อมูลตัวอย่าง
if len(val_df) > 0:
    sample = val_df.iloc[0]
    actual_label = "Blacklist" if sample['label'] == 1 else "Whitelist"
    
    # ต้อง import torch
    try:
        import torch
        predicted_label, confidence = predict_website_type(sample['text'][:500])
        print(f"   Sample from: {sample['source']}")
        print(f"   Actual: {actual_label}")
        print(f"   Predicted: {predicted_label} (confidence: {confidence:.3f})")
    except ImportError:
        print("   ⚠️ PyTorch not imported for testing")

# ====== 10. Final summary ======
print("\n" + "=" * 50)
print("🎉 TRAINING COMPLETE!")
print("=" * 50)
print(f"📦 Final Report:")
print(f"   • Total whitelist files: {len(all_files_white)}")
print(f"   • Total blacklist files: {len(all_files_black)}")
print(f"   • Total examples loaded: {len(data_list)}")
print(f"   • Training samples: {len(train_df)}")
print(f"   • Validation samples: {len(val_df)}")
print(f"   • Model saved to: {save_path}")
print("=" * 50)