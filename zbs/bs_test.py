import transformers
import pandas as pd
import glob
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
import re
import random
import numpy as np

print(f"Transformers version: {transformers.__version__}")

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# ====== 1. Clean text function ======
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    return text.lower().strip()

# ====== 2. Load data from files ======
def load_data_with_limit(path, label, limit=None):
    """Load data files with optional limit"""
    all_files = glob.glob(os.path.join(path, '*.csv'))
    data_list = []
    
    # Shuffle files for random selection
    random.shuffle(all_files)
    
    count = 0
    for filename in all_files:
        if limit and count >= limit:
            break
            
        try:
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                cleaned = clean_text(content)
                if cleaned.strip():  # Only add non-empty content
                    data_list.append({'text': cleaned, 'label': label})
                    count += 1
                    print(f"✅ {os.path.basename(filename)} ({'white' if label == 0 else 'black'})")
        except Exception as e:
            print(f"❌ Error reading file {filename}: {e}")
    
    return data_list

# Load training data
print("🟢 Loading whitelist training data (200 files)...")
path_white = 'csv_train/white'  # Updated path based on your bs_training output
white_data = load_data_with_limit(path_white, label=0, limit=200)

print("🔴 Loading blacklist training data (120 files)...")
path_black = 'csv_train/black'  # Updated path based on your bs_training output
black_data = load_data_with_limit(path_black, label=1, limit=200)

print(f"\n📊 Training Data Summary:")
print(f"   🟢 White (legitimate): {len(white_data)} files")
print(f"   🔴 Black (phishing): {len(black_data)} files")
print(f"   📊 Total training: {len(white_data) + len(black_data)} files")

# ====== 3. Create test data from remaining files ======
def load_test_data(path, label, skip_count, test_limit):
    """Load test data by skipping already used files"""
    all_files = glob.glob(os.path.join(path, '*.csv'))
    random.shuffle(all_files)
    
    # Skip files already used for training
    test_files = all_files[skip_count:skip_count + test_limit]
    
    test_data = []
    for filename in test_files:
        try:
            with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                cleaned = clean_text(content)
                if cleaned.strip():
                    test_data.append({'text': cleaned, 'label': label})
                    print(f"🧪 Test: {os.path.basename(filename)} ({'white' if label == 0 else 'black'})")
        except Exception as e:
            print(f"❌ Error reading test file {filename}: {e}")
    
    return test_data

print("\n🧪 Loading test data...")
print("🟢 Loading whitelist test data (25 files)...")
white_test = load_test_data(path_white, label=0, skip_count=200, test_limit=25)

print("🔴 Loading blacklist test data (25 files)...")
black_test = load_test_data(path_black, label=1, skip_count=200, test_limit=25)

print(f"\n📊 Test Data Summary:")
print(f"   🟢 White test: {len(white_test)} files")
print(f"   🔴 Black test: {len(black_test)} files")
print(f"   📊 Total test: {len(white_test) + len(black_test)} files")

# Combine training and test data
train_data = white_data + black_data
test_data = white_test + black_test

if not train_data:
    raise ValueError("No training data found. Check csv_train folders.")
if not test_data:
    raise ValueError("No test data found. Check csv_train folders.")

# Create DataFrames and datasets
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

print("\n📊 Training Data Distribution:")
print(train_df['label'].value_counts())
print("\n📊 Test Data Distribution:")
print(test_df['label'].value_counts())

# Create Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

print(f"\n✅ Datasets created:")
print(f"   🚀 Training dataset: {len(train_dataset)} examples")
print(f"   🧪 Test dataset: {len(test_dataset)} examples")

# ====== 4. Tokenization ======
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

def preprocess(examples):
    texts = [str(t) if t is not None else "" for t in examples["text"]]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

print("\n🔄 Tokenizing datasets...")
train_tokenized = train_dataset.map(preprocess, batched=True)
test_tokenized = test_dataset.map(preprocess, batched=True)

# ====== 5. Training ======
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions)
    }

args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    evaluation_strategy="epoch",
    eval_steps=500,
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

# ====== 6. Final evaluation ======
print("\n🧪 Running final evaluation...")
eval_results = trainer.evaluate()
print(f"📊 Final Evaluation Results:")
for key, value in eval_results.items():
    print(f"   {key}: {value:.4f}")

# ====== 7. Detailed predictions on test set ======
print("\n🔍 Generating detailed predictions...")
predictions = trainer.predict(test_tokenized)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = test_df['label'].values

print("\n📊 Detailed Classification Report:")
print(classification_report(y_true, y_pred, target_names=['Legitimate', 'Phishing']))

# ====== 8. Save fine-tuned model ======
save_path = "phishing_llm_2"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"\n💾 Model saved to: {save_path}")

# ====== 9. Final summary ======
print("\n" + "="*60)
print("📦 FINAL TRAINING REPORT")
print("="*60)
print(f"📊 Training Data:")
print(f"   🟢 Legitimate (white): {len(white_data)} examples")
print(f"   🔴 Phishing (black): {len(black_data)} examples")
print(f"   📊 Total training: {len(train_data)} examples")
print(f"\n🧪 Test Data:")
print(f"   🟢 Legitimate (white): {len(white_test)} examples") 
print(f"   🔴 Phishing (black): {len(black_test)} examples")
print(f"   📊 Total test: {len(test_data)} examples")
print(f"\n🎯 Model Performance:")
print(f"   📊 Test Accuracy: {eval_results.get('eval_accuracy', 'N/A'):.4f}")
print(f"   💾 Model saved to: {save_path}")
print("="*60)