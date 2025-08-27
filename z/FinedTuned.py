import transformers
import pandas as pd
import glob
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import os
import re

print(f"Transformers version: {transformers.__version__}")

# ====== 1. Clean text function ======
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)                      # Remove HTML tags
    text = re.sub(r'http\S+|www\S+', '', text)             # Remove URLs
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)         # Remove special characters
    text = re.sub(r'\s+', ' ', text)                       # Normalize whitespace
    return text.lower().strip()

# ====== 2. Load data from files ======

path_white = 'csv/white'
path_black = 'csv/black'
all_files_white = glob.glob(os.path.join(path_white, '*.csv'))
all_files_black = glob.glob(os.path.join(path_black, '*.csv'))

data_list = []

print("🟢 Reading whitelist files...")
for filename in all_files_white:
    try:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        cleaned = clean_text(content)
        data_list.append({'text': cleaned, 'label': 0})
        print(f"✅ {os.path.basename(filename)} (white)")
    except Exception as e:
        print(f"❌ Error reading whitelist file {filename}: {e}")

print("🔴 Reading blacklist files...")
for filename in all_files_black:
    try:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        cleaned = clean_text(content)
        data_list.append({'text': cleaned, 'label': 1})
        print(f"✅ {os.path.basename(filename)} (black)")
    except Exception as e:
        print(f"❌ Error reading blacklist file {filename}: {e}")

if not data_list:
    raise ValueError("No data found. Check csv folders.")

combined_df = pd.DataFrame(data_list)
print("\n📊 Sample from combined DataFrame:")
print(combined_df.sample(3))

dataset = Dataset.from_pandas(combined_df)
print("\n✅ Hugging Face dataset created:")
print(dataset)

# ====== 3. Tokenization ======

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

def preprocess(examples):
    texts = [str(t) if t is not None else "" for t in examples["text"]]
    return tokenizer(texts, truncation=True, padding="max_length", max_length=512)

tokenized = dataset.map(preprocess, batched=True)

# Split the dataset into train and validation sets (e.g., 80/20)
split_dataset = tokenized.train_test_split(test_size=0.2, seed=42)

print("*********************")
print(split_dataset)
print("*********************")

train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
# ====== 4. Training ======

args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,        # If GPU allows
    num_train_epochs=1,                   # Try 3–5
    evaluation_strategy="epoch",          # Helpful to monitor
    learning_rate=5e-4,                   # Stable for BERT
    logging_dir="./logs",
    save_strategy="epoch",               # Save model at end of each epoch
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

print("\n🚀 Starting model training...")
trainer.train()
print("✅ Training completed.")

# ====== 5. Save fine-tuned model ======

save_path = "llm_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"💾 Model saved to: {save_path}")

# ====== 6. Final summary ======
print("\n📦 Final Report")
print("Total whitelist files:", len(all_files_white))
print("Total blacklist files:", len(all_files_black))
print("Total examples loaded:", len(data_list))