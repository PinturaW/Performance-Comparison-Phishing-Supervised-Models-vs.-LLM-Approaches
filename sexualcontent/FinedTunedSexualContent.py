import transformers
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer

print(transformers.__version__)
# Sample data
data = {
    "text": [
        "Hello, how are you today?",
        "I love playing football with my friends",
        "The weather is beautiful today",
        "Let's go to the movies tonight",
        "I'm studying for my exam tomorrow",
        "This is a great restaurant",
        "I want to sleep with you",
        "You're so hot and sexy",
        "Let's have sex tonight",
        "Your body is amazing",
        "I'm horny right now",
        "Adult content warning",
        "She has a beautiful smile",
        "The book was very interesting",
        "I enjoy cooking Italian food",
        "Technology is advancing rapidly",
        "The sunset looks gorgeous",
        "Music helps me relax",
        "Exercise is good for health",
        "Learning new languages is fun"
    ],
    "label": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
}

dataset = Dataset.from_dict(data)

# Load tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize
def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)
tokenized = dataset.map(preprocess, batched=True)

# Training setup
args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    evaluation_strategy="no",  # or "epoch" if you add a validation set
    logging_dir="./logs"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized
)

trainer.train()

# ====== Save fine-tuned model ======
model.save_pretrained("llm_model")
tokenizer.save_pretrained("llm_model")
print("✅ Model saved to 'llm_model' folder")
