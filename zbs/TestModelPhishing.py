import torch
from transformers import BertForSequenceClassification, BertTokenizer
device = torch.device("cpu")
model.to(device)

model = BertForSequenceClassification.from_pretrained("./phishing_detector_model")
tokenizer = BertTokenizer.from_pretrained("./phishing_detector_model")
model.eval()

file_path = "csv/test/reddit.csv"
with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    full_text = f.read()

if len(full_text.strip()) == 0:
    print("⚠️ Input text is empty.")
else:
    print(f"🧪 Text preview: {full_text[:100]}")
    inputs = tokenizer(full_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        print(f"logits: {outputs.logits}")
        probs = torch.softmax(outputs.logits, dim=1)
        print(f"probs: {probs}")
        
        score_sexual = probs[0][1].item()
        score_non = probs[0][0].item()
        predicted_class = probs.argmax().item()

        print(f"\n📄 File: {file_path}")
        print(f"🧠 Document length: {len(full_text)} characters")
        print(f"🔎 Predicted class: {predicted_class} (1 = phishing, 0 = not phishing)")
        print(f"📊 Score [phishing]: {score_sexual:.4f} | [not phishing]: {score_non:.4f}")