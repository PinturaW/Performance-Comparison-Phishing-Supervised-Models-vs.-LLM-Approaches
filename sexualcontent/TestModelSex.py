import torch
from transformers import BertForSequenceClassification, BertTokenizer

model = BertForSequenceClassification.from_pretrained("llm_model")
tokenizer = BertTokenizer.from_pretrained("llm_model")

# ====== Test a prediction ======
test_text = "sex is so fun"
# Tokenize
# use softmax to see probability
inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True)
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)
    score = probs[0][1].item()  # Probability for class 1 (sexual)
    score0 = probs[0][0].item()  # Probability for class 1 (sexual)
    predicted_class = probs.argmax().item()


# Show result
print(f"🧠 Input: {test_text}")
print(f"🔎 Predicted class: {predicted_class} (1 = sexual, 0 = not sexual)")
print(f"📊 Score for class 1 (sexual): {score:.4f}")
print(f"📊 Score for class 0 (No sexual): {score0:.4f}")