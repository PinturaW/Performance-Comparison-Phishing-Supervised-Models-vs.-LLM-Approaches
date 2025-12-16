from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer
import torch

model_dirs = ["phishing_llm_newwy_1", "phishing_llm_newwy_3", "phishing_llm_newwy_5"]
tokenizer = BertTokenizer.from_pretrained(model_dirs[0])  # ใช้ tokenizer เดิม
results = {}

for mdir in model_dirs:
    model = BertForSequenceClassification.from_pretrained(mdir)
    trainer = Trainer(model=model, tokenizer=tokenizer)
    # eval_dataset ใส่ dataset test หรือ validation ที่ใช้เทรน เดิม
    eval_res = trainer.evaluate(eval_dataset=your_eval_dataset)
    results[mdir] = eval_res["eval_loss"]

print(results)
