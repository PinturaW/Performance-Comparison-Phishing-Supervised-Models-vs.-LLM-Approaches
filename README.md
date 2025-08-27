The Fine-tuning BERT for LLM
# performance-comparison-phishing-supervised-vs-llm
Got it 👍 Here’s the **complete README in one block**, already merged and polished with everything we discussed (overview, structure, usage, results, contribution, contact, license) so you can just drop it into your repo:

```markdown
# Performance Comparison in Detecting Phishing Web Pages  
### Between Supervised Learning and Large Language Models (LLMs)

## 📌 Overview
This project investigates and compares two approaches for **detecting phishing web pages**:
1. **Traditional supervised learning models** using TF-IDF features + SVM/other classifiers.  
2. **Large Language Models (LLMs)** fine-tuned on phishing and legitimate website datasets.

The goal is to evaluate the trade-off between performance, scalability, and real-world applicability of supervised models vs. modern LLMs in phishing detection.

---

## 📂 Repository Structure
```

├── z/                         # Core supervised learning scripts
│   ├── FinedTuned.py
│   ├── FinedTunedPhishing.py
│   └── ...
├── zbs/                       # Additional baseline/testing scripts
│   ├── TestModelPhishing.py
│   └── ...
├── training/ (local only)     # Raw phishing/legitimate training data (not pushed to GitHub)
├── website\_classifier\_model/  # Fine-tuned LLM checkpoints (excluded from GitHub)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── ...

````

> ⚠️ **Note:** Large datasets and model checkpoints are not included in this repo (due to GitHub file size limits).  
They are available via external storage (Google Drive / Hugging Face link).

---

## 🚀 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/PinturaW/performance-comparison-phishing-supervised-vs-llm.git
   cd performance-comparison-phishing-supervised-vs-llm
````

2. Create a virtual environment (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   .venv\Scripts\activate      # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🧪 Usage

### 1. Supervised Learning (SVM baseline)

```bash
python z/FinedTunedPhishing.py
```

* Extracts features with TF-IDF
* Trains Support Vector Machine (SVM)
* Outputs accuracy & test summary

### 2. Fine-tuned LLM

```bash
python zbs/TestModelPhishing.py
```

* Loads fine-tuned BERT/transformer model
* Evaluates phishing vs. legitimate samples
* Saves predictions into `/predictions`

---

## 📊 Results

### Final Evaluation (Epoch = 3)
| Metric                  | Value      |
|--------------------------|------------|
| eval_loss               | **0.2955** |
| eval_accuracy           | **0.9100** |
| eval_runtime            | 7.26 sec   |
| eval_samples_per_second | 13.77      |
| eval_steps_per_second   | 3.44       |

---

### Prediction Examples

**WHITE (Legitimate Websites)**
- ✅ `row_351_th.edreams.com` → Pred: Legitimate (97.68%) | Label: Not Phishing  
- ✅ `row_354_sanooklife.com` → Pred: Legitimate (97.14%) | Label: Not Phishing  
- ⚠️ `row_355_icatcare.org` → Pred: Phishing (97.30%) | **False Positive**  
- ✅ `row_373_microsoft.com` → Pred: Legitimate (97.58%) | Label: Not Phishing  

**BLACK (Phishing Websites)**
- ⚠️ `row_351_gold1-1.github` → Pred: Phishing (97.59%) | Label: Phishing  
- ⚠️ `row_358_gratulejemy2750` → Pred: Phishing (85.58%) | Label: Phishing  
- ❌ `row_361_bxterioronlin` → Pred: Legitimate (96.68%) | **False Negative**  
- ⚠️ `row_375_flarenetwork` → Pred: Phishing (94.16%) | Label: Phishing  

---

### Test Summary
- 🟢 **White (Legitimate)** : 47 / 50 correct → **94.0%**  
- 🔴 **Black (Phishing)**  : 44 / 50 correct → **88.0%**  
- 🌐 **Overall Accuracy**  : **91.0%** (91/100)

---

📌 **Insight:**  
- The model achieves **91% balanced accuracy** across legitimate + phishing websites.  
- **Strengths:** Very high precision on legitimate (white) websites.  
- **Weaknesses:** A few false negatives on black set (hard phishing cases), which can be improved with more adversarial examples in training.  

---

## 👥 Contributions

* **Supervised Learning Pipeline** – handled by project collaborators
* **LLM Pipeline** – developed and fine-tuned by **Wichuon Charoensombat (PinturaW)**

---

## 📬 Contact

Developed by **Wichuon Charoensombat** (PinturaW)
📧 Reach me on [LinkedIn](https://www.linkedin.com/in/wichuon-charoensombat) or GitHub

---
