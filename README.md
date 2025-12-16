# 🛡️ Phishing Website Detector

**Performance Comparison: Phishing Detection with Supervised Learning vs. Fine-tuned LLM (BERT)**

[![GitHub](https://img.shields.io/badge/GitHub-PinturaW-blue?logo=github)](https://github.com/PinturaW)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![BERT](https://img.shields.io/badge/Model-BERT-orange)](https://huggingface.co/transformers/)
[![Chrome Extension](https://img.shields.io/badge/Chrome-Extension-green?logo=googlechrome)](https://developer.chrome.com/docs/extensions/)

---

## 📌 Overview

This project investigates and compares **two approaches** for detecting phishing web pages:

1. **Traditional Supervised Learning**: Using TF-IDF features + SVM/RandomForest/Gradient Boosting classifiers
2. **Large Language Models (LLMs)**: Fine-tuned BERT-based models on phishing and legitimate website datasets

The goal is to evaluate the trade-off between **performance, scalability, and real-world applicability** of supervised models vs. modern LLMs in phishing detection.

### 🎯 Key Innovation

We developed a **real-time Chrome Extension** that uses the fine-tuned BERT model to protect users while browsing, providing instant phishing detection with **97.61% confidence** on zero-day threats.

---

## 🚀 Features

✅ **Dual Approach Comparison**: Supervised ML vs. Fine-tuned BERT  
✅ **Real-time Protection**: Chrome Extension with instant threat detection  
✅ **High Accuracy**: 92-96% accuracy across different models  
✅ **Zero-Day Detection**: Catches phishing sites before they're reported  
✅ **Comprehensive Analysis**: 23 extracted features + semantic HTML analysis  
✅ **User-Friendly Interface**: Visual warnings and risk confirmations  

---

## 📊 Performance Results

### Supervised Learning Models (10-Fold Cross-Validation)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Random Forest** | **96.80%** | 0.968 | 0.968 | 0.968 |
| Gradient Boosting | 95.60% | 0.956 | 0.956 | 0.956 |
| ANN | 95.80% | 0.958 | 0.958 | 0.958 |
| SVM | 92.60% | 0.926 | 0.926 | 0.926 |
| Decision Tree | 91.40% | 0.914 | 0.914 | 0.914 |

### Fine-Tuned BERT Performance

| Training Epochs | Overall Accuracy | Phishing Accuracy | Legitimate Accuracy | Macro F1-Score |
|----------------|------------------|-------------------|---------------------|----------------|
| **3 Epochs** ⭐ | **91.0%** | **88.0%** | **94.0%** | **0.91** |
| 5 Epochs | 92.0% | 84.0% | 100.0% | 0.92 |
| 1 Epoch | 85.0% | 74.0% | 96.0% | 0.85 |

**⭐ Recommended Model**: 3-epoch BERT provides the best balance between accuracy and generalization without overfitting.

---

## 🔍 Real-World Validation

**Zero-Day Phishing Detection Example:**
```
Target: tigottggpsonline.click
Status on PhishTank: "Unknown" (not yet verified)
Our Model Detection: Phishing (97.61% confidence) ✅
```

The Chrome Extension successfully detected active phishing threats **before** they were officially reported!

---

## 🛠️ Installation

### Prerequisites
```bash
Python 3.8+
pip
Google Chrome Browser
```

### Clone the Repository
```bash
git clone https://github.com/PinturaW/Performance-Comparison-Phishing-Supervised-Models-vs.-LLM-Approaches.git
cd Performance-Comparison-Phishing-Supervised-Models-vs.-LLM-Approaches
```

### Create Virtual Environment
```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🌐 Chrome Extension Setup

### 1. Install the Extension

1. Open Chrome and go to `chrome://extensions/`
2. Enable **Developer mode** (top-right corner)
3. Click **"Load unpacked"**
4. Select the `chrome_extension` folder from this repository
5. The extension icon should appear in your toolbar

### 2. Start the Flask API Server
```bash
cd backend
python app.py
```

The server will start on `http://localhost:5001`

### 3. Start Browsing with Protection

The extension will automatically:
- Monitor every page you visit
- Extract HTML content in real-time
- Send it to the BERT model for analysis
- Display instant warnings for phishing sites

---

## 📂 Project Structure
```
├── llm_model/                    # Fine-tuned BERT models
├── chrome_extension/             # Chrome extension files
│   ├── manifest.json
│   ├── popup.html
│   ├── content.js
│   └── background.js
├── backend/                      # Flask API server
│   └── app.py
├── data/                         # Training datasets
├── notebooks/                    # Jupyter notebooks for analysis
├── bs.py                         # Web scraping utilities
├── newwy_train.py               # BERT training script
├── newwy_test.py                # BERT testing script
└── README.md
```

---

## 🎯 How It Works

### Supervised Learning Pipeline

1. **Data Collection**: Collect 800 URLs (400 phishing, 400 legitimate)
2. **Feature Extraction**: Extract 23 features from URL, HTML, and Domain
   - URL-based: Length, special characters, suspicious patterns
   - HTML-based: External resources, pop-ups, iframes
   - Domain-based: WHOIS data, DNS records, domain age
3. **Model Training**: Train multiple classifiers with cross-validation
4. **Evaluation**: Compare performance metrics

### LLM (BERT) Pipeline

1. **Data Collection**: Same 800 URLs dataset
2. **HTML Preprocessing**: Convert raw HTML to human-readable semantic text
3. **Fine-Tuning**: Train BERT on structured text strings
4. **Optimization**: Test different epoch configurations (1, 3, 5, 10)
5. **Deployment**: Integrate best model (3 epochs) into Chrome Extension

### Chrome Extension Architecture
```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Browser   │ ───> │   Extension  │ ───> │  Flask API  │
│  (User)     │      │  (HTML Extract)│     │  (BERT Model)│
└─────────────┘      └──────────────┘      └─────────────┘
                            │                      │
                            └──────────────────────┘
                                Prediction Result
                             (Phishing / Legitimate)
```

---

## 📈 Key Findings

### Strengths Comparison

| Aspect | Supervised Learning | Fine-Tuned BERT |
|--------|---------------------|-----------------|
| **Accuracy** | Higher (96.8%) | Competitive (91%) |
| **Feature Engineering** | Manual (23 features) | Automated (raw HTML) |
| **Interpretability** | High (feature importance) | Lower (black box) |
| **Scalability** | Limited by features | Handles complex patterns |
| **Real-time Performance** | Fast | Moderate (requires GPU) |

### Deployment Recommendation

- **Production Use**: 3-epoch BERT model in Chrome Extension
- **Confidence Threshold**: Flag predictions below 85% for manual review
- **Continuous Learning**: Regular retraining with new phishing samples
- **Monitoring**: Track false positives and false negatives

---

## 🔧 API Endpoints

### `/api/predict/simple` (GET)
Quick prediction for testing
```bash
curl "http://localhost:5001/api/predict/simple"
```

### `/api/predict` (POST)
Full prediction with HTML content
```bash
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"html": "<html>...</html>", "url": "https://example.com"}'
```

### `/health` (GET)
Health check endpoint
```bash
curl "http://localhost:5001/health"
```

---

## 📊 Chrome Extension Features

### 1. Real-time Monitoring
- Automatically scans every webpage you visit
- Runs in the background without disrupting browsing

### 2. Visual Warning System
- 🟢 **Safe**: Green badge for legitimate sites
- 🔴 **Danger**: Red warning page for phishing sites
- ⚠️ **Caution**: Yellow alert for suspicious sites

### 3. Risk Confirmation
- Users can choose to proceed at their own risk
- Detailed threat information displayed
- Educational tips about phishing indicators

### 4. Performance Metrics
- Processing speed: ~13.77 samples/second
- Average detection time: <1 second
- Memory usage: Optimized for Chrome

---

## 🧪 Testing

### Run Supervised Learning Tests
```bash
python phishing_detector_model/test_models.py
```

### Run BERT Model Tests
```bash
python newwy_test.py
```

### Test Chrome Extension
1. Load the extension in Chrome
2. Visit test URLs from `data/test_urls.txt`
3. Check console for detection results

---

## 📚 Research Paper

This project is based on our research paper:
**"Phishing Website Detector via LLM-based and Supervised Learning"**

### Key Contributions:
1. Comprehensive comparison of ML vs. LLM approaches
2. Real-world Chrome Extension implementation
3. Zero-day phishing detection capability
4. Optimal epoch configuration for BERT fine-tuning

---

## 👨‍💻 Author

**Wichuon Charoensombat (PinturaW)**
- 📧 Email: wichuon.cha@gmail.com
- 🔗 GitHub: [@PinturaW](https://github.com/PinturaW)
- 💼 LinkedIn: [wichuon-charoensombat](https://linkedin.com/in/wichuon-charoensombat)

### Project Team
- **Supervised Learning Pipeline**: Developed by project collaborators
- **LLM Pipeline & Chrome Extension**: Developed by Wichuon Charoensombat

---

## 🙏 Acknowledgments

- **Thammasat University (SIIT)** for providing research resources
- **PhishTank** for providing real-world phishing data
- **Hugging Face** for BERT model and transformers library
- **OpenPhish** for additional phishing URL datasets

---

## 📞 Contact & Support

If you have questions or need support:
- 📧 Email: wichuon.cha@gmail.com

---

<p align="center">
  <b>⭐ If you find this project helpful, please consider giving it a star! ⭐</b>
</p>

<p align="center">
  Made with ❤️ for a safer internet
</p>
