# 🛠️ HTML Processing, Model Training, Testing, and Chrome Extension

This project provides a pipeline for processing HTML data, training a machine learning model, testing the model, and deploying it as a Chrome Extension for phishing detection.

---

## 📂 Project Structure

```
├── 

paser.py

                     # Code to convert HTML to strings
├── 

train.py

                     # Code to train the model using string data
├── 

test.py

                      # Code to test the trained model
├── chrome_extension/            # Chrome Extension source code
│   ├── background.js            # Background script for extension
│   ├── content.js               # Content script for interacting with web pages
│   ├── manifest.json            # Chrome Extension manifest file
│   ├── popup.html               # Popup UI for the extension
│   ├── popup.js                 # Logic for the popup UI
│   ├── settings.html            # Settings page for the extension
│   ├── settings.js              # Logic for the settings page
│   ├── warning.html             # Warning page for phishing alerts
│   ├── warning.js               # Logic for the warning page
├── datatrain/                   # Training datasets
│   ├── blackTraining/           # Phishing data for training
│   ├── whiteTraining/           # Legitimate data for training
├── datastring/                  # Processed string data
│   ├── black/                   # Phishing string data
│   ├── white/                   # Legitimate string data
├── backend/                     # Backend server (if applicable)
│   └── app.py                   # Flask API server for model inference
├── 

requirements.txt

             # Python dependencies
├── 

README.md

                    # Project documentation
```

---

## 🚀 Features

1. **HTML to String Conversion**  
   Extracts meaningful text from raw HTML files for further processing.

2. **Model Training**  
   Fine-tunes a BERT model using the processed string data.

3. **Model Testing**  
   Evaluates the trained model on unseen data to measure performance.

4. **Chrome Extension**  
   Deploys the trained model in a browser extension for real-time phishing detection.

---

## 🛠️ Usage

### 1. Convert HTML to Strings
Use the `paser.py` script to process raw HTML files and extract meaningful text.

```python
from bs4 import BeautifulSoup

def html_to_string(html_content: str) -> str:
    """Convert HTML content to plain text."""
    soup = BeautifulSoup(html_content, "html.parser")
    return soup.get_text(strip=True)
```

Run the script:
```bash
python 

paser.py


```

---

### 2. Train the Model
Use the `train.py` script to fine-tune a BERT model on the processed string data.

```python
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load and preprocess data
train_data, _ = load_data_by_row_range(PATH_WHITE, label=0, start_row=TRAIN_START, end_row=TRAIN_END)

# Train the model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
training_args = TrainingArguments(output_dir="./results", num_train_epochs=3, per_device_train_batch_size=8)
trainer = Trainer(model=model, args=training_args, train_dataset=encodings)
trainer.train()
```

Run the script:
```bash
python 

train.py


```

---

### 3. Test the Model
Use the `test.py` script to evaluate the trained model on test data.

```python
# Load test data
test_data, _ = load_data_by_row_range(PATH_WHITE, label=0, start_row=TEST_START, end_row=TEST_END)

# Evaluate the model
predictions = trainer.predict(test_encodings)
print(predictions)
```

Run the script:
```bash
python 

test.py


```

---

### 4. Chrome Extension
The Chrome Extension uses the trained model to detect phishing websites in real-time.

#### Setup the Extension
1. Open Chrome and go to `chrome://extensions/`.
2. Enable **Developer mode**.
3. Click **"Load unpacked"** and select the `chrome_extension/` folder.

#### Extension Files
- **`background.js`**: Handles background tasks and communication.
- **`content.js`**: Extracts HTML content from web pages.
- **`popup.html` & `popup.js`**: Provides a user interface for the extension.
- **`settings.html` & `settings.js`**: Allows users to configure the extension.
- **`warning.html` & `warning.js`**: Displays phishing alerts to users.

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/PinturaW/Performance-Comparison-Phishing-Supervised-Models-vs.-LLM-Approaches.git
   cd Performance-Comparison-Phishing-Supervised-Models-vs.-LLM-Approaches
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the backend server (if applicable):
   ```bash
   cd backend
   python app.py
   ```

4. Load the Chrome Extension as described above.

---

## 🛡️ License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```

---

### Changes Made:
1. Improved indentation and alignment for better readability.
2. Removed unnecessary blank lines and ensured consistent formatting.
3. Grouped related sections logically for easier navigation.

Let me know if you need further adjustments! 😊
