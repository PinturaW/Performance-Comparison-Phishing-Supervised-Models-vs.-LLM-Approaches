import torch
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import glob
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
print(torch.cuda.is_available())  # True = using GPU
print(torch.cuda.device_count())  # Number of GPUs available
class WebsiteDataset(Dataset):
    """Custom Dataset class for website text classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):  # ลด max_length
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # เพิ่มการประมวลผลข้อความ - เน้น keywords สำคัญ
        # ดึง title และ domain ที่สำคัญมาใส่ข้างหน้า
        lines = text.split('\n')
        important_info = []
        
        for line in lines:
            if line.startswith('Title:') or line.startswith('Domain:') or line.startswith('Meta_description:'):
                important_info.append(line)
        
        # รวมข้อมูลสำคัญ + ข้อความเต็ม
        enhanced_text = ' '.join(important_info) + ' ' + text
        
        encoding = self.tokenizer(
            enhanced_text,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=self.max_length
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_training_data(whitelist_folder, blacklist_folder):
    """Load training data from whitelist and blacklist folders"""
    
    texts = []
    labels = []
    
    print(f"🔍 Looking for training data...")
    print(f"   • Whitelist folder: {whitelist_folder}")
    print(f"   • Blacklist folder: {blacklist_folder}")
    
    # Load whitelist data (label = 0)
    if os.path.exists(whitelist_folder):
        whitelist_files = glob.glob(os.path.join(whitelist_folder, '*.csv'))
        print(f"📁 Found {len(whitelist_files)} whitelist files")
        
        for file_path in whitelist_files:
            try:
                print(f"   📄 Reading whitelist: {os.path.basename(file_path)}")
                df = pd.read_csv(file_path, encoding='utf-8')
                print(f"      Columns: {list(df.columns)}")
                
                if 'data_string' in df.columns and len(df) > 0:
                    text = df['data_string'].iloc[0]
                    if text and len(str(text).strip()) > 0:
                        texts.append(str(text))
                        labels.append(0)  # Whitelist = 0
                        print(f"      ✅ Added whitelist sample ({len(str(text))} chars)")
                    else:
                        print(f"      ⚠️ Empty data_string in {file_path}")
                else:
                    print(f"      ⚠️ No 'data_string' column in {file_path}")
            except Exception as e:
                print(f"❌ Error reading whitelist file {file_path}: {e}")
    else:
        print(f"⚠️ Whitelist folder not found: {whitelist_folder}")
    
    # Load blacklist data (label = 1)
    if os.path.exists(blacklist_folder):
        blacklist_files = glob.glob(os.path.join(blacklist_folder, '*.csv'))
        print(f"📁 Found {len(blacklist_files)} blacklist files")
        
        for file_path in blacklist_files:
            try:
                print(f"   📄 Reading blacklist: {os.path.basename(file_path)}")
                df = pd.read_csv(file_path, encoding='utf-8')
                print(f"      Columns: {list(df.columns)}")
                
                if 'data_string' in df.columns and len(df) > 0:
                    text = df['data_string'].iloc[0]
                    if text and len(str(text).strip()) > 0:
                        texts.append(str(text))
                        labels.append(1)  # Blacklist = 1
                        print(f"      ✅ Added blacklist sample ({len(str(text))} chars)")
                    else:
                        print(f"      ⚠️ Empty data_string in {file_path}")
                else:
                    print(f"      ⚠️ No 'data_string' column in {file_path}")
            except Exception as e:
                print(f"❌ Error reading blacklist file {file_path}: {e}")
    else:
        print(f"⚠️ Blacklist folder not found: {blacklist_folder}")
    
    print(f"✅ Loaded {len(texts)} samples total")
    print(f"   • Whitelist (0): {labels.count(0)}")
    print(f"   • Blacklist (1): {labels.count(1)}")
    
    return texts, labels

def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # คำนวณ metrics แยกแต่ละ class
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, average=None, zero_division=0)
    acc = accuracy_score(labels, predictions)
    
    # แสดงรายละเอียดแต่ละ class
    results = {
        'accuracy': acc,
        'f1_macro': np.mean(f1),
        'precision_macro': np.mean(precision),
        'recall_macro': np.mean(recall)
    }
    
    # เพิ่มรายละเอียดแต่ละ class
    for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
        class_name = 'whitelist' if i == 0 else 'blacklist'
        results[f'{class_name}_precision'] = p
        results[f'{class_name}_recall'] = r
        results[f'{class_name}_f1'] = f
        results[f'{class_name}_support'] = s
    
    return results

def train_classifier(whitelist_folder, blacklist_folder, output_dir="./website_classifier_model"):
    """Train the website classifier"""
    
    print("🚀 Starting Website Classifier Training")
    print("=" * 60)
    
    # Load data
    texts, labels = load_training_data(whitelist_folder, blacklist_folder)
    
    if len(texts) == 0:
        print("❌ No training data found!")
        return False
    
    if len(set(labels)) < 2:
        print("❌ Need both whitelist (0) and blacklist (1) data!")
        print(f"Current labels: {set(labels)}")
        return False
    
    # Split data
    try:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"📊 Data split:")
        print(f"   • Training: {len(train_texts)} samples")
        print(f"   • Validation: {len(val_texts)} samples")
        print(f"   • Training distribution: {np.bincount(train_labels)}")
        print(f"   • Validation distribution: {np.bincount(val_labels)}")
        
    except Exception as e:
        print(f"❌ Error splitting data: {e}")
        return False
    
    # Initialize tokenizer and model
    try:
        print("🤖 Loading BERT model...")
        model_name = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2,  # Binary classification
            problem_type="single_label_classification"
        )
        print("✅ BERT model loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading BERT model: {e}")
        return False
    
    # Create datasets
    try:
        print("📦 Creating datasets...")
        train_dataset = WebsiteDataset(train_texts, train_labels, tokenizer)
        val_dataset = WebsiteDataset(val_texts, val_labels, tokenizer)
        print("✅ Datasets created successfully")
        
    except Exception as e:
        print(f"❌ Error creating datasets: {e}")
        return False
    
    # Training arguments - Improved for better learning
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,                 # เพิ่ม epochs
        per_device_train_batch_size=2,      # ลด batch size เพิ่มเติม
        per_device_eval_batch_size=2,       
        learning_rate=2e-5,                 # เพิ่ม learning rate
        warmup_steps=50,                    # ลด warmup
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=25,                   
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",   # เปลี่ยนเป็น accuracy
        greater_is_better=True,
        report_to=None,  
        save_total_limit=2,  
        seed=42,                            # เพิ่ม seed สำหรับความเสถียร
        dataloader_drop_last=False,         # ไม่ทิ้งข้อมูล
        fp16=False,                         # ปิด mixed precision
    )
    
    # Initialize trainer
    try:
        print("🎯 Initializing trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        print("✅ Trainer initialized successfully")
        
    except Exception as e:
        print(f"❌ Error initializing trainer: {e}")
        return False
    
    # Train model
    try:
        print("🎯 Starting training...")
        print("⏳ This may take several minutes...")
        trainer.train()
        print("✅ Training completed!")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Save model and tokenizer
    try:
        print(f"💾 Saving model to {output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        print("✅ Model saved successfully!")
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return False
    
    # Evaluate on validation set
    try:
        print("📊 Final evaluation...")
        eval_results = trainer.evaluate()
        print(f"📊 Final validation results:")
        for key, value in eval_results.items():
            if isinstance(value, float):
                print(f"   • {key}: {value:.4f}")
            else:
                print(f"   • {key}: {value}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        
    return True

def test_classifier(model_path="./website_classifier_model", test_file="csv/test/reddit.csv"):
    """Test the trained classifier"""
    
    print("🧪 Testing Website Classifier")
    print("=" * 40)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("💡 Please train the model first!")
        return None
    
    # Load model and tokenizer
    try:
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        
        device = torch.device("cpu")
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded from: {model_path}")
        print(f"🖥️ Using device: {device}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 The model might be corrupted. Please retrain!")
        return None
    
    # Read test file
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return None
    
    try:
        # Read CSV file properly
        print(f"📄 Reading test file: {test_file}")
        df = pd.read_csv(test_file, encoding='utf-8')
        print(f"   Columns found: {list(df.columns)}")
        
        if 'data_string' in df.columns and len(df) > 0:
            full_text = str(df['data_string'].iloc[0])
        else:
            # Fallback to reading as text file
            with open(test_file, "r", encoding="utf-8", errors="ignore") as f:
                full_text = f.read()
            
        if len(full_text.strip()) == 0:
            print("⚠️ Input text is empty.")
            return None
            
        print(f"📏 Text length: {len(full_text)} characters")
        print(f"🧪 Text preview: {full_text[:100]}...")
        
        # Tokenize and predict
        inputs = tokenizer(
            full_text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        print(f"🧠 Raw logits: {outputs.logits}")
        
        # Check for NaN values
        if torch.isnan(outputs.logits).any():
            print("❌ Model output contains NaN values!")
            print("💡 This usually means the model wasn't trained properly.")
            print("💡 Please retrain the model with proper training data.")
            return None
        
        # Calculate probabilities
        probs = torch.softmax(outputs.logits, dim=1)
        print(f"📊 Probabilities: {probs}")
        
        # Extract scores
        score_whitelist = probs[0][0].item()  # class 0 = whitelist (not phishing)
        score_blacklist = probs[0][1].item()  # class 1 = blacklist (phishing)
        
        predicted_class = probs.argmax().item()
        
        # Display results
        print(f"\n" + "="*50)
        print(f"📄 File: {test_file}")
        print(f"📏 Document length: {len(full_text)} characters")
        print(f"🔎 Predicted class: {predicted_class} (1 = blacklist, 0 = whitelist)")
        print(f"🏷️  Classification: {'🚨 BLACKLIST' if predicted_class == 1 else '✅ WHITELIST'}")
        print(f"📊 Confidence Scores:")
        print(f"   • Whitelist (class 0): {score_whitelist:.4f}")
        print(f"   • Blacklist (class 1): {score_blacklist:.4f}")
        print(f"=" * 50)
        
        return {
            'predicted_class': predicted_class,
            'classification': 'blacklist' if predicted_class == 1 else 'whitelist',
            'score_whitelist': score_whitelist,
            'score_blacklist': score_blacklist,
            'text_length': len(full_text)
        }
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return None

def batch_test(model_path="./website_classifier_model", test_folder="csv/test"):
    """Test multiple files in a folder"""
    
    if not os.path.exists(test_folder):
        print(f"❌ Test folder not found: {test_folder}")
        return
    
    csv_files = glob.glob(os.path.join(test_folder, '*.csv'))
    
    if not csv_files:
        print(f"⚠️ No CSV files found in: {test_folder}")
        return
    
    print(f"🔍 Found {len(csv_files)} CSV files to test")
    
    results = []
    
    for i, file_path in enumerate(csv_files, 1):
        print(f"\n[{i}/{len(csv_files)}] Testing: {os.path.basename(file_path)}")
        result = test_classifier(model_path, file_path)
        if result:
            result['file'] = file_path
            results.append(result)
        print("-" * 30)
    
    # Summary
    if results:
        print(f"\n🎉 BATCH TESTING COMPLETE!")
        print(f"📊 Summary:")
        
        whitelist_count = sum(1 for r in results if r['classification'] == 'whitelist')
        blacklist_count = sum(1 for r in results if r['classification'] == 'blacklist')
        
        print(f"   • Total files tested: {len(results)}")
        print(f"   • Whitelist: {whitelist_count}")
        print(f"   • Blacklist: {blacklist_count}")
        
        print(f"\n📋 Detailed Results:")
        for result in results:
            status = "🚨 BLACKLIST" if result['classification'] == 'blacklist' else "✅ WHITELIST"
            confidence = result['score_blacklist'] if result['classification'] == 'blacklist' else result['score_whitelist']
            print(f"   • {os.path.basename(result['file'])}: {status} (confidence: {confidence:.3f})")
    
    return results

# ====== Main Execution ======
if __name__ == "__main__":
    print("🚀 Website Blacklist/Whitelist Classifier")
    print("=" * 60)
    
    # Configuration - Updated to match your folder structure
    WHITELIST_FOLDER = "csv/white"     # Folder containing whitelist websites (label 0)
    BLACKLIST_FOLDER = "csv/black"     # Folder containing blacklist websites (label 1)
    MODEL_PATH = "./website_classifier_model"
    TEST_FILE = "csv/black/ufafat8.csv"
    TEST_FOLDER = "csv/test"
    
    # Check if folders exist and count files
    print("🔍 Checking your data:")
    folders_to_check = [WHITELIST_FOLDER, BLACKLIST_FOLDER, TEST_FOLDER]
    for folder in folders_to_check:
        if os.path.exists(folder):
            files = glob.glob(os.path.join(folder, '*.csv'))
            print(f"✅ {folder}: {len(files)} CSV files found")
            for file in files:
                print(f"   📄 {os.path.basename(file)}")
        else:
            print(f"❌ {folder}: Folder not found")
    
    print("")
    
    # Choose mode
    mode = input("Choose mode:\n1. Train model\n2. Test single file\n3. Batch test\n4. Train and test\nEnter choice (1-4): ")
    
    if mode == "1" or mode == "4":
        # Training
        print("\n🎯 TRAINING MODE")
        print("=" * 60)
        try:
            train_classifier(WHITELIST_FOLDER, BLACKLIST_FOLDER, MODEL_PATH)
            print("✅ Training completed successfully!")
        except Exception as e:
            print(f"❌ Training failed: {e}")
            print("💡 Please check your data files and try again.")
            import traceback
            traceback.print_exc()
    
    if mode == "2" or mode == "4":
        # Single file testing
        print(f"\n🧪 SINGLE FILE TESTING")
        print("=" * 40)
        try:
            result = test_classifier(MODEL_PATH, TEST_FILE)
            if result:
                print("✅ Testing completed successfully!")
            else:
                print("❌ Testing failed!")
        except Exception as e:
            print(f"❌ Testing failed: {e}")
            import traceback
            traceback.print_exc()
    
    if mode == "3":
        # Batch testing
        print(f"\n🔍 BATCH TESTING")
        print("=" * 40)
        try:
            results = batch_test(MODEL_PATH, TEST_FOLDER)
            if results:
                print("✅ Batch testing completed successfully!")
            else:
                print("❌ Batch testing failed!")
        except Exception as e:
            print(f"❌ Batch testing failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✨ Process completed!")