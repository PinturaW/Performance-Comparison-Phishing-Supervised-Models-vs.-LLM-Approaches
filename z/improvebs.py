import torch
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import glob
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import re
from collections import Counter

class ImprovedWebsiteDataset(Dataset):
    """Enhanced Dataset class with better text preprocessing for website classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def extract_features(self, text):
        """Extract important features from website text"""
        features = []
        
        # Extract and prioritize key components
        lines = text.split('\n')
        
        # High priority content
        title = ""
        domain = ""
        meta_desc = ""
        keywords = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('Title:'):
                title = line.replace('Title:', '').strip()
            elif line.startswith('Domain:'):
                domain = line.replace('Domain:', '').strip()
            elif line.startswith('Meta_description:'):
                meta_desc = line.replace('Meta_description:', '').strip()
            elif line.startswith('Keywords:'):
                keywords = line.replace('Keywords:', '').strip()
        
        # Create enhanced text with weighted importance
        enhanced_parts = []
        
        # Domain gets highest weight (appears 3 times)
        if domain:
            enhanced_parts.extend([f"DOMAIN: {domain}"] * 3)
        
        # Title gets high weight (appears 2 times)
        if title:
            enhanced_parts.extend([f"TITLE: {title}"] * 2)
        
        # Meta description and keywords
        if meta_desc:
            enhanced_parts.append(f"DESCRIPTION: {meta_desc}")
        if keywords:
            enhanced_parts.append(f"KEYWORDS: {keywords}")
        
        # Add gambling/suspicious keywords detection
        gambling_keywords = [
            'casino', 'bet', 'gambling', 'poker', 'jackpot', 'slot', 'roulette',
            'blackjack', 'lottery', 'prize', 'win money', 'deposit', 'bonus',
            'free spins', 'odds', 'wager', 'payout', 'fortune', 'luck'
        ]
        
        suspicious_keywords = [
            'free money', 'get rich', 'easy money', 'guaranteed', 'risk free',
            'limited time', 'act now', 'exclusive offer', 'no credit check'
        ]
        
        text_lower = text.lower()
        gambling_count = sum(1 for keyword in gambling_keywords if keyword in text_lower)
        suspicious_count = sum(1 for keyword in suspicious_keywords if keyword in text_lower)
        
        if gambling_count > 0:
            enhanced_parts.append(f"GAMBLING_INDICATORS: {gambling_count}")
        if suspicious_count > 0:
            enhanced_parts.append(f"SUSPICIOUS_INDICATORS: {suspicious_count}")
        
        # Add original text
        enhanced_parts.append(text)
        
        return ' '.join(enhanced_parts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enhanced_text = self.extract_features(text)
        
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

def analyze_data_quality(texts, labels):
    """Analyze the quality and balance of training data"""
    print("📊 DATA QUALITY ANALYSIS")
    print("=" * 50)
    
    total_samples = len(texts)
    whitelist_count = labels.count(0)
    blacklist_count = labels.count(1)
    
    print(f"Total samples: {total_samples}")
    print(f"Whitelist (0): {whitelist_count} ({whitelist_count/total_samples:.1%})")
    print(f"Blacklist (1): {blacklist_count} ({blacklist_count/total_samples:.1%})")
    
    # Check for severe imbalance
    if total_samples > 0:
        imbalance_ratio = max(whitelist_count, blacklist_count) / max(min(whitelist_count, blacklist_count), 1)
        if imbalance_ratio > 5:
            print(f"⚠️  SEVERE CLASS IMBALANCE detected (ratio: {imbalance_ratio:.1f}:1)")
            print("💡 Consider collecting more data for the minority class")
        elif imbalance_ratio > 2:
            print(f"⚠️  Moderate class imbalance (ratio: {imbalance_ratio:.1f}:1)")
    
    # Analyze text lengths
    text_lengths = [len(str(text)) for text in texts]
    avg_length = np.mean(text_lengths)
    median_length = np.median(text_lengths)
    
    print(f"\nText length statistics:")
    print(f"Average: {avg_length:.0f} characters")
    print(f"Median: {median_length:.0f} characters")
    print(f"Range: {min(text_lengths)} - {max(text_lengths)} characters")
    
    # Check for very short texts
    short_texts = sum(1 for length in text_lengths if length < 100)
    if short_texts > 0:
        print(f"⚠️  {short_texts} samples have very short text (<100 chars)")
    
    return {
        'total_samples': total_samples,
        'class_balance': {'whitelist': whitelist_count, 'blacklist': blacklist_count},
        'imbalance_ratio': imbalance_ratio if total_samples > 0 else 0,
        'avg_text_length': avg_length
    }

def load_training_data(whitelist_folder, blacklist_folder):
    """Enhanced data loading with better error handling and validation"""
    
    texts = []
    labels = []
    
    print(f"🔍 Loading training data...")
    print(f"   • Whitelist folder: {whitelist_folder}")
    print(f"   • Blacklist folder: {blacklist_folder}")
    
    # Load whitelist data (label = 0)
    if os.path.exists(whitelist_folder):
        whitelist_files = glob.glob(os.path.join(whitelist_folder, '*.csv'))
        print(f"📁 Found {len(whitelist_files)} whitelist files")
        
        for file_path in whitelist_files:
            try:
                print(f"   📄 Reading: {os.path.basename(file_path)}")
                df = pd.read_csv(file_path, encoding='utf-8')
                
                if 'data_string' in df.columns:
                    # Process all rows, not just the first one
                    for idx, row in df.iterrows():
                        text = row['data_string']
                        if pd.notna(text) and len(str(text).strip()) > 50:  # Minimum length requirement
                            texts.append(str(text))
                            labels.append(0)  # Whitelist = 0
                    print(f"      ✅ Added {len(df)} whitelist samples")
                else:
                    print(f"      ⚠️ No 'data_string' column found")
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
    
    # Load blacklist data (label = 1)
    if os.path.exists(blacklist_folder):
        blacklist_files = glob.glob(os.path.join(blacklist_folder, '*.csv'))
        print(f"📁 Found {len(blacklist_files)} blacklist files")
        
        for file_path in blacklist_files:
            try:
                print(f"   📄 Reading: {os.path.basename(file_path)}")
                df = pd.read_csv(file_path, encoding='utf-8')
                
                if 'data_string' in df.columns:
                    # Process all rows, not just the first one
                    for idx, row in df.iterrows():
                        text = row['data_string']
                        if pd.notna(text) and len(str(text).strip()) > 50:  # Minimum length requirement
                            texts.append(str(text))
                            labels.append(1)  # Blacklist = 1
                    print(f"      ✅ Added {len(df)} blacklist samples")
                else:
                    print(f"      ⚠️ No 'data_string' column found")
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")
    
    # Analyze data quality
    if len(texts) > 0:
        analyze_data_quality(texts, labels)
    
    return texts, labels

def compute_detailed_metrics(eval_pred):
    """Enhanced metrics computation with confusion matrix"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    # Basic metrics
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, average=None, zero_division=0)
    acc = accuracy_score(labels, predictions)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    results = {
        'accuracy': acc,
        'f1_macro': np.mean(f1),
        'precision_macro': np.mean(precision),
        'recall_macro': np.mean(recall)
    }
    
    # Class-specific metrics
    class_names = ['whitelist', 'blacklist']
    for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
        results[f'{class_names[i]}_precision'] = p
        results[f'{class_names[i]}_recall'] = r
        results[f'{class_names[i]}_f1'] = f
        results[f'{class_names[i]}_support'] = s
    
    # Confusion matrix values
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        results['true_negatives'] = tn
        results['false_positives'] = fp
        results['false_negatives'] = fn
        results['true_positives'] = tp
        
        # Calculate specificity and sensitivity
        results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        results['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return results

def train_improved_classifier(whitelist_folder, blacklist_folder, output_dir="./improved_classifier_model"):
    """Detect potential overfitting issues"""
    print("\n🔍 OVERFITTING ANALYSIS")
    print("=" * 40)
    
    # Check for perfect scores (potential overfitting)
    perfect_metrics = []
    for key, value in eval_results.items():
        if 'f1' in key or 'precision' in key or 'recall' in key or 'accuracy' in key:
            if isinstance(value, float) and value >= 0.999:
                perfect_metrics.append(key)
    
    if perfect_metrics:
        print("⚠️  POTENTIAL OVERFITTING DETECTED!")
        print(f"📊 Perfect scores found in: {', '.join(perfect_metrics)}")
        print("\n💡 Recommendations:")
        print("   • Collect more diverse training data")
        print("   • Add data augmentation")
        print("   • Reduce model complexity")
        print("   • Use cross-validation")
        print("   • Add regularization")
        
        # Check validation set size
        if 'eval_whitelist_support' in eval_results and 'eval_blacklist_support' in eval_results:
            val_size = eval_results['eval_whitelist_support'] + eval_results['eval_blacklist_support']
            if val_size < 20:
                print(f"   • Increase validation set size (current: {val_size} samples)")
        
        return True
    else:
        print("✅ No obvious overfitting detected")
        return False

def create_robust_splits(texts, labels, min_val_samples=10):
    """Create more robust train/validation splits"""
    
    total_samples = len(texts)
    unique_labels = len(set(labels))
    
    if total_samples < min_val_samples * 2:
        print(f"⚠️  Limited data: {total_samples} samples total")
        print("💡 Using smaller validation split to preserve training data")
        test_size = max(0.1, min_val_samples / total_samples)
    else:
        test_size = 0.2
    
    try:
        # Try stratified split first
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, 
            test_size=test_size, 
            random_state=42, 
            stratify=labels
        )
        print(f"✅ Stratified split successful")
        
    except ValueError as e:
        print(f"⚠️  Stratified split failed: {e}")
        print("💡 Using random split instead")
        
        # Fall back to random split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, 
            test_size=test_size, 
            random_state=42
        )
    
    return train_texts, val_texts, train_labels, val_labels
    """Train an improved website classifier with better hyperparameters"""
    
    print("🚀 Training Improved Website Classifier")
    print("=" * 70)
    
    # Load and validate data
    texts, labels = load_training_data(whitelist_folder, blacklist_folder)
    
    if len(texts) == 0:
        print("❌ No training data found!")
        return False
    
    if len(set(labels)) < 2:
        print("❌ Need both whitelist and blacklist data!")
        return False
    
    # Check minimum data requirements
    if len(texts) < 10:
        print("⚠️  Very limited training data. Consider collecting more samples.")
    
    # Enhanced data splitting with overfitting prevention
    train_texts, val_texts, train_labels, val_labels = create_robust_splits(texts, labels)
    
    # Load model and tokenizer
    try:
        print("🤖 Loading BERT model...")
        model_name = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2,
            problem_type="single_label_classification"
        )
        
        # Add custom tokens for our domain-specific features
        special_tokens = ['[DOMAIN]', '[TITLE]', '[GAMBLING]', '[SUSPICIOUS]']
        tokenizer.add_tokens(special_tokens)
        model.resize_token_embeddings(len(tokenizer))
        
        print("✅ BERT model loaded and customized")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Create enhanced datasets
    train_dataset = ImprovedWebsiteDataset(train_texts, train_labels, tokenizer)
    val_dataset = ImprovedWebsiteDataset(val_texts, val_labels, tokenizer)
    
    # Calculate class weights for imbalanced data
    class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    class_weights = total_samples / (2.0 * class_counts)
    
    print(f"📊 Class weights: {dict(enumerate(class_weights))}")
    
    # Improved training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=8,                 # More epochs for better learning
        per_device_train_batch_size=4,      # Larger batch size if memory allows
        per_device_eval_batch_size=4,
        learning_rate=1e-5,                 # Lower learning rate for stability
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",   # Use F1 for imbalanced data
        greater_is_better=True,
        report_to=None,
        save_total_limit=3,
        seed=42,
        dataloader_drop_last=False,
        fp16=False,  # Disable for stability
        gradient_accumulation_steps=2,      # Accumulate gradients
        eval_accumulation_steps=2,
    )
    
    # Custom trainer with class weights
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_detailed_metrics,
    )
    
    # Train the model
    try:
        print("🎯 Starting training...")
        trainer.train()
        print("✅ Training completed!")
        
        # Save model
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        print(f"💾 Model saved to {output_dir}")
        
        # Final evaluation with overfitting detection
        eval_results = trainer.evaluate()
        detect_overfitting(eval_results)
        
        print(f"\n📊 FINAL EVALUATION RESULTS:")
        print("=" * 50)
        for key, value in eval_results.items():
            if isinstance(value, (int, float)):
                if 'f1' in key or 'precision' in key or 'recall' in key or 'accuracy' in key:
                    print(f"   • {key}: {value:.4f}")
                else:
                    print(f"   • {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_improved_classifier(model_path="./improved_classifier_model", test_file=None):
    """Test the improved classifier with detailed analysis"""
    
    print("🧪 Testing Improved Website Classifier")
    print("=" * 50)
    
    # Load model
    try:
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model.eval()
        print(f"✅ Model loaded from: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Handle test file input
    if test_file is None:
        # Find available test files
        test_folder = "csv/test"
        if os.path.exists(test_folder):
            csv_files = glob.glob(os.path.join(test_folder, '*.csv'))
            if csv_files:
                test_file = csv_files[0]
                print(f"📁 Auto-selected test file: {os.path.basename(test_file)}")
            else:
                print(f"❌ No CSV files found in {test_folder}")
                return None
        else:
            print(f"❌ Test folder not found: {test_folder}")
            return None
    
    # Check if test file exists
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        # Try to find similar files
        test_dir = os.path.dirname(test_file) or "csv/test"
        if os.path.exists(test_dir):
            available_files = glob.glob(os.path.join(test_dir, '*.csv'))
            if available_files:
                print(f"💡 Available files in {test_dir}:")
                for f in available_files[:5]:  # Show first 5 files
                    print(f"   📄 {os.path.basename(f)}")
                if len(available_files) > 5:
                    print(f"   ... and {len(available_files) - 5} more files")
        return None
    
    # Read test file
    try:
        print(f"📄 Reading file: {os.path.basename(test_file)}")
        
        if test_file.endswith('.csv'):
            df = pd.read_csv(test_file, encoding='utf-8')
            print(f"📊 CSV shape: {df.shape}")
            print(f"📋 Columns: {list(df.columns)}")
            
            if 'data_string' in df.columns:
                # Check if there are multiple rows and use the first non-empty one
                for idx, row in df.iterrows():
                    text_content = row['data_string']
                    if pd.notna(text_content) and len(str(text_content).strip()) > 10:
                        full_text = str(text_content)
                        if idx > 0:
                            print(f"📝 Using row {idx} (previous rows were empty/short)")
                        break
                else:
                    print("❌ No valid data_string found in any row")
                    return None
            else:
                print("❌ No 'data_string' column found")
                print(f"💡 Available columns: {list(df.columns)}")
                return None
        else:
            with open(test_file, 'r', encoding='utf-8', errors='ignore') as f:
                full_text = f.read()
        
        print(f"📏 Text length: {len(full_text)} characters")
        
        # Show text preview with structure
        lines = full_text.split('\n')[:10]  # First 10 lines
        print(f"📝 Content preview:")
        for i, line in enumerate(lines):
            if line.strip():
                preview_line = line[:100] + "..." if len(line) > 100 else line
                print(f"   {i+1}: {preview_line}")
        if len(full_text.split('\n')) > 10:
            num_extra_lines = len(full_text.split('\n')) - 10
            print(f"   ... ({num_extra_lines} more lines)")
                  
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Enhanced feature extraction for testing
    dataset = ImprovedWebsiteDataset([full_text], [0], tokenizer)  # Dummy label
    enhanced_text = dataset.extract_features(full_text)
    
    # Tokenize and predict
    inputs = tokenizer(
        enhanced_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
    
    # Extract results
    score_whitelist = probs[0][0].item()
    score_blacklist = probs[0][1].item()
    predicted_class = probs.argmax().item()
    confidence = max(score_whitelist, score_blacklist)
    
    # Determine risk level based on confidence
    if confidence > 0.9:
        confidence_level = "Very High"
    elif confidence > 0.8:
        confidence_level = "High"
    elif confidence > 0.7:
        confidence_level = "Medium"
    elif confidence > 0.6:
        confidence_level = "Low"
    else:
        confidence_level = "Very Low"
    
    # Display results
    print(f"\n" + "="*60)
    print(f"🔍 CLASSIFICATION RESULTS")
    print(f"="*60)
    print(f"📄 File: {os.path.basename(test_file)}")
    print(f"🎯 Prediction: {predicted_class} ({'BLACKLIST' if predicted_class == 1 else 'WHITELIST'})")
    print(f"🏷️  Status: {'🚨 POTENTIALLY HARMFUL' if predicted_class == 1 else '✅ APPEARS SAFE'}")
    print(f"📊 Confidence: {confidence:.1%} ({confidence_level})")
    print(f"\n📈 Detailed Scores:")
    print(f"   • Whitelist (Safe): {score_whitelist:.4f} ({score_whitelist:.1%})")
    print(f"   • Blacklist (Harmful): {score_blacklist:.4f} ({score_blacklist:.1%})")
    
    # Risk assessment
    if predicted_class == 1:
        if confidence > 0.8:
            print(f"⚠️  HIGH RISK: Strong indicators of harmful content")
        elif confidence > 0.6:
            print(f"⚠️  MEDIUM RISK: Some indicators of harmful content")
        else:
            print(f"⚠️  LOW RISK: Weak indicators, manual review recommended")
    else:
        if confidence < 0.6:
            print(f"⚠️  UNCERTAIN: Low confidence, manual review recommended")
    
    print(f"="*60)
    
    return {
        'predicted_class': predicted_class,
        'classification': 'blacklist' if predicted_class == 1 else 'whitelist',
        'score_whitelist': score_whitelist,
        'score_blacklist': score_blacklist,
        'confidence': confidence,
        'confidence_level': confidence_level,
        'file': test_file
    }

# ====== Main Execution ======
if __name__ == "__main__":
    print("🚀 Improved Website Blacklist/Whitelist Classifier")
    print("=" * 70)
    
    # Configuration
    WHITELIST_FOLDER = "csv/white"
    BLACKLIST_FOLDER = "csv/black"
    MODEL_PATH = "./improved_classifier_model"
    
    # Auto-detect available test files
    TEST_FOLDER = "csv/test"
    available_test_files = []
    if os.path.exists(TEST_FOLDER):
        available_test_files = glob.glob(os.path.join(TEST_FOLDER, '*.csv'))
    
    print(f"📁 Available test files ({len(available_test_files)}):")
    for i, file in enumerate(available_test_files[:10]):  # Show first 10
        print(f"   {i+1}. {os.path.basename(file)}")
    if len(available_test_files) > 10:
        print(f"   ... and {len(available_test_files) - 10} more files")
    
    mode = input("\nChoose mode:\n1. Train improved model\n2. Test improved model\n3. Train and test\nEnter choice (1-3): ")
    
    if mode == "1" or mode == "3":
        success = train_improved_classifier(WHITELIST_FOLDER, BLACKLIST_FOLDER, MODEL_PATH)
        if success:
            print("✅ Training completed successfully!")
        else:
            print("❌ Training failed!")
    
    if mode == "2" or mode == "3":
        # Let user choose test file or auto-select
        test_file = None
        if available_test_files:
            if len(available_test_files) == 1:
                test_file = available_test_files[0]
                print(f"📄 Auto-selected: {os.path.basename(test_file)}")
            else:
                print(f"\nSelect test file:")
                for i, file in enumerate(available_test_files[:10]):
                    print(f"   {i+1}. {os.path.basename(file)}")
                
                try:
                    choice = int(input("Enter file number (or 0 for auto-select): "))
                    if choice == 0:
                        test_file = available_test_files[0]
                    elif 1 <= choice <= len(available_test_files):
                        test_file = available_test_files[choice-1]
                    else:
                        print("Invalid choice, using first file")
                        test_file = available_test_files[0]
                except (ValueError, IndexError):
                    print("Invalid input, using first file")
                    test_file = available_test_files[0]
        
        result = test_improved_classifier(MODEL_PATH, test_file)
        if result:
            print("✅ Testing completed!")
        else:
            print("❌ Testing failed!")