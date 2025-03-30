#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install transformers --upgrade')


# In[2]:


import pandas as pd
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW  
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[3]:


# Cell 2: Force PyTorch-only BERT
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disable TensorFlow logging
from transformers import logging
logging.set_verbosity_error()  # Disable transformers warnings


# In[4]:


import torch
print(f"Installed PyTorch: {torch.__version__}")  # Must have double underscores
print(f"Available attributes: {[x for x in dir(torch) if 'version' in x]}")
print(f"CUDA available: {torch.cuda.is_available()}")


# In[5]:


import torch

# CORRECT PyTorch version check (DOUBLE underscore)
print(f"PyTorch version: {torch.__version__}")  # <-- MUST HAVE DOUBLE UNDERSCORES

# GPU check
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")  # This one uses single .version
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA not available (using CPU)")

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# In[6]:


# Cell 4: Data loading (adapt to your dataset)
import pandas as pd
import re  # <-- ADD THIS IMPORT
from sklearn.model_selection import train_test_split

# Load and preprocess data
df = pd.read_csv('tweets_dataset.csv')[['tweet', 'label']].dropna()

# Clean tweets by removing @mentions
df['cleaned_tweet'] = df['tweet'].apply(lambda x: re.sub(r'@[^\s]+', '', str(x)))

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    df['cleaned_tweet'], 
    df['label'], 
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)


# In[7]:


# Cell 4: Environment Reset
from torch.utils.data import Dataset
import torch

# Delete any existing problematic class
if 'TweetDataset' in globals():
    del TweetDataset

# Verify cleanup
assert 'TweetDataset' not in globals(), "RESTART YOUR KERNEL NOW (Kernel → Restart)"
print("Environment ready for clean class definition")


# In[8]:


from torch.utils.data import Dataset
import torch
import pandas as pd
from transformers import BertTokenizer  # Added import

class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):  # DOUBLE underscore __init_
        """Initialize dataset with texts and labels"""
        self.texts = texts.reset_index(drop=True) if hasattr(texts, 'reset_index') else texts
        self.labels = labels.reset_index(drop=True) if hasattr(labels, 'reset_index') else labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, idx):  # DOUBLE underscore __getitem_
        """Get one sample by index"""
        text = str(self.texts.iloc[idx] if hasattr(self.texts, 'iloc') else self.texts[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        }
    
    def __len__(self):  # DOUBLE underscore __len_
        """Get total number of samples"""
        return len(self.texts)

# ===== VERIFICATION TEST =====
print("Testing TweetDataset class...")

# Create test data
test_texts = pd.Series(["Feeling depressed today", "I'm happy"])
test_labels = pd.Series([1, 0])
test_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

try:
    # Test initialization
    test_ds = TweetDataset(test_texts, test_labels, test_tokenizer)
    print("✓ Initialization successful")
    
    # Test _len_
    assert len(test_ds) == 2, "Length test failed"
    print("✓ _len_ works")
    
    # Test _getitem_
    sample = test_ds[0]
    assert 'input_ids' in sample, "Missing input_ids"
    assert 'attention_mask' in sample, "Missing attention_mask"
    assert sample['label'] in [0, 1], "Invalid label"
    print("✓ _getitem_ works")
    
    print("✅ ALL TESTS PASSED - CLASS IS CORRECT")
except Exception as e:
    print(f"❌ TEST FAILED: {str(e)}")
    print("Please RESTART your kernel and try again")


# In[9]:


# Cell 6: Depression Data Preparation (FINAL FIXED VERSION)
import pandas as pd
import numpy as np  # <-- ADD THIS IMPORT
from sklearn.model_selection import train_test_split

# 1. Load dataset
df = pd.read_csv(r'C:\Users\HP\tweets_dataset.csv')

# 2. Clean and process labels
print("Before cleaning:")
print("Missing labels:", df['label'].isna().sum())
print("Label distribution:\n", df['label'].value_counts(dropna=False))

# Handle missing/invalid labels
df = df.dropna(subset=['label'])  # Remove rows with missing labels
df = df[~df['label'].isin([np.inf, -np.inf])]  # Now works with np imported

# Convert to integers (1=depression, 0=non-depression)
y = df['label'].astype(int)
X = df['tweet']

# 3. Stratified split
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Verification
print("\nAfter cleaning:")
print(f"Total samples: {len(X)} (Depression: {y.sum()} = {y.mean():.1%})")
print(f"\nTraining set: {len(X_train)} samples (Depression rate: {y_train.mean():.1%})")
print(f"Validation set: {len(X_val)} samples (Depression rate: {y_val.mean():.1%})")

# Show sample distribution
print("\nClass distribution in training set:")
print(y_train.value_counts())


# In[ ]:


from transformers import BertForSequenceClassification

# Define model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2  # 2 classes (depression/non-depression)
).to(device)  # Uses GPU if available, else CPU

print("✅ Model defined!")


# In[ ]:


from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5)  # Standard learning rate for BERT
print("✅ Optimizer ready!")


# In[ ]:


from torch.utils.data import Dataset
import torch
from transformers import BertTokenizer

class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        }
    
    def __len__(self):
        return len(self.texts)


# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('tweets_dataset.csv')[['tweet', 'label']].dropna()
df['cleaned_tweet'] = df['tweet'].apply(lambda x: re.sub(r'@\w+', '', str(x)))

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    df['cleaned_tweet'], 
    df['label'], 
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# Initialize tokenizer and create datasets
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_data = TweetDataset(X_train, y_train, tokenizer)
val_data = TweetDataset(X_val, y_val, tokenizer)


# In[ ]:


# 1. IMPORTS (ADD 're' AT THE TOP)
import pandas as pd
import torch
import re  # <-- THIS IS THE CRITICAL MISSING IMPORT
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split

# 2. DATASET CLASS
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        }
    
    def __len__(self):
        return len(self.texts)

# 3. LOAD AND CLEAN DATA (NOW WILL WORK)
df = pd.read_csv('tweets_dataset.csv')[['tweet', 'label']].dropna()
df['cleaned_tweet'] = df['tweet'].apply(lambda x: re.sub(r'@\w+', '', str(x)))  # Now works!

# Rest of your code...


# In[ ]:


import torch

# Define device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# In[ ]:


# COMPLETE FIXED CODE (pastes directly into your notebook)
from sklearn.metrics import classification_report
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np

# 1. Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. Load model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)

# 3. CREATE VALIDATION DATA (if missing) - REAL FIX IS HERE
# ======================================================
# Replace this with YOUR actual validation data loading code!
# This is just a working placeholder:
val_inputs = torch.randint(0, 1000, (64, 512))  # 64 fake samples
val_labels = torch.randint(0, 2, (64,))         # Fake binary labels
val_data = TensorDataset(val_inputs, val_labels)
# ======================================================

# 4. Evaluation
model.eval()
predictions, true_labels = [], []

with torch.no_grad():
    for batch in DataLoader(val_data, batch_size=16):
        inputs = {'input_ids': batch[0].to(device)}  # Adapt keys to your data
        outputs = model(**inputs)
        predictions.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
        true_labels.extend(batch[1].cpu().numpy())  # Assuming labels are index 1

print(classification_report(true_labels, predictions))


# In[ ]:




