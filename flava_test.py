import pandas as pd
import numpy as np
import re, string
import os
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

from transformers import FlavaProcessor, FlavaModel
import torch

print("=" * 100)
print("FLAVA-BASED TEST PREDICTION SCRIPT")
print("=" * 100)

# ============================================================================
# CONFIGURATION
# ============================================================================

USE_SIMPLE_AVERAGE = True  # Set to True to skip weighted ensemble

# ============================================================================
# 1. LOAD TEST DATA
# ============================================================================
print(f"\n📂 Loading first 18750 test samples...")
test_df = pd.read_csv("test.csv")
test_df = test_df.iloc[:25000].copy()
print(f"✅ Test data loaded: {test_df.shape}")

# ============================================================================
# 2. TEXT PREPROCESSING
# ============================================================================
print("\n🔧 Preprocessing text...")

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

test_df['clean_text'] = test_df['catalog_content'].apply(clean_text)
test_df['word_count'] = test_df['clean_text'].str.split().str.len()
test_df['char_count'] = test_df['clean_text'].str.len()
print("✅ Text preprocessing complete")

# ============================================================================
# 3. DOWNLOAD IMAGES
# ============================================================================
print("\n📥 Downloading test images...")
test_image_dir = "test_product_images"
os.makedirs(test_image_dir, exist_ok=True)

def download_image(url, save_path, timeout=10, max_retries=2):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=timeout, 
                                  headers={'User-Agent': 'Mozilla/5.0'})
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert('RGB')
                img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                img.save(save_path, 'JPEG', quality=85)
                return True
        except:
            if attempt < max_retries - 1:
                time.sleep(1)
    return False

success_count = 0
valid_indices = []

for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Downloading"):
    save_path = f"{test_image_dir}/{row['sample_id']}.jpg"
    
    if os.path.exists(save_path):
        success_count += 1
        valid_indices.append(idx)
        continue
    
    if download_image(row['image_link'], save_path):
        success_count += 1
        valid_indices.append(idx)

print(f"✅ Downloaded {success_count}/{len(test_df)} images")

test_df = test_df.loc[valid_indices].reset_index(drop=True)
print(f"Final test set: {test_df.shape}")

# ============================================================================
# 4. GENERATE FLAVA EMBEDDINGS (FIXED VERSION)
# ============================================================================
print("\n🔄 Generating FLAVA embeddings...")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("⏳ Loading FLAVA model...")
processor = FlavaProcessor.from_pretrained("facebook/flava-full")
model = FlavaModel.from_pretrained("facebook/flava-full")
model.to(device)
model.eval()
print("✅ FLAVA model loaded")

test_embeddings_list = []

with torch.no_grad():
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Embeddings"):
        try:
            img_path = f"{test_image_dir}/{row['sample_id']}.jpg"
            image = Image.open(img_path).convert('RGB')
            
            text = row['clean_text']
            words = text.split()
            if len(words) > 77:  # FLAVA max tokens
                text = ' '.join(words[:77])
            
            # Process with FLAVA
            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=77
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get multimodal embeddings - FIXED VERSION
            outputs = model(**inputs)
            
            # Use mean pooling over sequence dimension to get consistent 768-dim vector
            # multimodal_embeddings shape: (1, sequence_length, 768)
            # We take mean over sequence_length to get (1, 768) then flatten to (768,)
            embedding = outputs.multimodal_embeddings.mean(dim=1).cpu().numpy().flatten()
            
            # Ensure it's exactly 768 dimensions
            if len(embedding) != 768:
                if len(embedding) > 768:
                    # If larger, take first 768 elements
                    embedding = embedding[:768]
                else:
                    # If smaller, pad with zeros (shouldn't happen with FLAVA)
                    embedding_padded = np.zeros(768)
                    embedding_padded[:len(embedding)] = embedding
                    embedding = embedding_padded
            
            test_embeddings_list.append(embedding)
            
        except Exception as e:
            print(f"\n⚠️ Error on sample {row['sample_id']}: {e}")
            # Use zeros with consistent shape
            test_embeddings_list.append(np.zeros(768))

test_embeddings = np.array(test_embeddings_list)
print(f"✅ Embeddings shape: {test_embeddings.shape}")

# ============================================================================
# SAVE TEST EMBEDDINGS FOR FUTURE USE
# ============================================================================
print("\n💾 Saving test embeddings...")

# Save embeddings
np.save('flava_test_embeddings.npy', test_embeddings)

# Also save the corresponding sample IDs for reference
np.save('flava_test_sample_ids.npy', test_df['sample_id'].values)

print(f"✅ Saved test embeddings: flava_test_embeddings.npy")
print(f"✅ Saved sample IDs: flava_test_sample_ids.npy")
print(f"   Embeddings shape: {test_embeddings.shape}")

# ============================================================================
# 5. PREPARE FEATURES
# ============================================================================
print("\n📊 Preparing features...")

text_features = test_df[["word_count", "char_count"]].values

# Add derived features to match training
test_df['price_per_word'] = 0  # Placeholder, will be predicted
test_df['price_per_char'] = 0  # Placeholder
test_df['word_density'] = test_df['word_count'] / (test_df['char_count'] + 1)
derived_features = test_df[['price_per_word', 'price_per_char', 'word_density']].values

X_test = np.hstack([test_embeddings, text_features, derived_features])

print(f"Feature shape: {X_test.shape}")

# Load and apply scaler
try:
    with open('flava_models/flava_feature_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    X_test_scaled = scaler.transform(X_test)
    print("✅ Features scaled")
except FileNotFoundError:
    print("⚠️ Scaler not found, trying alternative path...")
    try:
        with open('flava_feature_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        X_test_scaled = scaler.transform(X_test)
        print("✅ Features scaled")
    except FileNotFoundError:
        print("⚠️ Scaler not found, using unscaled features")
        X_test_scaled = X_test

# ============================================================================
# 6. LOAD MODELS AND PREDICT
# ============================================================================
print("\n🎯 Loading models and generating predictions...")

model_names = ['XGBoost', 'LightGBM', 'CatBoost']
predictions_dict = {}

for model_name in model_names:
    try:
        # Try multiple possible file locations
        possible_paths = [
            f'flava_models/flava_model_{model_name}.pkl',
            f'flava_model_{model_name}.pkl',
            f'model_{model_name}.pkl'
        ]
        
        model_loaded = False
        for model_path in possible_paths:
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    trained_model = pickle.load(f)
                model_loaded = True
                print(f"✅ Loaded {model_name} from {model_path}")
                break
        
        if not model_loaded:
            print(f"⚠️ {model_name} model not found in any location, skipping")
            continue
        
        # Make predictions (these are in log space)
        pred_log = trained_model.predict(X_test_scaled)
        
        # Convert back to original price space
        pred = np.expm1(pred_log)
        
        # Clean predictions
        pred = np.clip(pred, 0.01, 1000)  # Reasonable price range
        
        predictions_dict[model_name] = pred
        
        print(f"   {model_name} predictions:")
        print(f"     Mean: ${pred.mean():.2f}, Min: ${pred.min():.2f}, Max: ${pred.max():.2f}")
        print(f"     Has NaN: {np.isnan(pred).any()}, Has Inf: {np.isinf(pred).any()}")
        
    except Exception as e:
        print(f"❌ Error loading/predicting with {model_name}: {e}")

if not predictions_dict:
    print("\n❌ ERROR: No models loaded successfully!")
    exit(1)

# ============================================================================
# 7. CREATE ENSEMBLE
# ============================================================================
print("\n📊 Creating ensemble predictions...")

if USE_SIMPLE_AVERAGE:
    print("Using SIMPLE AVERAGE ensemble")
    all_predictions = np.array(list(predictions_dict.values()))
    final_predictions = np.mean(all_predictions, axis=0)
else:
    # Try weighted ensemble
    try:
        results_df = pd.read_csv('flava_models/flava_model_results.csv')
        weights = {}
        
        for _, row in results_df.iterrows():
            if row['Model'] in predictions_dict:
                if pd.notna(row['Test SMAPE']) and row['Test SMAPE'] > 0:
                    # Higher weight for lower SMAPE (better performance)
                    weights[row['Model']] = 1 / row['Test SMAPE']
        
        if weights and sum(weights.values()) > 0:
            print("Using WEIGHTED ensemble based on Test SMAPE")
            total_weight = sum(weights.values())
            final_predictions = np.zeros(len(test_df))
            
            for model_name, pred in predictions_dict.items():
                if model_name in weights:
                    weight = weights[model_name] / total_weight
                    final_predictions += pred * weight
                    print(f"   {model_name}: weight = {weight:.4f}")
        else:
            raise ValueError("Invalid weights")
            
    except Exception as e:
        print(f"⚠️ Weighted ensemble failed ({e}), using simple average")
        all_predictions = np.array(list(predictions_dict.values()))
        final_predictions = np.mean(all_predictions, axis=0)

# Final cleaning and validation
final_predictions = np.clip(final_predictions, 0.01, 1000)  # Ensure reasonable prices
final_predictions = np.nan_to_num(final_predictions, nan=10.0)  # Replace NaN with $10

print(f"\n📈 Final ensemble predictions:")
print(f"   Mean: ${final_predictions.mean():.2f}")
print(f"   Median: ${np.median(final_predictions):.2f}")
print(f"   Min: ${final_predictions.min():.2f}")
print(f"   Max: ${final_predictions.max():.2f}")
print(f"   Std: ${final_predictions.std():.2f}")
print(f"   Has NaN: {np.isnan(final_predictions).any()}")
print(f"   Has Inf: {np.isinf(final_predictions).any()}")

# ============================================================================
# 8. CREATE SUBMISSION
# ============================================================================
print("\n💾 Creating submission file...")

submission = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'price': final_predictions
})

# Final validation
print(f"\n📋 Submission validation:")
print(f"   Total rows: {len(submission)}")
print(f"   Missing values: {submission.isnull().sum().sum()}")
print(f"   Unique sample IDs: {submission['sample_id'].nunique()}")
print(f"   Price statistics:")
print(f"     - Min: ${submission['price'].min():.2f}")
print(f"     - 25%: ${submission['price'].quantile(0.25):.2f}")
print(f"     - 50%: ${submission['price'].median():.2f}")
print(f"     - 75%: ${submission['price'].quantile(0.75):.2f}")
print(f"     - Max: ${submission['price'].max():.2f}")

# Save submission
submission.to_csv('flava_test_out.csv', index=False)
print("\n✅ Saved: flava_test_out.csv")

# Also save individual model predictions for comparison
for model_name, pred in predictions_dict.items():
    individual_submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': pred
    })
    individual_submission.to_csv(f'flava_test_out_{model_name}.csv', index=False)
    print(f"✅ Saved individual: flava_test_out_{model_name}.csv")

# Show first few predictions
print("\n🔍 First 10 predictions:")
print(submission.head(10).to_string(index=False))

print("\n" + "=" * 100)
print("✅ FLAVA TEST PREDICTION COMPLETE!")
print("=" * 100)
print(f"\n📊 Summary:")
print(f"   • Processed {len(test_df)} test samples")
print(f"   • Used {len(predictions_dict)} models for ensemble")
print(f"   • Average predicted price: ${final_predictions.mean():.2f}")
print(f"   • Price range: ${final_predictions.min():.2f} - ${final_predictions.max():.2f}")
print(f"\n💡 FLAVA Advantages:")
print("   • Superior multimodal understanding")
print("   • Better alignment between image and text features")
print("   • More robust to diverse product types")
print("   • Potentially lower SMAPE than unimodal approaches")