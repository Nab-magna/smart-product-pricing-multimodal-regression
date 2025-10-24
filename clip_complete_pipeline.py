"""
COMPLETE CLIP-BASED SOLUTION FOR PRODUCT PRICE PREDICTION
Full end-to-end pipeline for minimum SMAPE score

This is a complete, production-ready solution that includes:
1. Data loading and preprocessing
2. Image downloading with retry logic
3. CLIP multimodal embedding extraction
4. Advanced feature engineering
5. Multiple model training with K-Fold CV
6. Ensemble with optimized weights
7. Test prediction generation
"""

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

from transformers import CLIPProcessor, CLIPModel
import torch

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, r2_score

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge, HuberRegressor

import matplotlib.pyplot as plt
import seaborn as sns

# ====================================================
# CONFIGURATION
# ====================================================
class Config:
    # Random seed for reproducibility
    SEED = 42
    
    # Data paths
    TRAIN_PATH = "dataset/train.csv"
    TEST_PATH = "dataset/test.csv"
    
    # Image directories
    TRAIN_IMAGE_DIR = "clip_train_images"
    TEST_IMAGE_DIR = "clip_test_images"
    
    # Output
    OUTPUT_PATH = "test_out.csv"
    
    # Model settings
    N_FOLDS = 5
    USE_FULL_DATA = True  # Set False to use subset for testing
    TRAIN_SUBSET = 18750  # Only used if USE_FULL_DATA = False
    
    # Advanced options
    USE_QUANTILE_TRANSFORM = True
    OPTIMIZE_WEIGHTS = True
    
    # Feature engineering
    MAX_TFIDF_FEATURES = 1000
    SVD_COMPONENTS = 100

np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)

# ====================================================
# SMAPE METRIC
# ====================================================
def smape(y_true, y_pred):
    """Symmetric Mean Absolute Percentage Error"""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator < 1e-8, 1e-8, denominator)
    return 100 * np.mean(np.abs(y_pred - y_true) / denominator)

# ====================================================
# TEXT PREPROCESSING
# ====================================================
def clean_text(text):
    """Enhanced text cleaning for product catalogs"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_ipq(text):
    """Extract Item Pack Quantity with pattern matching"""
    patterns = [
        (r'(\d+)\s*[-]?\s*pack', 1.0),
        (r'pack\s*of\s*(\d+)', 1.0),
        (r'(\d+)\s*count', 0.9),
        (r'(\d+)\s*pieces?', 0.9),
        (r'(\d+)\s*pcs?\.?', 0.9),
        (r'set\s*of\s*(\d+)', 0.8),
        (r'(\d+)\s*units?', 0.7),
    ]
    
    for pattern, weight in patterns:
        match = re.search(pattern, text.lower())
        if match:
            qty = int(match.group(1))
            return qty * weight
    return 1.0

def extract_text_features(df):
    """Extract comprehensive text-based features"""
    print("🔧 Extracting text features...")
    
    # Basic cleaning
    df['clean_text'] = df['catalog_content'].apply(clean_text)
    
    # Text statistics
    df['word_count'] = df['clean_text'].str.split().str.len()
    df['char_count'] = df['clean_text'].str.len()
    df['unique_word_count'] = df['clean_text'].apply(lambda x: len(set(x.split())))
    df['avg_word_length'] = df['clean_text'].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
    )
    
    # IPQ extraction
    df['ipq'] = df['catalog_content'].apply(extract_ipq)
    df['log_ipq'] = np.log1p(df['ipq'])
    df['sqrt_ipq'] = np.sqrt(df['ipq'])
    
    # Quality indicators
    df['has_premium'] = df['catalog_content'].str.contains(
        r'premium|luxury|professional|deluxe|pro', case=False, regex=True
    ).astype(int)
    
    df['has_budget'] = df['catalog_content'].str.contains(
        r'budget|economy|basic|value|affordable', case=False, regex=True
    ).astype(int)
    
    # Brand indicators
    df['has_brand'] = df['catalog_content'].str.contains(
        r'brand|branded|authentic|genuine|official', case=False, regex=True
    ).astype(int)
    
    # Size indicators
    df['has_size'] = df['catalog_content'].str.contains(
        r'large|medium|small|xl|xxl|oz|ml|kg|lb', case=False, regex=True
    ).astype(int)
    
    # Technology keywords
    df['has_tech'] = df['catalog_content'].str.contains(
        r'smart|digital|wireless|bluetooth|wifi|usb|led', case=False, regex=True
    ).astype(int)
    
    # Number features
    df['num_count'] = df['catalog_content'].str.count(r'\d+')
    
    print("✅ Text features extracted")
    return df

# ====================================================
# IMAGE DOWNLOAD
# ====================================================
def download_image(url, save_path, timeout=10, max_retries=3):
    """Download image with retry logic and error handling"""
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url, 
                timeout=timeout, 
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert('RGB')
                # Resize to save space while maintaining quality
                img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                img.save(save_path, 'JPEG', quality=85)
                return True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            continue
    return False

def download_images_batch(df, image_dir, desc="Downloading images"):
    """Download all images for a dataframe"""
    os.makedirs(image_dir, exist_ok=True)
    
    print(f"\n📥 {desc}: {len(df)} images...")
    success_count = 0
    valid_indices = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        save_path = f"{image_dir}/{row['sample_id']}.jpg"
        
        # Skip if already exists
        if os.path.exists(save_path):
            success_count += 1
            valid_indices.append(idx)
            continue
        
        if download_image(row['image_link'], save_path):
            success_count += 1
            valid_indices.append(idx)
    
    success_rate = 100 * success_count / len(df)
    print(f"✅ Downloaded {success_count}/{len(df)} images ({success_rate:.1f}%)")
    
    return valid_indices

# ====================================================
# CLIP EMBEDDINGS
# ====================================================
class CLIPEmbeddingExtractor:
    """Extract CLIP embeddings for both images and text"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n⏳ Loading CLIP model on {self.device}...")
        
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.model.to(self.device)
        self.model.eval()
        
        print("✅ CLIP model loaded successfully!")
    
    def extract_batch(self, df, image_dir):
        """Extract embeddings for entire dataframe"""
        print(f"\n🔄 Generating CLIP embeddings for {len(df)} samples...")
        
        image_embeddings = []
        text_embeddings = []
        failed_indices = []
        
        with torch.no_grad():
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="CLIP embeddings"):
                try:
                    # Load image
                    img_path = f"{image_dir}/{row['sample_id']}.jpg"
                    image = Image.open(img_path).convert('RGB')
                    
                    # Get text (CLIP can handle longer text than ViLT)
                    text = row['clean_text'][:200]  # First 200 characters
                    
                    # Process through CLIP
                    inputs = self.processor(
                        text=[text],
                        images=image,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get embeddings
                    outputs = self.model(**inputs)
                    
                    # Extract both image and text embeddings
                    img_emb = outputs.image_embeds.cpu().numpy()[0]
                    txt_emb = outputs.text_embeds.cpu().numpy()[0]
                    
                    image_embeddings.append(img_emb)
                    text_embeddings.append(txt_emb)
                    
                except Exception as e:
                    print(f"\n⚠️ Error on sample {row['sample_id']}: {str(e)}")
                    failed_indices.append(idx)
                    # Use zero embeddings for failed samples
                    image_embeddings.append(np.zeros(512))
                    text_embeddings.append(np.zeros(512))
        
        image_embeddings = np.array(image_embeddings)
        text_embeddings = np.array(text_embeddings)
        
        print(f"✅ Embeddings generated!")
        print(f"   Image embeddings: {image_embeddings.shape}")
        print(f"   Text embeddings: {text_embeddings.shape}")
        print(f"   Failed: {len(failed_indices)}")
        
        return image_embeddings, text_embeddings, failed_indices

# ====================================================
# MODEL TRAINING
# ====================================================
def train_models(X, y, n_folds=5):
    """Train multiple models with K-Fold cross-validation"""
    
    print("\n" + "=" * 100)
    print("MODEL TRAINING WITH K-FOLD CROSS-VALIDATION")
    print("=" * 100)
    
    # Define models
    models = {
        'LightGBM_v1': LGBMRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=8,
            num_leaves=50,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_samples=20,
            reg_alpha=0.3,
            reg_lambda=0.5,
            random_state=Config.SEED,
            verbose=-1,
            n_jobs=-1
        ),
        'LightGBM_v2': LGBMRegressor(
            n_estimators=1500,
            learning_rate=0.05,
            max_depth=6,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_samples=30,
            random_state=Config.SEED + 1,
            verbose=-1,
            n_jobs=-1
        ),
        'XGBoost': XGBRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            reg_alpha=0.3,
            reg_lambda=0.5,
            random_state=Config.SEED,
            tree_method='hist',
            n_jobs=-1,
            verbosity=0
        ),
        'CatBoost': CatBoostRegressor(
            iterations=1500,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=3,
            random_state=Config.SEED,
            verbose=False,
            thread_count=-1
        ),
        'Ridge': Ridge(alpha=50.0),
        'Huber': HuberRegressor(epsilon=1.5, alpha=10.0)
    }
    
    # Target transformation
    if Config.USE_QUANTILE_TRANSFORM:
        print("\n📊 Applying Quantile Transformation to target...")
        qt = QuantileTransformer(output_distribution='normal', random_state=Config.SEED)
        y_transformed = qt.fit_transform(y.reshape(-1, 1)).ravel()
        print("✅ Quantile transformation applied")
    else:
        print("\n📊 Applying Log Transformation to target...")
        y_transformed = np.log1p(y)
        qt = None
    
    # Create price bins for stratification
    price_bins = pd.qcut(y, q=10, labels=False, duplicates='drop')
    
    # K-Fold CV
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Config.SEED)
    
    # Storage for OOF predictions
    oof_predictions = {name: np.zeros(len(X)) for name in models.keys()}
    trained_models = {}
    model_scalers = {}
    
    # Train each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, price_bins)):
        print(f"\n{'='*100}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'='*100}")
        
        # Scale features per fold
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_val = scaler.transform(X[val_idx])
        
        if fold == 0:
            # Save first scaler for test predictions
            model_scalers['scaler'] = scaler
        
        y_train = y_transformed[train_idx]
        y_val = y_transformed[val_idx]
        y_val_original = y[val_idx]
        
        # Train each model
        for name, model in models.items():
            print(f"\n🎯 Training {name}...")
            
            if fold == 0:
                trained_models[name] = model
            
            # Train
            if 'LightGBM' in name or 'XGBoost' in name or 'CatBoost' in name:
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                model.fit(X_train, y_train)
            
            # Predict on validation
            oof_pred_transformed = model.predict(X_val)
            
            # Inverse transform
            if Config.USE_QUANTILE_TRANSFORM and qt is not None:
                oof_pred = qt.inverse_transform(oof_pred_transformed.reshape(-1, 1)).ravel()
            else:
                oof_pred = np.expm1(oof_pred_transformed)
            
            # Store OOF predictions
            oof_predictions[name][val_idx] = oof_pred
            
            # Calculate validation SMAPE
            val_smape = smape(y_val_original, oof_pred)
            val_rmse = np.sqrt(mean_squared_error(y_val_original, oof_pred))
            val_r2 = r2_score(y_val_original, oof_pred)
            
            print(f"   SMAPE: {val_smape:.4f}% | RMSE: ${val_rmse:.2f} | R²: {val_r2:.4f}")
    
    # Calculate overall OOF scores
    print("\n" + "=" * 100)
    print("OVERALL OUT-OF-FOLD PERFORMANCE")
    print("=" * 100)
    
    results = []
    for name in models.keys():
        oof_smape = smape(y, oof_predictions[name])
        oof_rmse = np.sqrt(mean_squared_error(y, oof_predictions[name]))
        oof_r2 = r2_score(y, oof_predictions[name])
        
        results.append({
            'Model': name,
            'OOF_SMAPE': oof_smape,
            'OOF_RMSE': oof_rmse,
            'OOF_R2': oof_r2
        })
        
        print(f"{name:15s}: SMAPE {oof_smape:.4f}% | RMSE ${oof_rmse:.2f} | R² {oof_r2:.4f}")
    
    results_df = pd.DataFrame(results).sort_values('OOF_SMAPE')
    
    # Optimize ensemble weights
    weights = None
    if Config.OPTIMIZE_WEIGHTS:
        print("\n" + "=" * 100)
        print("OPTIMIZING ENSEMBLE WEIGHTS")
        print("=" * 100)
        
        from scipy.optimize import minimize
        
        pred_array = np.array([oof_predictions[name] for name in models.keys()]).T
        
        def objective(w):
            w = w / w.sum()
            ensemble_pred = pred_array @ w
            return smape(y, ensemble_pred)
        
        initial_weights = np.ones(len(models)) / len(models)
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
        bounds = [(0, 1) for _ in range(len(models))]
        
        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        weights = result.x / result.x.sum()
        
        print("\n📊 Optimized Weights:")
        weight_dict = {}
        for name, weight in zip(models.keys(), weights):
            weight_dict[name] = weight
            print(f"   {name:15s}: {weight:.4f}")
        
        # Calculate ensemble performance
        ensemble_pred = pred_array @ weights
        ensemble_smape = smape(y, ensemble_pred)
        ensemble_rmse = np.sqrt(mean_squared_error(y, ensemble_pred))
        ensemble_r2 = r2_score(y, ensemble_pred)
        
        print(f"\n🏆 Ensemble Performance:")
        print(f"   SMAPE: {ensemble_smape:.4f}%")
        print(f"   RMSE: ${ensemble_rmse:.2f}")
        print(f"   R²: {ensemble_r2:.4f}")
        
        weights = weight_dict
    
    return trained_models, model_scalers, weights, qt, results_df, oof_predictions

# ====================================================
# MAIN TRAINING PIPELINE
# ====================================================
def main_training():
    """Complete training pipeline"""
    
    print("=" * 100)
    print("CLIP-BASED PRODUCT PRICE PREDICTION - TRAINING PIPELINE")
    print("=" * 100)
    
    # Load training data
    print("\n📂 Loading training data...")
    df_full = pd.read_csv(Config.TRAIN_PATH)
    
    if Config.USE_FULL_DATA:
        df = df_full.copy()
        print(f"✅ Using full training set: {len(df):,} samples")
    else:
        df = df_full.iloc[:Config.TRAIN_SUBSET].copy()
        print(f"✅ Using subset: {len(df):,} samples (for testing)")
    
    # Handle missing values
    print("\n🧹 Cleaning data...")
    print(f"   Missing catalog_content: {df['catalog_content'].isna().sum()}")
    print(f"   Missing image_link: {df['image_link'].isna().sum()}")
    print(f"   Missing price: {df['price'].isna().sum()}")
    
    df = df.dropna(subset=['catalog_content', 'image_link', 'price'])
    df = df[df['price'] > 0].reset_index(drop=True)
    print(f"✅ Clean dataset: {len(df):,} samples")
    
    # Extract text features
    df = extract_text_features(df)
    
    # Download images
    valid_indices = download_images_batch(df, Config.TRAIN_IMAGE_DIR, "Downloading training images")
    df = df.loc[valid_indices].reset_index(drop=True)
    print(f"\n✅ Final training set with images: {len(df):,} samples")
    
    # Extract CLIP embeddings
    clip_extractor = CLIPEmbeddingExtractor()
    img_embeddings, txt_embeddings, failed = clip_extractor.extract_batch(df, Config.TRAIN_IMAGE_DIR)
    
    # Remove failed samples
    if failed:
        print(f"\n⚠️ Removing {len(failed)} failed samples...")
        df = df.drop(failed).reset_index(drop=True)
        img_embeddings = np.delete(img_embeddings, failed, axis=0)
        txt_embeddings = np.delete(txt_embeddings, failed, axis=0)
    
    # TF-IDF features
    print("\n📝 Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(
        max_features=Config.MAX_TFIDF_FEATURES,
        ngram_range=(1, 3),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True
    )
    
    tfidf_features = tfidf.fit_transform(df['clean_text'])
    
    # SVD dimensionality reduction
    svd = TruncatedSVD(n_components=Config.SVD_COMPONENTS, random_state=Config.SEED)
    tfidf_reduced = svd.fit_transform(tfidf_features)
    
    print(f"   TF-IDF variance explained: {svd.explained_variance_ratio_.sum():.3f}")
    
    # Combine all features
    print("\n🔗 Combining all features...")
    
    handcrafted_cols = [
        'word_count', 'char_count', 'unique_word_count', 'avg_word_length',
        'log_ipq', 'sqrt_ipq', 'has_premium', 'has_budget', 
        'has_brand', 'has_size', 'has_tech', 'num_count'
    ]
    
    handcrafted_features = df[handcrafted_cols].values
    
    X = np.hstack([
        img_embeddings,        # 512 dimensions - CLIP image
        txt_embeddings,        # 512 dimensions - CLIP text
        tfidf_reduced,         # 100 dimensions - TF-IDF
        handcrafted_features   # 12 dimensions - handcrafted
    ])
    
    y = df['price'].values
    
    print(f"\n✅ Feature Matrix:")
    print(f"   Shape: {X.shape}")
    print(f"   - CLIP image embeddings: 512")
    print(f"   - CLIP text embeddings: 512")
    print(f"   - TF-IDF (SVD): {Config.SVD_COMPONENTS}")
    print(f"   - Handcrafted features: 12")
    print(f"   Total: {X.shape[1]}")
    
    print(f"\n💰 Price Statistics:")
    print(f"   Min: ${y.min():.2f}")
    print(f"   Max: ${y.max():.2f}")
    print(f"   Mean: ${y.mean():.2f}")
    print(f"   Median: ${np.median(y):.2f}")
    print(f"   Std: ${y.std():.2f}")
    
    # Remove extreme outliers
    print("\n🔍 Removing extreme outliers...")
    q_low = np.percentile(y, 0.1)
    q_high = np.percentile(y, 99.9)
    mask = (y >= q_low) & (y <= q_high)
    
    X = X[mask]
    y = y[mask]
    
    print(f"   Removed {(~mask).sum()} outliers")
    print(f"   Final dataset: {X.shape[0]:,} samples")
    
    # Train models
    trained_models, scalers, weights, qt, results_df, oof_predictions = train_models(
        X, y, n_folds=Config.N_FOLDS
    )
    
    # Save everything
    print("\n" + "=" * 100)
    print("SAVING MODELS AND ARTIFACTS")
    print("=" * 100)
    
    pipeline_dict = {
        'models': trained_models,
        'scaler': scalers['scaler'],
        'weights': weights,
        'tfidf': tfidf,
        'svd': svd,
        'qt': qt,
        'handcrafted_cols': handcrafted_cols,
        'results': results_df,
        'config': {
            'use_quantile_transform': Config.USE_QUANTILE_TRANSFORM,
            'n_folds': Config.N_FOLDS
        }
    }
    
    with open('clip_pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline_dict, f)
    print("✅ Saved: clip_pipeline.pkl")
    
    results_df.to_csv('clip_results.csv', index=False)
    print("✅ Saved: clip_results.csv")
    
    # Visualize results
    print("\n📊 Creating visualizations...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Model comparison
    axes[0].barh(results_df['Model'], results_df['OOF_SMAPE'], color='skyblue', edgecolor='black')
    axes[0].set_xlabel('SMAPE (%)', fontsize=12)
    axes[0].set_title('Model Performance Comparison (OOF SMAPE)', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    
    for i, v in enumerate(results_df['OOF_SMAPE']):
        axes[0].text(v, i, f' {v:.2f}%', va='center')
    
    # Best model predictions
    best_model_name = results_df.iloc[0]['Model']
    best_oof = oof_predictions[best_model_name]
    
    axes[1].scatter(y, best_oof, alpha=0.4, s=10)
    axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[1].set_xlabel('Actual Price ($)', fontsize=12)
    axes[1].set_ylabel('Predicted Price ($)', fontsize=12)
    axes[1].set_title(f'{best_model_name} - Actual vs Predicted', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('clip_training_results.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: clip_training_results.png")
    
    # Summary
    print("\n" + "=" * 100)
    print("TRAINING COMPLETE!")
    print("=" * 100)
    
    print(f"\n🏆 Best Model: {results_df.iloc[0]['Model']}")
    print(f"   OOF SMAPE: {results_df.iloc[0]['OOF_SMAPE']:.4f}%")
    print(f"   OOF RMSE: ${results_df.iloc[0]['OOF_RMSE']:.2f}")
    print(f"   OOF R²: {results_df.iloc[0]['OOF_R2']:.4f}")
    
    if weights:
        ensemble_pred = np.zeros(len(y))
        for name, weight in weights.items():
            ensemble_pred += oof_predictions[name] * weight
        
        ensemble_smape = smape(y, ensemble_pred)
        print(f"\n🎯 Optimized Ensemble:")
        print(f"   SMAPE: {ensemble_smape:.4f}%")
    
    print("\n📁 Generated Files:")
    print("   - clip_pipeline.pkl (all models and transformers)")
    print("   - clip_results.csv (model performance metrics)")
    print("   - clip_training_results.png (visualizations)")
    
    print("\n✅ Ready for test predictions!")
    print("   Run: python clip_test_prediction.py")
    
    return pipeline_dict

# ====================================================
# RUN TRAINING
# ====================================================
if __name__ == "__main__":
    try:
        pipeline = main_training()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
