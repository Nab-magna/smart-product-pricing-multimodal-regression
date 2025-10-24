import pandas as pd
import numpy as np
import re, string
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import os
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import time

# For FLAVA embeddings
from transformers import FlavaProcessor, FlavaModel
import torch

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("FLAVA-BASED MULTIMODAL ANALYSIS PIPELINE")
print("=" * 80)

# ============================================================================
# 1. LOAD DATA (SUBSET: FIRST 18,750 SAMPLES)
# ============================================================================
print("\n" + "=" * 80)
print("LOADING DATASET")
print("=" * 80)

train_path = "train.csv"
df_full = pd.read_csv(train_path)

# Use only first 1/4 of data (18,750 samples)
df = df_full.iloc[:25000].copy()

print(f"✅ Dataset Loaded Successfully!")
print(f"Full dataset shape: {df_full.shape}")
print(f"Subset shape (1/4): {df.shape}")
print(f"Columns: {df.columns.tolist()}\n")
print(df.head())

# ============================================================================
# 2. DATA CLEANING & PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("DATA CLEANING & PREPROCESSING")
print("=" * 80)

# Check for missing values
print("\n📊 Missing Values:")
print(df.isnull().sum())

# Remove rows with missing critical data
df = df.dropna(subset=['catalog_content', 'image_link', 'price'])
print(f"\nAfter removing missing values: {df.shape}")

# Check for duplicates
print(f"\n🔍 Duplicate Rows: {df.duplicated().sum()}")
df.drop_duplicates(inplace=True)
print(f"Shape after removing duplicates: {df.shape}")

# Text cleaning function
def clean_text(text):
    """Enhanced text cleaning for product catalogs"""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    
    return text

# Apply text cleaning
df["clean_text"] = df["catalog_content"].apply(clean_text)

# Text statistics
df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))
df["char_count"] = df["clean_text"].apply(len)

print("\n📝 Text Statistics:")
print(df[["word_count", "char_count"]].describe())

# ============================================================================
# 3. DOWNLOAD IMAGES
# ============================================================================
print("\n" + "=" * 80)
print("DOWNLOADING IMAGES")
print("=" * 80)

# Create directory for images
image_dir = "product_images"
os.makedirs(image_dir, exist_ok=True)

def download_image(url, save_path, timeout=10, max_retries=2):
    """Download image from URL with retry logic"""
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url, 
                timeout=timeout, 
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert('RGB')
                # Resize to reasonable size to save space
                img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                img.save(save_path, 'JPEG', quality=85)
                return True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retry
            continue
    return False

# Download images
print(f"\n📥 Downloading {len(df)} images...")
success_count = 0
failed_samples = []
valid_indices = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
    save_path = f"{image_dir}/{row['sample_id']}.jpg"
    
    # Skip if already downloaded
    if os.path.exists(save_path):
        success_count += 1
        valid_indices.append(idx)
        continue
    
    if download_image(row['image_link'], save_path):
        success_count += 1
        valid_indices.append(idx)
    else:
        failed_samples.append(row['sample_id'])

print(f"\n✅ Downloaded {success_count}/{len(df)} images ({100*success_count/len(df):.1f}%)")
print(f"❌ Failed: {len(failed_samples)} samples")

# Keep only samples with successfully downloaded images
df = df.loc[valid_indices].reset_index(drop=True)
print(f"\nFinal dataset shape: {df.shape}")

# ============================================================================
# 4. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "=" * 80)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# Price distribution
print("\n💰 Price Statistics:")
print(df["price"].describe())

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 1. Price Distribution
axes[0, 0].hist(df["price"], bins=50, color='skyblue', edgecolor='black')
axes[0, 0].set_title("Price Distribution", fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel("Price ($)")
axes[0, 0].set_ylabel("Frequency")

# 2. Price Distribution (Log Scale)
axes[0, 1].hist(np.log1p(df["price"]), bins=50, color='salmon', edgecolor='black')
axes[0, 1].set_title("Price Distribution (Log Scale)", fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel("Log(Price)")
axes[0, 1].set_ylabel("Frequency")

# 3. Price Boxplot
axes[0, 2].boxplot(df["price"], vert=True)
axes[0, 2].set_title("Price Boxplot", fontsize=14, fontweight='bold')
axes[0, 2].set_ylabel("Price ($)")

# 4. Word Count Distribution
axes[1, 0].hist(df["word_count"], bins=50, color='lightgreen', edgecolor='black')
axes[1, 0].set_title("Word Count Distribution", fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel("Word Count")
axes[1, 0].set_ylabel("Frequency")

# 5. Character Count Distribution
axes[1, 1].hist(df["char_count"], bins=50, color='plum', edgecolor='black')
axes[1, 1].set_title("Character Count Distribution", fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel("Character Count")
axes[1, 1].set_ylabel("Frequency")

# 6. Price vs Word Count Scatter
axes[1, 2].scatter(df["word_count"], df["price"], alpha=0.3, s=10)
axes[1, 2].set_title("Price vs Word Count", fontsize=14, fontweight='bold')
axes[1, 2].set_xlabel("Word Count")
axes[1, 2].set_ylabel("Price ($)")

plt.tight_layout()
plt.savefig('flava_eda_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlation analysis
print("\n🔗 Correlation Matrix:")
correlation = df[["price", "word_count", "char_count"]].corr()
print(correlation)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title("Feature Correlation Heatmap", fontsize=16, fontweight='bold')
plt.savefig('flava_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 5. GENERATE FLAVA EMBEDDINGS
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING FLAVA MULTIMODAL EMBEDDINGS")
print("=" * 80)

# Load FLAVA model and processor
print("\n⏳ Loading FLAVA model (this may take a moment)...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load components separately to avoid chat template issues
from transformers import BertTokenizer, BertModel, ViTImageProcessor, ViTModel

# Load FLAVA components individually
try:
    processor = FlavaProcessor.from_pretrained(
        "facebook/flava-full", 
        local_files_only=False,
        trust_remote_code=True
    )
    model = FlavaModel.from_pretrained("facebook/flava-full")
except:
    print("⚠ Falling back to flava-base due to loading issues...")
    processor = FlavaProcessor.from_pretrained("facebook/flava-base")
    model = FlavaModel.from_pretrained("facebook/flava-base")

model.to(device)
model.eval()

print("✅ FLAVA model loaded successfully!")
print(f"   Model: facebook/flava-full")
print(f"   Multimodal embedding dimension: 768")

# Generate embeddings
print(f"\n🔄 Generating FLAVA embeddings for {len(df)} samples...")

embeddings_list = []
failed_embeddings = []

with torch.no_grad():
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating embeddings"):
        try:
            # Load image
            img_path = f"{image_dir}/{row['sample_id']}.jpg"
            image = Image.open(img_path).convert('RGB')
            
            # Prepare text (FLAVA can handle longer text than ViLT)
            text = row['clean_text']
            words = text.split()
            if len(words) > 77:  # FLAVA max tokens is 77
                text = ' '.join(words[:77])
            
            # Process inputs - FLAVA handles both modalities together
            inputs = processor(
                text=[text], 
                images=[image], 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=77
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get multimodal embeddings
            outputs = model(**inputs)
            
            # FIX: Use pooled output or mean pooling instead of multimodal_embeddings
            # Option 1: Use multimodal pooled output (if available)
            if hasattr(outputs, 'multimodal_pooler_output') and outputs.multimodal_pooler_output is not None:
                embedding = outputs.multimodal_pooler_output.cpu().numpy().flatten()
            
            # Option 2: Use mean pooling over sequence dimension
            elif hasattr(outputs, 'multimodal_embeddings') and outputs.multimodal_embeddings is not None:
                # Mean pooling: average over sequence length dimension
                embedding = outputs.multimodal_embeddings.mean(dim=1).cpu().numpy().flatten()
            
            # Option 3: Use image-text matching score embeddings
            elif hasattr(outputs, 'itm_scores') and outputs.itm_scores is not None:
                embedding = outputs.itm_scores.cpu().numpy().flatten()
            
            # Option 4: Fallback - use CLS token (first token)
            else:
                embedding = outputs.multimodal_embeddings[:, 0, :].cpu().numpy().flatten()
            
            # Ensure consistent embedding size (768 for FLAVA)
            if len(embedding) != 768:
                # If not 768, use mean pooling to get to 768
                if len(embedding) > 768:
                    # Reshape and mean pool if larger
                    embedding = embedding.reshape(-1, 768).mean(axis=0)
                else:
                    # Pad if smaller (shouldn't happen with FLAVA)
                    embedding_padded = np.zeros(768)
                    embedding_padded[:len(embedding)] = embedding
                    embedding = embedding_padded
            
            embeddings_list.append(embedding)
            
        except Exception as e:
            print(f"\n⚠ Error processing sample {row['sample_id']}: {str(e)}")
            failed_embeddings.append(idx)
            # Use zeros with consistent shape
            embeddings_list.append(np.zeros(768))

# Convert to numpy array - now all embeddings should have same shape
embeddings = np.array(embeddings_list)

print(f"\n✅ Embeddings generated!")
print(f"Embedding shape: {embeddings.shape}")
print(f"Embedding dimensions: {embeddings.shape[1]}")
print(f"Failed embeddings: {len(failed_embeddings)}")

# Remove failed samples
if failed_embeddings:
    df = df.drop(failed_embeddings).reset_index(drop=True)
    embeddings = np.delete(embeddings, failed_embeddings, axis=0)
    print(f"Final dataset shape after removing failures: {df.shape}")

# ============================================================================
# 6. VISUALIZE EMBEDDINGS
# ============================================================================
print("\n" + "=" * 80)
print("VISUALIZING EMBEDDINGS")
print("=" * 80)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("\n📊 Reducing dimensions for visualization...")
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings_scaled)

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

# Visualize embeddings
plt.figure(figsize=(12, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                     c=df["price"].values, cmap='viridis', 
                     alpha=0.6, s=50)
plt.colorbar(scatter, label='Price ($)')
plt.title("FLAVA Multimodal Embeddings Visualization (PCA)", fontsize=16, fontweight='bold')
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.savefig('flava_embeddings_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 7. SAVE PROCESSED DATA
# ============================================================================
print("\n" + "=" * 80)
print("SAVING PROCESSED DATA")
print("=" * 80)

print("\n💾 Saving processed data...")

# Save cleaned dataframe
df.to_csv("flava_processed_data.csv", index=False)
print("✅ Saved: flava_processed_data.csv")

# Save embeddings
np.save("flava_embeddings.npy", embeddings)
print("✅ Saved: flava_embeddings.npy")

# Save scaler for later use
import pickle
with open('flava_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Saved: flava_scaler.pkl")

# ============================================================================
# 8. SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

summary = {
    "Total Records (1/4 subset)": len(df),
    "Images Downloaded": success_count,
    "Average Price": f"${df['price'].mean():.2f}",
    "Median Price": f"${df['price'].median():.2f}",
    "Average Word Count": f"{df['word_count'].mean():.1f}",
    "Embedding Dimensions": embeddings.shape[1],
    "Embedding Type": "FLAVA (Facebook's Multimodal)"
}

print("\n📋 Summary Statistics:")
for key, value in summary.items():
    print(f"  {key}: {value}")

print("\n" + "=" * 80)
print("✅ FLAVA MULTIMODAL ANALYSIS COMPLETE!")
print("=" * 80)
print("\n💡 Key Advantages of FLAVA over ViLT:")
print("   • Better aligned vision-language representations")
print("   • Trained on larger and more diverse datasets")
print("   • Stronger multimodal fusion capabilities")
print("   • Improved performance on downstream tasks")