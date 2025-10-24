import pandas as pd
import numpy as np
import re, string
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# For SBERT embeddings
from sentence_transformers import SentenceTransformer

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# 1. LOAD TEST DATA
# ============================================================================
print("=" * 80)
print("LOADING TEST DATASET")
print("=" * 80)

test_path = "test.csv"
df_test = pd.read_csv(test_path)

print(f"✅ Test Dataset Loaded Successfully!")
print(f"Shape: {df_test.shape}")
print(f"Columns: {df_test.columns.tolist()}\n")
print(df_test.head())

# ============================================================================
# 2. DATA CLEANING & PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("DATA CLEANING & PREPROCESSING")
print("=" * 80)

# Check for missing values
print("\n📊 Missing Values:")
print(df_test.isnull().sum())

# Check for duplicates
print(f"\n🔍 Duplicate Rows: {df_test.duplicated().sum()}")
df_test.drop_duplicates(inplace=True)
print(f"Shape after removing duplicates: {df_test.shape}")

# Text cleaning function (same as training)
def clean_text(text):
    """Enhanced text cleaning for product catalogs"""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)  # Remove punctuation
    text = re.sub(r"\d+", " ", text)  # Remove numbers (optional)
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    
    return text

# Apply text cleaning
df_test["clean_text"] = df_test["catalog_content"].apply(clean_text)

# Text statistics
df_test["word_count"] = df_test["clean_text"].apply(lambda x: len(x.split()))
df_test["char_count"] = df_test["clean_text"].apply(len)

print("\n📝 Text Statistics:")
print(df_test[["word_count", "char_count"]].describe())

# ============================================================================
# 3. BASIC EDA (Without Price)
# ============================================================================
print("\n" + "=" * 80)
print("EXPLORATORY DATA ANALYSIS (TEST SET)")
print("=" * 80)

# Visualizations
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Word Count Distribution
axes[0].hist(df_test["word_count"], bins=50, color='lightgreen', edgecolor='black')
axes[0].set_title("Word Count Distribution", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Word Count")
axes[0].set_ylabel("Frequency")

# 2. Character Count Distribution
axes[1].hist(df_test["char_count"], bins=50, color='plum', edgecolor='black')
axes[1].set_title("Character Count Distribution", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Character Count")
axes[1].set_ylabel("Frequency")

# 3. Word Count Boxplot
axes[2].boxplot(df_test["word_count"], vert=True)
axes[2].set_title("Word Count Boxplot", fontsize=14, fontweight='bold')
axes[2].set_ylabel("Word Count")

plt.tight_layout()
plt.savefig('test_eda_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 4. OUTLIER HANDLING (Apply same preprocessing as training)
# ============================================================================
print("\n" + "=" * 80)
print("OUTLIER DETECTION & HANDLING")
print("=" * 80)

def detect_outliers_iqr(data, column):
    """Detect outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Detect outliers in word count
print("\n🔍 Word Count Outliers (IQR Method):")
word_outliers_iqr, word_lower, word_upper = detect_outliers_iqr(df_test, "word_count")
print(f"Number of outliers: {len(word_outliers_iqr)}")
print(f"Lower bound: {word_lower:.2f}")
print(f"Upper bound: {word_upper:.2f}")
print(f"Percentage: {len(word_outliers_iqr)/len(df_test)*100:.2f}%")

# Handle outliers - Capping (Winsorization)
df_test_clean = df_test.copy()
df_test_clean["word_count"] = df_test_clean["word_count"].clip(lower=word_lower, upper=word_upper)

print(f"\n📊 Dataset shapes:")
print(f"Original: {df_test.shape}")
print(f"After capping: {df_test_clean.shape}")

# ============================================================================
# 5. GENERATE SBERT EMBEDDINGS
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING SBERT EMBEDDINGS FOR TEST DATA")
print("=" * 80)

# Load SBERT model (same model as training)
print("\n⏳ Loading SBERT model (this may take a moment)...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # Same model as training
print("✅ Model loaded successfully!")

# Generate embeddings for entire test set
print(f"\n🔄 Generating embeddings for {len(df_test_clean)} test samples...")
test_embeddings = model.encode(
    df_test_clean["clean_text"].tolist(),
    show_progress_bar=True,
    batch_size=100
)

print(f"✅ Embeddings generated!")
print(f"Embedding shape: {test_embeddings.shape}")
print(f"Embedding dimensions: {test_embeddings.shape[1]}")

# Add embeddings to dataframe
df_test_clean["embeddings"] = list(test_embeddings)

# Visualize embeddings using PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

print("\n📊 Reducing dimensions for visualization...")
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(test_embeddings)

pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings_scaled)

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

# Visualize embeddings (colored by word count since no price available)
plt.figure(figsize=(12, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                     c=df_test_clean["word_count"].values, cmap='viridis', 
                     alpha=0.6, s=50)
plt.colorbar(scatter, label='Word Count')
plt.title("SBERT Embeddings Visualization - Test Set (PCA)", fontsize=16, fontweight='bold')
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.savefig('test_embeddings_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 6. SAVE PROCESSED TEST DATA
# ============================================================================
print("\n💾 Saving processed test data...")

# Save cleaned data without embeddings (for reference)
df_test_clean_no_embed = df_test_clean.drop(columns=['embeddings'])
df_test_clean_no_embed.to_csv("test_processed_data.csv", index=False)
print("✅ Saved: test_processed_data.csv")

# Save embeddings as numpy array
np.save("test_sbert_embeddings.npy", test_embeddings)
print("✅ Saved: test_sbert_embeddings.npy")

# Save full data with embeddings (if needed)
df_test_clean.to_csv("test_with_embeddings.csv", index=False)
print("✅ Saved: test_with_embeddings.csv")

# ============================================================================
# 7. SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY - TEST SET")
print("=" * 80)

summary = {
    "Total Test Records": len(df_test),
    "Records After Cleaning": len(df_test_clean),
    "Average Word Count": f"{df_test_clean['word_count'].mean():.1f}",
    "Median Word Count": f"{df_test_clean['word_count'].median():.1f}",
    "Average Char Count": f"{df_test_clean['char_count'].mean():.1f}",
    "Embedding Dimensions": test_embeddings.shape[1],
    "Total Samples with Embeddings": len(df_test_clean)
}

print("\n📋 Summary Statistics:")
for key, value in summary.items():
    print(f"  {key}: {value}")

print("\n" + "=" * 80)
print("✅ TEST DATA PROCESSING COMPLETE!")
print("=" * 80)
print("\nGenerated files:")
print("  - test_processed_data.csv: Cleaned test data without embeddings")
print("  - test_sbert_embeddings.npy: SBERT embeddings as numpy array")
print("  - test_with_embeddings.csv: Full data with embeddings column")
print("  - test_eda_analysis.png: EDA visualizations")
print("  - test_embeddings_visualization.png: PCA visualization of embeddings")
