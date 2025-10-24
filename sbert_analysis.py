import pandas as pd
import numpy as np
import re, string
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# For SBERT embeddings
from sentence_transformers import SentenceTransformer

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("=" * 80)
print("LOADING DATASET")
print("=" * 80)

train_path = "train.csv"
df = pd.read_csv(train_path)

print(f"✅ Dataset Loaded Successfully!")
print(f"Shape: {df.shape}")
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
    text = re.sub(r"\d+", " ", text)  # Remove numbers (optional)
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
# 3. EXPLORATORY DATA ANALYSIS (EDA)
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
plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlation analysis
print("\n🔗 Correlation Matrix:")
correlation = df[["price", "word_count", "char_count"]].corr()
print(correlation)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title("Feature Correlation Heatmap", fontsize=16, fontweight='bold')
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 4. OUTLIER DETECTION & HANDLING
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

def detect_outliers_zscore(data, column, threshold=3):
    """Detect outliers using Z-score method"""
    z_scores = np.abs(stats.zscore(data[column]))
    outliers = data[z_scores > threshold]
    return outliers

# Detect outliers in price
print("\n🔍 Price Outliers (IQR Method):")
price_outliers_iqr, price_lower, price_upper = detect_outliers_iqr(df, "price")
print(f"Number of outliers: {len(price_outliers_iqr)}")
print(f"Lower bound: ${price_lower:.2f}")
print(f"Upper bound: ${price_upper:.2f}")
print(f"Percentage: {len(price_outliers_iqr)/len(df)*100:.2f}%")

print("\n🔍 Price Outliers (Z-score Method):")
price_outliers_zscore = detect_outliers_zscore(df, "price")
print(f"Number of outliers: {len(price_outliers_zscore)}")
print(f"Percentage: {len(price_outliers_zscore)/len(df)*100:.2f}%")

# Detect outliers in word count
print("\n🔍 Word Count Outliers (IQR Method):")
word_outliers_iqr, word_lower, word_upper = detect_outliers_iqr(df, "word_count")
print(f"Number of outliers: {len(word_outliers_iqr)}")
print(f"Lower bound: {word_lower:.2f}")
print(f"Upper bound: {word_upper:.2f}")
print(f"Percentage: {len(word_outliers_iqr)/len(df)*100:.2f}%")

# Visualize outliers
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Before handling outliers
axes[0].boxplot([df["price"], df["word_count"]], labels=['Price', 'Word Count'])
axes[0].set_title("Before Outlier Handling", fontsize=14, fontweight='bold')
axes[0].set_ylabel("Value")

# Handle outliers - Multiple strategies
# Strategy 1: Capping (Winsorization)
df_capped = df.copy()
df_capped["price"] = df_capped["price"].clip(lower=price_lower, upper=price_upper)
df_capped["word_count"] = df_capped["word_count"].clip(lower=word_lower, upper=word_upper)

# Strategy 2: Remove extreme outliers (optional)
df_filtered = df[
    (df["price"] >= price_lower) & (df["price"] <= price_upper) &
    (df["word_count"] >= word_lower) & (df["word_count"] <= word_upper)
].copy()

print(f"\n📊 Dataset shapes:")
print(f"Original: {df.shape}")
print(f"After capping: {df_capped.shape}")
print(f"After filtering: {df_filtered.shape}")

# After handling outliers (using capped version)
axes[1].boxplot([df_capped["price"], df_capped["word_count"]], 
                labels=['Price', 'Word Count'])
axes[1].set_title("After Outlier Handling (Capping)", fontsize=14, fontweight='bold')
axes[1].set_ylabel("Value")

plt.tight_layout()
plt.savefig('outlier_handling.png', dpi=300, bbox_inches='tight')
plt.show()

# Use the capped version for further analysis
df_clean = df_capped.copy()

# ============================================================================
# 5. GENERATE SBERT EMBEDDINGS
# ============================================================================
print("\n" + "=" * 80)
print("GENERATING SBERT EMBEDDINGS")
print("=" * 80)

# Load SBERT model
print("\n⏳ Loading SBERT model (this may take a moment)...")
model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast and efficient model
print("✅ Model loaded successfully!")

# Generate embeddings for a sample (or full dataset)
# For large datasets, you might want to process in batches
SAMPLE_SIZE = 75000  # Adjust based on your needs and memory
sample_df = df_clean.sample(n=min(SAMPLE_SIZE, len(df_clean)), random_state=42)

print(f"\n🔄 Generating embeddings for {len(sample_df)} samples...")
embeddings = model.encode(
    sample_df["clean_text"].tolist(),
    show_progress_bar=True,
    batch_size=100
)

print(f"✅ Embeddings generated!")
print(f"Embedding shape: {embeddings.shape}")
print(f"Embedding dimensions: {embeddings.shape[1]}")

# Add embeddings to dataframe
sample_df = sample_df.copy()
sample_df["embeddings"] = list(embeddings)

# Visualize embeddings using PCA
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
                     c=sample_df["price"].values, cmap='viridis', 
                     alpha=0.6, s=50)
plt.colorbar(scatter, label='Price ($)')
plt.title("SBERT Embeddings Visualization (PCA)", fontsize=16, fontweight='bold')
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
plt.savefig('embeddings_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# Save processed data
print("\n💾 Saving processed data...")
df_clean.to_csv("processed_data.csv", index=False)
print("✅ Saved: processed_data.csv")

# Save embeddings (for the sample)
np.save("sbert_embeddings.npy", embeddings)
sample_df.to_csv("sample_with_embeddings.csv", index=False)
print("✅ Saved: sbert_embeddings.npy")
print("✅ Saved: sample_with_embeddings.csv")

# ============================================================================
# 6. SUMMARY STATISTICS
# ============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

summary = {
    "Total Records": len(df),
    "Records After Cleaning": len(df_clean),
    "Average Price": f"${df_clean['price'].mean():.2f}",
    "Median Price": f"${df_clean['price'].median():.2f}",
    "Average Word Count": f"{df_clean['word_count'].mean():.1f}",
    "Embedding Dimensions": embeddings.shape[1],
    "Sample Size for Embeddings": len(sample_df)
}

print("\n📋 Summary Statistics:")
for key, value in summary.items():
    print(f"  {key}: {value}")

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE!")
print("=" * 80)
