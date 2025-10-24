"""
Text Preprocessing and Feature Engineering
This script performs advanced text preprocessing and feature extraction
"""

import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def clean_text(text):
    """Clean and normalize text"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep important ones
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\(\)\[\]\{\}\/\&\%\$\@]', '', text)
    
    return text.strip()

def extract_numerical_features(text):
    """Extract numerical features from text"""
    features = {}
    
    # Extract numbers with units (e.g., "500ml", "2kg")
    numbers_with_units = re.findall(r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+)', text)
    
    # Common units for volume, weight, size
    volume_units = ['ml', 'l', 'litre', 'liter', 'oz', 'gallon']
    weight_units = ['g', 'kg', 'gram', 'kilogram', 'lb', 'oz', 'pound']
    size_units = ['cm', 'mm', 'inch', 'ft', 'meter', 'm']
    count_units = ['pack', 'piece', 'pcs', 'count', 'ct']
    
    features['has_volume'] = 0
    features['has_weight'] = 0
    features['has_size'] = 0
    features['has_count'] = 0
    features['max_number'] = 0
    features['min_number'] = 0
    features['avg_number'] = 0
    features['total_numbers'] = 0
    
    all_numbers = []
    
    for num, unit in numbers_with_units:
        num_val = float(num)
        all_numbers.append(num_val)
        unit_lower = unit.lower()
        
        if any(u in unit_lower for u in volume_units):
            features['has_volume'] = 1
        if any(u in unit_lower for u in weight_units):
            features['has_weight'] = 1
        if any(u in unit_lower for u in size_units):
            features['has_size'] = 1
        if any(u in unit_lower for u in count_units):
            features['has_count'] = 1
    
    # Extract standalone numbers
    standalone_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    all_numbers.extend([float(n) for n in standalone_numbers])
    
    if all_numbers:
        features['max_number'] = max(all_numbers)
        features['min_number'] = min(all_numbers)
        features['avg_number'] = np.mean(all_numbers)
        features['total_numbers'] = len(all_numbers)
    
    return features

def extract_text_statistics(text):
    """Extract comprehensive text statistics"""
    if pd.isna(text) or text == "":
        return {
            'char_count': 0,
            'word_count': 0,
            'avg_word_length': 0,
            'sentence_count': 0,
            'uppercase_ratio': 0,
            'digit_ratio': 0,
            'special_char_ratio': 0,
            'unique_word_ratio': 0,
            'title_word_count': 0,
            'exclamation_count': 0,
            'question_count': 0,
            'comma_count': 0,
            'dot_count': 0,
            'bracket_count': 0,
            'percent_count': 0,
            'dollar_count': 0,
            'ampersand_count': 0
        }
    
    text = str(text)
    words = text.split()
    
    stats = {
        'char_count': len(text),
        'word_count': len(words),
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        'sentence_count': len(re.findall(r'[.!?]+', text)),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
        'digit_ratio': sum(1 for c in text if c.isdigit()) / len(text) if len(text) > 0 else 0,
        'special_char_ratio': sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text) if len(text) > 0 else 0,
        'unique_word_ratio': len(set(words)) / len(words) if words else 0,
        'title_word_count': sum(1 for w in words if w and w[0].isupper()),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'comma_count': text.count(','),
        'dot_count': text.count('.'),
        'bracket_count': text.count('(') + text.count(')') + text.count('[') + text.count(']'),
        'percent_count': text.count('%'),
        'dollar_count': text.count('$'),
        'ampersand_count': text.count('&')
    }
    
    return stats

def extract_brand_and_category_features(text):
    """Extract brand and category related features"""
    text_lower = text.lower()
    
    features = {}
    
    # Common brand indicators
    brand_keywords = ['brand', 'trademark', 'tm', '®', '©']
    features['has_brand_indicator'] = int(any(kw in text_lower for kw in brand_keywords))
    
    # Category keywords
    electronics = ['electronic', 'digital', 'phone', 'laptop', 'tablet', 'camera', 'speaker', 'headphone']
    clothing = ['shirt', 'pant', 'dress', 'jacket', 'shoe', 'clothing', 'apparel', 'wear']
    food = ['food', 'snack', 'beverage', 'drink', 'edible', 'nutrition', 'vitamin']
    beauty = ['beauty', 'cosmetic', 'skincare', 'makeup', 'cream', 'lotion', 'serum']
    home = ['home', 'kitchen', 'furniture', 'decor', 'appliance']
    
    features['is_electronics'] = int(any(kw in text_lower for kw in electronics))
    features['is_clothing'] = int(any(kw in text_lower for kw in clothing))
    features['is_food'] = int(any(kw in text_lower for kw in food))
    features['is_beauty'] = int(any(kw in text_lower for kw in beauty))
    features['is_home'] = int(any(kw in text_lower for kw in home))
    
    # Quality indicators
    quality_keywords = ['premium', 'luxury', 'deluxe', 'professional', 'pro', 'ultra', 'super', 'best', 'top']
    features['quality_indicator_count'] = sum(1 for kw in quality_keywords if kw in text_lower)
    
    # Size indicators
    size_keywords = ['large', 'small', 'medium', 'xl', 'xxl', 'mini', 'jumbo', 'giant']
    features['has_size_indicator'] = int(any(kw in text_lower for kw in size_keywords))
    
    return features

def preprocess_dataset(df, is_train=True):
    """Main preprocessing function"""
    print("Starting preprocessing...")
    
    # Clean text
    print("Cleaning text...")
    df['catalog_clean'] = df['catalog_content'].apply(clean_text)
    
    # Extract text statistics
    print("Extracting text statistics...")
    text_stats = []
    for text in tqdm(df['catalog_clean'], desc="Text stats"):
        text_stats.append(extract_text_statistics(text))
    
    text_stats_df = pd.DataFrame(text_stats)
    text_stats_df.columns = ['text_' + col for col in text_stats_df.columns]
    
    # Extract numerical features
    print("Extracting numerical features...")
    num_features = []
    for text in tqdm(df['catalog_clean'], desc="Numerical features"):
        num_features.append(extract_numerical_features(text))
    
    num_features_df = pd.DataFrame(num_features)
    num_features_df.columns = ['num_' + col for col in num_features_df.columns]
    
    # Extract brand and category features
    print("Extracting brand/category features...")
    brand_features = []
    for text in tqdm(df['catalog_clean'], desc="Brand/category features"):
        brand_features.append(extract_brand_and_category_features(text))
    
    brand_features_df = pd.DataFrame(brand_features)
    brand_features_df.columns = ['brand_' + col for col in brand_features_df.columns]
    
    # Combine all features
    df_processed = pd.concat([
        df[['sample_id', 'catalog_clean']],
        text_stats_df,
        num_features_df,
        brand_features_df
    ], axis=1)
    
    if is_train:
        df_processed['price'] = df['price']
    
    print(f"Preprocessing complete. Shape: {df_processed.shape}")
    return df_processed

if __name__ == "__main__":
    # Load data
    print("Loading datasets...")
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    
    # Preprocess
    train_processed = preprocess_dataset(train_df, is_train=True)
    test_processed = preprocess_dataset(test_df, is_train=False)
    
    # Save processed data
    train_processed.to_csv("train_processed.csv", index=False)
    test_processed.to_csv("test_processed.csv", index=False)
    
    print("\n✅ Preprocessing complete!")
    print(f"Train shape: {train_processed.shape}")
    print(f"Test shape: {test_processed.shape}")
    print("\nFiles saved:")
    print("- train_processed.csv")
    print("- test_processed.csv")