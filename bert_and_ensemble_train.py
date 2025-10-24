"""
Advanced Training with BERT Embeddings and Ensemble Models
This script trains multiple models with BERT embeddings and advanced features
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import time
import warnings
warnings.filterwarnings('ignore')

def smape_score(y_true, y_pred):
    """Calculate SMAPE score"""
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred) / np.where(denominator == 0, 1, denominator)
    return 100.0 * np.mean(diff)

def create_price_bins(prices, n_bins=10):
    """Create stratification bins for price"""
    return pd.qcut(prices, q=n_bins, labels=False, duplicates='drop')

def load_and_prepare_features():
    """Load all features and embeddings"""
    print("Loading processed data...")
    train_df = pd.read_csv("train_processed.csv")
    test_df = pd.read_csv("test_processed.csv")
    
    print("Loading BERT embeddings...")
    train_cls = np.load("train_bert_cls_embeddings.npy")
    train_mean = np.load("train_bert_mean_embeddings.npy")
    train_max = np.load("train_bert_max_embeddings.npy")
    
    test_cls = np.load("test_bert_cls_embeddings.npy")
    test_mean = np.load("test_bert_mean_embeddings.npy")
    test_max = np.load("test_bert_max_embeddings.npy")
    
    # Combine embeddings
    print("Combining embeddings...")
    train_embeddings = np.hstack([train_cls, train_mean, train_max])
    test_embeddings = np.hstack([test_cls, test_mean, test_max])
    
    print(f"Combined embeddings shape: {train_embeddings.shape}")
    
    # Apply PCA to reduce dimensionality
    print("Applying PCA...")
    pca = PCA(n_components=512, random_state=42)
    train_embeddings_pca = pca.fit_transform(train_embeddings)
    test_embeddings_pca = pca.transform(test_embeddings)
    
    print(f"PCA embeddings shape: {train_embeddings_pca.shape}")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Get handcrafted features
    feature_cols = [col for col in train_df.columns 
                   if col.startswith(('text_', 'num_', 'brand_'))]
    
    train_features = train_df[feature_cols].fillna(0).values
    test_features = test_df[feature_cols].fillna(0).values
    
    print(f"Handcrafted features shape: {train_features.shape}")
    
    # Combine PCA embeddings with handcrafted features
    X_train = np.hstack([train_embeddings_pca, train_features])
    X_test = np.hstack([test_embeddings_pca, test_features])
    
    # Scale features
    print("Scaling features...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    y_train = train_df['price'].values
    
    print(f"\nFinal feature shapes:")
    print(f"X_train: {X_train_scaled.shape}")
    print(f"X_test: {X_test_scaled.shape}")
    print(f"y_train: {y_train.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, train_df, test_df

def get_models():
    """Define ensemble models with optimized hyperparameters"""
    models = {
        'XGBoost_L1': XGBRegressor(
            n_estimators=3000,
            learning_rate=0.02,
            max_depth=7,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=3.0,
            reg_lambda=5.0,
            gamma=0.3,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ),
        
        'XGBoost_L2': XGBRegressor(
            n_estimators=3000,
            learning_rate=0.025,
            max_depth=6,
            min_child_weight=4,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_alpha=2.5,
            reg_lambda=6.0,
            gamma=0.25,
            tree_method="hist",
            random_state=123,
            n_jobs=-1,
            verbosity=0
        ),
        
        'LightGBM_L1': LGBMRegressor(
            n_estimators=3000,
            learning_rate=0.02,
            num_leaves=96,
            max_depth=9,
            min_data_in_leaf=30,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            lambda_l1=2.0,
            lambda_l2=3.0,
            objective='regression_l1',
            metric='mae',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        
        'LightGBM_Huber': LGBMRegressor(
            n_estimators=3000,
            learning_rate=0.025,
            num_leaves=80,
            max_depth=8,
            min_data_in_leaf=35,
            feature_fraction=0.75,
            bagging_fraction=0.75,
            bagging_freq=5,
            lambda_l1=1.5,
            lambda_l2=2.5,
            objective='huber',
            alpha=0.9,
            metric='mae',
            random_state=123,
            n_jobs=-1,
            verbose=-1
        ),
        
        'CatBoost_RMSE': CatBoostRegressor(
            iterations=3000,
            depth=8,
            learning_rate=0.02,
            l2_leaf_reg=7.0,
            random_seed=42,
            loss_function='RMSE',
            bootstrap_type='Bernoulli',
            subsample=0.7,
            rsm=0.8,
            grow_policy='Lossguide',
            eval_metric='RMSE',
            verbose=0
        ),
        
        'CatBoost_MAE': CatBoostRegressor(
            iterations=3000,
            depth=9,
            learning_rate=0.015,
            l2_leaf_reg=8.0,
            random_seed=123,
            loss_function='MAE',
            bootstrap_type='Bernoulli',
            subsample=0.65,
            rsm=0.75,
            grow_policy='Lossguide',
            eval_metric='MAE',
            verbose=0
        ),
        
        'LightGBM_Tweedie': LGBMRegressor(
            n_estimators=3000,
            learning_rate=0.02,
            num_leaves=100,
            max_depth=8,
            min_data_in_leaf=30,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            lambda_l1=1.0,
            lambda_l2=2.0,
            objective='tweedie',
            tweedie_variance_power=1.5,
            metric='rmse',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    }
    
    return models

def train_with_kfold(X_train, y_train, X_test, n_splits=5):
    """Train models using K-Fold cross-validation"""
    
    # Use log transformation for target
    y_train_log = np.log1p(y_train)
    
    # Create stratified folds based on price bins
    price_bins = create_price_bins(y_train, n_bins=10)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    models = get_models()
    
    # Store predictions
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros((len(X_test), len(models)))
    
    fold_scores = []
    model_weights = {name: [] for name in models.keys()}
    
    start_time = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, price_bins), 1):
        print(f"\n{'='*70}")
        print(f"FOLD {fold}/{n_splits}")
        print('='*70)
        
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr_log, y_val_log = y_train_log[train_idx], y_train_log[val_idx]
        y_val_orig = y_train[val_idx]
        
        fold_val_preds = []
        fold_test_preds = []
        
        for model_idx, (name, model) in enumerate(models.items()):
            print(f"\nTraining {name}...", end=' ')
            t0 = time.time()
            
            try:
                if 'XGBoost' in name:
                    model.fit(
                        X_tr, y_tr_log,
                        eval_set=[(X_val, y_val_log)],
                        verbose=False
                    )
                elif 'LightGBM' in name:
                    model.fit(
                        X_tr, y_tr_log,
                        eval_set=[(X_val, y_val_log)],
                        callbacks=[
                            # Use callbacks parameter for early stopping
                        ]
                    )
                elif 'CatBoost' in name:
                    model.fit(
                        X_tr, y_tr_log,
                        eval_set=(X_val, y_val_log),
                        use_best_model=True,
                        verbose=False
                    )
            except Exception as e:
                print(f"Warning: {e}, using basic fit")
                model.fit(X_tr, y_tr_log)
            
            # Predictions
            val_pred_log = model.predict(X_val)
            val_pred_orig = np.expm1(val_pred_log)
            
            # Calculate SMAPE
            model_smape = smape_score(y_val_orig, val_pred_orig)
            model_weights[name].append(1.0 / (model_smape + 1e-6))
            
            fold_val_preds.append(val_pred_orig)
            
            # Test predictions
            test_pred_log = model.predict(X_test)
            test_pred_orig = np.expm1(test_pred_log)
            fold_test_preds.append(test_pred_orig)
            
            print(f"SMAPE: {model_smape:.4f} (time: {time.time()-t0:.1f}s)")
        
        # Weighted ensemble for this fold
        fold_val_preds = np.array(fold_val_preds)
        fold_test_preds = np.array(fold_test_preds)
        
        # Calculate weights based on inverse SMAPE
        current_weights = np.array([model_weights[name][-1] for name in models.keys()])
        current_weights = current_weights / current_weights.sum()
        
        # Weighted average
        weighted_val_pred = np.average(fold_val_preds, axis=0, weights=current_weights)
        oof_preds[val_idx] = weighted_val_pred
        
        # Fold score
        fold_smape = smape_score(y_val_orig, weighted_val_pred)
        fold_scores.append(fold_smape)
        
        print(f"\n{'='*70}")
        print(f"Fold {fold} Weighted Ensemble SMAPE: {fold_smape:.4f}")
        print(f"Model weights: {dict(zip(models.keys(), current_weights))}")
        print('='*70)
        
        # Accumulate test predictions
        for i in range(len(models)):
            test_preds[:, i] += fold_test_preds[i] / n_splits
    
    # Calculate final OOF score
    final_oof_smape = smape_score(y_train, oof_preds)
    
    # Calculate average weights across folds
    avg_weights = np.array([np.mean(model_weights[name]) for name in models.keys()])
    avg_weights = avg_weights / avg_weights.sum()
    
    # Final test predictions with averaged weights
    final_test_preds = np.average(test_preds, axis=1, weights=avg_weights)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Final OOF SMAPE: {final_oof_smape:.4f}")
    print(f"Fold SMAPE mean: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
    print(f"Fold scores: {[f'{s:.4f}' for s in fold_scores]}")
    print(f"\nFinal model weights:")
    for name, weight in zip(models.keys(), avg_weights):
        print(f"  {name}: {weight:.4f}")
    print(f"\nTotal training time: {total_time/60:.1f} minutes")
    print("="*70)
    
    return final_test_preds, final_oof_smape, fold_scores

def create_submission(test_df, predictions, filename="submission.csv"):
    """Create submission file"""
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })
    
    submission.to_csv(filename, index=False)
    print(f"\n✅ Submission saved: {filename}")
    print("\nFirst few predictions:")
    print(submission.head(10))
    print(f"\nPrediction statistics:")
    print(f"  Min: ${predictions.min():.2f}")
    print(f"  Max: ${predictions.max():.2f}")
    print(f"  Mean: ${predictions.mean():.2f}")
    print(f"  Median: ${np.median(predictions):.2f}")
    
    return submission

def main():
    print("="*70)
    print("ADVANCED BERT-BASED PRICE PREDICTION TRAINING")
    print("="*70)
    
    # Load and prepare features
    X_train, X_test, y_train, train_df, test_df = load_and_prepare_features()
    
    # Train models
    print("\n" + "="*70)
    print("STARTING K-FOLD TRAINING")
    print("="*70)
    
    test_predictions, oof_score, fold_scores = train_with_kfold(
        X_train, y_train, X_test, n_splits=5
    )
    
    # Create submission
    submission = create_submission(
        test_df, 
        test_predictions, 
        filename=f"submission_bert_smape_{oof_score:.4f}.csv"
    )
    
    # Save OOF predictions for stacking
    oof_df = pd.DataFrame({
        'sample_id': train_df['sample_id'],
        'actual_price': y_train,
        'predicted_price': test_predictions[:len(y_train)] if len(test_predictions) > len(y_train) else np.zeros(len(y_train))
    })
    oof_df.to_csv("oof_predictions.csv", index=False)
    print("\n✅ OOF predictions saved: oof_predictions.csv")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()