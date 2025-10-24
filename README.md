<div align="center">
🛒 Smart Product Pricing Challenge
Multimodal Regression for E-commerce Price Prediction








Predicting optimal e-commerce product prices using text, vision, and multimodal fusion models.

</div>
🎯 Project Overview

This project presents a multimodal machine learning pipeline for predicting optimal product prices from product descriptions, specifications, and (optionally) image data.

The solution integrates text embeddings (E5, BERT, SBERT), vision-language models (VILT, FLAVA, CLIP), and ensemble regressors to minimize SMAPE (Symmetric Mean Absolute Percentage Error) on the test set.

<div align="center">
🏆 Best Performance: 56.2 SMAPE (E5 Ensemble Pipeline)
💡 CLIP Multimodal Pipeline: 60 SMAPE on Test Set
</div>
📊 Challenge Details
Aspect	Details
🎯 Objective	Predict product prices using multimodal product data
📈 Dataset	~75K training samples, 75K test samples
📏 Metric	Symmetric Mean Absolute Percentage Error (SMAPE)
🚫 Constraints	No external pricing data, model ≤ 8B params
🧩 Modalities Used	Text (title, description, IPQ), optional image embeddings
🏆 Best Result	56.2 SMAPE (E5 Ensemble), 60 SMAPE (CLIP Pipeline)
🏆 Model Performance Comparison
<div align="center">
Model / Pipeline	Type	SMAPE Score	Notes
E5 Ensemble	Text + Statistical Features	56.2	Best performing model
VILT Pipeline	Vision-Language Transformer	57.0	Multimodal fine-tuning
BERT Large	Text-only	59.3	Standard transformer
CLIP Pipeline	Image + Text Multimodal	60.0	Performed well on visual data
SBERT Baseline	Sentence-BERT	62.0	Text baseline
</div>
📈 Performance Progression
SBERT (62.0) → BERT Large (59.3) → CLIP (60.0) → VILT (57.0) → E5 Ensemble (56.2)

🧠 Technical Approach
1️⃣ Text Processing & Feature Engineering

Cleaned and tokenized product text fields (title, description, specs)

Extracted linguistic and statistical features (character count, digit ratio, etc.)

Generated transformer-based embeddings (E5, BERT, SBERT)

2️⃣ Vision + Multimodal Integration

CLIP Pipeline (clip_complete_pipeline.py): Extracted image-text representations via OpenAI CLIP, achieving 60 SMAPE

FLAVA Pipeline (flava_training.py, flava_test.py): Used Facebook FLAVA for unified vision-language embeddings

VILT Pipeline (vilt_pipeline.ipynb): Applied Vision-Language Transformer for early fusion

Normalized, concatenated multimodal features with tabular and text-based engineered features

3️⃣ Ensemble Regression Models

Combined the strengths of multiple gradient boosting models:

XGBoost

LightGBM

CatBoost

Tweedie LGBM

📁 Repository Structure
smart-product-pricing-multimodal-regression/
├── 📄 README.md                         # Project documentation (you’re here)
├── 📜 LICENSE                           # MIT License
├── 📓 E5_pipeline.ipynb                 # E5 embeddings + ensemble (Best: 56.2 SMAPE)
├── 📓 vilt_pipeline.ipynb               # Vision-Language Transformer (57.0 SMAPE)
├── 📓 quantile_regression_and_feature_eng.ipynb  # Text feature analysis and regression
├── 📓 ensemble_training_nb.ipynb        # K-Fold ensembling workflow
├── 🧠 bert_and_ensemble_train.py        # BERT + ensemble pipeline
├── 🧠 clip_complete_pipeline.py         # CLIP multimodal pipeline (60 SMAPE)
├── 🧠 flava_preprocessing.py            # Data preprocessor for FLAVA model
├── 🧠 flava_training.py / flava_test.py # FLAVA multimodal training & evaluation
├── 🧠 sbert_analysis.py / test_sbert_analysis.py  # SBERT experiments
├── 🧠 text_preprocess_feature_eng.py    # Text preprocessing + feature engineering
└── 📁 data/ (not included)              # Expected train.csv / test.csv

⚙️ Setup & Execution
🧩 Installation
pip install -r requirements.txt
# or install major dependencies manually:
pip install transformers sentence-transformers catboost lightgbm xgboost pandas numpy scikit-learn torch torchvision

🚀 Usage
Step	Command / Notebook	Description
1️⃣	text_preprocess_feature_eng.py	Generate textual statistical features
2️⃣	bert_and_ensemble_train.py or E5_pipeline.ipynb	Run text-based training
3️⃣	clip_complete_pipeline.py	Run multimodal CLIP pipeline
4️⃣	flava_training.py → flava_test.py	Train/test FLAVA model
5️⃣	ensemble_training_nb.ipynb	Combine predictions for ensemble results
📊 Results Summary
<div align="center">
Pipeline	Modality	SMAPE (↓)	Remarks
E5 Ensemble	Text + Engineered	56.2	Best
VILT	Vision-Language	57.0	Close second
CLIP	Image + Text	60.0	Strong multimodal baseline
BERT Large	Text	59.3	Robust text encoder
SBERT	Text	62.0	Baseline
</div>
🔍 Key Insights
Finding	Impact	Explanation
🧠 Textual signals dominate	High	Product titles/descriptions carry most price cues
🖼️ CLIP generalizes well visually	Medium	60 SMAPE without fine-tuned image labels
🧩 Multimodal fusion boosts robustness	High	Fusion improved over single-modality baselines
🎯 E5 embeddings outperform BERT/SBERT	High	Strong semantic capture of pricing semantics
🔁 Ensembling reduces error variance	Medium	K-Fold + model blending improved generalization
🔮 Future Work

Advanced Multimodal Fusion (Late fusion / cross-attention)

Hyperparameter Optimization using Optuna or Ray Tune

End-to-End Vision-Language Regression

Synthetic Data Augmentation for rare product categories

Deployment via FastAPI + Streamlit dashboard

🏁 Conclusion

The project successfully demonstrates that:

Text-based embeddings (E5) provide the strongest standalone performance (56.2 SMAPE)

CLIP offers competitive multimodal results (60 SMAPE) when integrating visual signals

The combination of ensemble methods and multimodal fusion leads to robust generalization across diverse product types

<div align="center">
🚀 "Smart Product Pricing – where language meets vision for value prediction."

Best Result: 56.2 SMAPE (E5 Ensemble)
CLIP Multimodal Pipeline: 60 SMAPE (Test Set)

</div>
