1️. Project Overview
Fine-tuning DistilBERT for sentiment analysis on IMDb dataset
Binary classification (positive or negative reviews)
Uses Hugging Face Transformers and PyTorch
Experiment tracking with Comet ML
2️ Installation & Setup
Clone the repository
Install dependencies (transformers, datasets, torch, comet_ml)
3️ Dataset & Preprocessing
Uses IMDb dataset from Hugging Face
Tokenization with distilbert-base-uncased
Splitting into train & test sets
4️ Model Training
AutoModelForSequenceClassification with Trainer API
Evaluation metrics: Accuracy, Precision, Recall, F1-score
5️ Running Inference
Load the trained model
Predict sentiment of new reviews
6️ Model Performance
Table with accuracy, precision, recall, F1-score
7️ Comet ML Integration
Logs metrics, confusion matrix, and experiment details
