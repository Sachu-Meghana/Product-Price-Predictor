import os
import pandas as pd
import numpy as np
import re
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Correct SMAPE implementation
def smape(y_true, y_pred):
    return (1/len(y_true)) * np.sum(np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2)) * 100

# Extract only the most essential features
def extract_features(text):
    features = {}
    
    # Brand (first word of title)
    title = text.split('\n', 1)[0].replace("Item Name:", "").strip()
    features['brand'] = title.split()[0] if title.split() else ''
    
    # Item Pack Quantity
    ipq_match = re.search(r'Pack of\s*(\d+)|(\d+)\s*[Pp]ack|(\d+)\s*[Cc]ount', text)
    features['ipq'] = int(next((m for m in ipq_match.groups() if m), 1)) if ipq_match else 1
    
    # Weight
    weight_match = re.search(r'(\d+\.?\d*)\s*[Oo]unce|(\d+\.?\d*)\s*[Oo]z', text)
    features['weight'] = float(next((m for m in weight_match.groups() if m), 0)) if weight_match else 0
    
    # Quality indicators (binary)
    features['is_organic'] = 1 if 'organic' in text.lower() else 0
    features['is_premium'] = 1 if any(term in text.lower() for term in ['premium', 'gourmet']) else 0
    
    return features

# Main function
def main():
    print("Starting ultra-minimal pricing model...")
    
    # Set paths
    current_dir = os.path.abspath('')
    DATASET_FOLDER = os.path.join(os.path.dirname(current_dir), 'ML_CHALLANGE\dataset')
    MODEL_FOLDER = os.path.join(os.path.dirname(current_dir), 'models')
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    
    # Load datasets
    train = pd.read_csv(os.path.join(DATASET_FOLDER, 'train.csv'))
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    print(f"Train: {train.shape}, Test: {test.shape}")
    
    # Process training data
    print("Processing training data...")
    train_features = [extract_features(content) for content in train['catalog_content']]
    train_df = pd.DataFrame(train_features)
    
    # Create brand frequency encoding (more efficient than LabelEncoder)
    brand_counts = train_df['brand'].value_counts()
    train_df['brand_freq'] = train_df['brand'].map(brand_counts)
    
    # Add price target
    train_df['price'] = train['price']
    train_df['price_log'] = np.log1p(train_df['price'])
    
    # Feature engineering
    train_df['ipq_weight'] = train_df['ipq'] * train_df['weight'].fillna(0)
    
    # Prepare features and target
    feature_cols = ['brand_freq', 'ipq', 'weight', 'is_organic', 'is_premium', 'ipq_weight']
    X = train_df[feature_cols].fillna(0)
    y = train_df['price_log']
    
    # Train model
    print("Training model...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'verbose': -1
    }
    
    model = lgb.LGBMRegressor(**params, n_estimators=100)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(10)]
    )
    
    # Evaluate model
    preds_log = model.predict(X_val)
    preds = np.expm1(preds_log)
    y_val_orig = np.expm1(y_val)
    
    smape_val = smape(y_val_orig, preds)
    print(f"Validation SMAPE: {smape_val:.2f}%")
    
    # Process test data
    print("Processing test data...")
    test_features = [extract_features(content) for content in test['catalog_content']]
    test_df = pd.DataFrame(test_features)
    
    # Apply brand frequency encoding to test data
    test_df['brand_freq'] = test_df['brand'].map(brand_counts).fillna(0)
    
    # Feature engineering
    test_df['ipq_weight'] = test_df['ipq'] * test_df['weight'].fillna(0)
    
    # Make predictions
    print("Making predictions...")
    X_test = test_df[feature_cols].fillna(0)
    test_preds_log = model.predict(X_test)
    test_preds = np.expm1(test_preds_log)
    
    # Create submission file
    submission = pd.DataFrame({
        'sample_id': test['sample_id'],
        'price': test_preds
    })
    
    submission_path = os.path.join(DATASET_FOLDER, 'test_out.csv')
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    
    # Save model
    import joblib
    model_path = os.path.join(MODEL_FOLDER, 'lgbm_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save brand frequency mapping for future use
    brand_map_path = os.path.join(MODEL_FOLDER, 'brand_frequency_map.joblib')
    joblib.dump(brand_counts, brand_map_path)
    
    print(f"Final validation SMAPE: {smape_val:.2f}%")
    print("Done!")

if __name__ == "__main__":
    main()

