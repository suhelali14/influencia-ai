"""
Comprehensive Model Training Pipeline
Trains all ML models with India-specific data and global fallback
"""

import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 70)
print("INFLUENCIA ML MODEL TRAINING PIPELINE")
print("=" * 70)

# Check for India-specific data
india_data_exists = all([
    os.path.exists('data/raw/india_creators.csv'),
    os.path.exists('data/raw/india_campaigns.csv'),
    os.path.exists('data/raw/india_interactions.csv')
])

if india_data_exists:
    print("\n✓ Using India-specific dataset")
    creators_file = 'data/raw/india_creators.csv'
    campaigns_file = 'data/raw/india_campaigns.csv'
    interactions_file = 'data/raw/india_interactions.csv'
    model_prefix = 'india'
else:
    print("\n✓ Using global dataset")
    creators_file = 'data/raw/creators_full.csv'
    campaigns_file = 'data/raw/campaigns_full.csv'
    interactions_file = 'data/raw/interactions_full.csv'
    model_prefix = 'global'

print(f"\nLoading data from:")
print(f"  - {creators_file}")
print(f"  - {campaigns_file}")
print(f"  - {interactions_file}")

# Load data
creators = pd.read_csv(creators_file)
campaigns = pd.read_csv(campaigns_file)
interactions = pd.read_csv(interactions_file)

print(f"\n✓ Data loaded successfully!")
print(f"  Creators: {len(creators):,}")
print(f"  Campaigns: {len(campaigns):,}")
print(f"  Interactions: {len(interactions):,}")

# Prepare features
print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

from feature_engineering.feature_engineer import FeatureEngineer

print("\nExtracting features for training...")
feature_list = []
labels = []

for idx, interaction in interactions.head(10000).iterrows():  # Use subset for faster training
    try:
        creator = creators[creators['creator_id'] == interaction['creator_id']].iloc[0]
        campaign = campaigns[campaigns['campaign_id'] == interaction['campaign_id']].iloc[0]
        
        # Extract features
        campaign_feat = FeatureEngineer.campaign_features(campaign)
        interaction_feat = FeatureEngineer.interaction_features(creator, campaign)
        
        # Combine features
        combined = {**campaign_feat, **interaction_feat}
        
        # Add basic creator features (without historical data)
        combined['follower_count'] = float(creator.get('follower_count', 0))
        combined['engagement_rate'] = float(creator.get('engagement_rate', 0))
        combined['total_campaigns'] = float(creator.get('total_campaigns', 0))
        combined['success_rate'] = float(creator.get('success_rate', 0))
        combined['overall_rating'] = float(creator.get('overall_rating', 0))
        
        feature_list.append(combined)
        labels.append(interaction['match_score'])
        
        if (idx + 1) % 2000 == 0:
            print(f"  Processed {idx + 1:,} interactions...")
    except Exception as e:
        continue

features_df = pd.DataFrame(feature_list)
labels = np.array(labels)

print(f"\n✓ Feature extraction complete!")
print(f"  Training samples: {len(features_df):,}")
print(f"  Features per sample: {len(features_df.columns)}")
print(f"  Feature names: {list(features_df.columns)[:10]}...")

# Train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    features_df, labels, test_size=0.2, random_state=42
)

print(f"\nTrain set: {len(X_train):,} samples")
print(f"Test set: {len(X_test):,} samples")

# ============================================================
# 1. TRAIN XGBOOST MODEL
# ============================================================
print("\n" + "=" * 70)
print("TRAINING XGBOOST MODEL")
print("=" * 70)

try:
    import xgboost as xgb
    
    print("\nTraining XGBoost regressor...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    xgb_model.fit(X_train, y_train)
    
    # Evaluate
    train_score = xgb_model.score(X_train, y_train)
    test_score = xgb_model.score(X_test, y_test)
    
    print(f"✓ XGBoost training complete!")
    print(f"  Train R²: {train_score:.4f}")
    print(f"  Test R²: {test_score:.4f}")
    
    # Save model
    model_path = f'models/{model_prefix}_xgboost_model.json'
    os.makedirs('models', exist_ok=True)
    xgb_model.save_model(model_path)
    print(f"  Saved to: {model_path}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    xgboost_metrics = {
        'train_r2': float(train_score),
        'test_r2': float(test_score),
        'n_estimators': 100,
        'max_depth': 6
    }
    
except ImportError:
    print("⚠ XGBoost not installed. Installing...")
    os.system('python -m pip install xgboost')
    xgboost_metrics = {'status': 'installation_required'}
except Exception as e:
    print(f"✗ XGBoost training failed: {e}")
    xgboost_metrics = {'status': 'failed', 'error': str(e)}

# ============================================================
# 2. TRAIN SIMPLE NEURAL NETWORK (Lightweight alternative to Two-Tower)
# ============================================================
print("\n" + "=" * 70)
print("TRAINING NEURAL NETWORK")
print("=" * 70)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    
    print("\nPreparing neural network...")
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train.values)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Simple feedforward network
    class SimpleNN(nn.Module):
        def __init__(self, input_dim):
            super(SimpleNN, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.network(x)
    
    model = SimpleNN(input_dim=X_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    print("\nTraining neural network...")
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_tensor)
        test_pred = model(X_test_tensor)
        
        train_mse = criterion(train_pred, y_train_tensor).item()
        test_mse = criterion(test_pred, y_test_tensor).item()
    
    print(f"\n✓ Neural network training complete!")
    print(f"  Train MSE: {train_mse:.4f}")
    print(f"  Test MSE: {test_mse:.4f}")
    
    # Save model
    model_path = f'models/{model_prefix}_neural_network.pth'
    torch.save(model.state_dict(), model_path)
    print(f"  Saved to: {model_path}")
    
    nn_metrics = {
        'train_mse': float(train_mse),
        'test_mse': float(test_mse),
        'epochs': epochs,
        'architecture': '128-64-32-1'
    }
    
except Exception as e:
    print(f"✗ Neural network training failed: {e}")
    import traceback
    traceback.print_exc()
    nn_metrics = {'status': 'failed', 'error': str(e)}

# ============================================================
# SAVE TRAINING REPORT
# ============================================================
print("\n" + "=" * 70)
print("GENERATING TRAINING REPORT")
print("=" * 70)

training_report = {
    'timestamp': datetime.now().isoformat(),
    'dataset': model_prefix,
    'data_stats': {
        'creators': len(creators),
        'campaigns': len(campaigns),
        'interactions': len(interactions),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'features': len(features_df.columns)
    },
    'models': {
        'xgboost': xgboost_metrics,
        'neural_network': nn_metrics
    }
}

report_path = f'models/{model_prefix}_training_report.json'
with open(report_path, 'w') as f:
    json.dump(training_report, f, indent=2)

print(f"\n✓ Training report saved to: {report_path}")

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"\nModels trained with {model_prefix} dataset:")
print(f"  ✓ XGBoost (R²: {xgboost_metrics.get('test_r2', 'N/A')})")
print(f"  ✓ Neural Network (MSE: {nn_metrics.get('test_mse', 'N/A')})")
print(f"\nModels saved to: models/")
print("=" * 70)
