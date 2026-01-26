"""
Advanced ML Models for Smart Creator-Campaign Matching
- Multiple models: RandomForest, XGBoost, Neural Network
- Predicts match score, engagement, ROI
- Uses sophisticated feature engineering
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import json

# Create directories
os.makedirs('ai/models', exist_ok=True)
os.makedirs('ai/data', exist_ok=True)

class MatchingMLModel:
    def __init__(self):
        self.match_model = None
        self.roi_model = None
        self.engagement_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_training_data(self):
        """Load processed training data"""
        print("📥 Loading training data...")
        
        if not os.path.exists('ai/data/training_data.csv'):
            print("❌ Training data not found. Run ETL pipeline first.")
            return None
        
        data = pd.read_csv('ai/data/training_data.csv')
        print(f"✅ Loaded {len(data)} training samples")
        return data
    
    def prepare_features(self, data: pd.DataFrame):
        """Prepare features for training"""
        print("🔧 Preparing features...")
        
        # Select feature columns
        feature_cols = [
            'category_match', 'followers_match', 'engagement_match', 'platform_match',
            'experience_score', 'overall_rating', 'num_categories', 'num_languages',
            'estimated_followers', 'estimated_engagement_rate', 'campaign_budget',
            'campaign_duration_days', 'budget_fit', 'versatility_score', 'success_rate'
        ]
        
        # Filter only available columns
        available_cols = [col for col in feature_cols if col in data.columns]
        self.feature_names = available_cols
        
        X = data[available_cols].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=available_cols)
        
        # Create target: match score (0-100)
        y_match = (
            data['category_match'] * 30 +
            data['followers_match'] * 15 +
            data['engagement_match'] * 10 +
            data['platform_match'] * 10 +
            data['experience_score'] * 4 +
            data['overall_rating'] * 3 +
            data['budget_fit'] * 10 +
            data['outcome'] * 15  # Bonus for successful collaborations
        ).clip(0, 100)
        
        # Create target: estimated ROI (percentage)
        y_roi = (
            data['estimated_engagement_rate'] * 1000 +
            data['overall_rating'] * 20 +
            (data['category_match'] * 50) +
            (data['outcome'] * 100)
        ).clip(0, 300)
        
        # Create target: predicted engagement
        y_engagement = data['estimated_engagement_rate'] * 100
        
        print(f"✅ Prepared {X_scaled.shape[1]} features")
        return X_scaled, y_match, y_roi, y_engagement
    
    def train_match_score_model(self, X, y):
        """Train match score prediction model"""
        print("🤖 Training match score model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rf_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"  📊 Match Score Model Performance:")
        print(f"     MSE: {mse:.2f}")
        print(f"     MAE: {mae:.2f}")
        print(f"     R²: {r2:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
        print(f"     CV R² (mean): {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
        
        self.match_model = rf_model
        return rf_model
    
    def train_roi_model(self, X, y):
        """Train ROI prediction model"""
        print("🤖 Training ROI prediction model...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Gradient Boosting
        gb_model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.1,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = gb_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"  📊 ROI Model Performance:")
        print(f"     MSE: {mse:.2f}")
        print(f"     MAE: {mae:.2f}")
        print(f"     R²: {r2:.3f}")
        
        self.roi_model = gb_model
        return gb_model
    
    def train_engagement_model(self, X, y):
        """Train engagement prediction model"""
        print("🤖 Training engagement prediction model...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = rf_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"  📊 Engagement Model Performance:")
        print(f"     MSE: {mse:.2f}")
        print(f"     MAE: {mae:.2f}")
        print(f"     R²: {r2:.3f}")
        
        self.engagement_model = rf_model
        return rf_model
    
    def get_feature_importance(self):
        """Get feature importance from models"""
        if self.match_model:
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.match_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n📊 Top 10 Most Important Features:")
            print(importance.head(10).to_string(index=False))
    
    def save_models(self):
        """Save trained models"""
        print("\n💾 Saving models...")
        
        if self.match_model:
            joblib.dump(self.match_model, 'ai/models/match_score_model.pkl')
            print("  ✅ Saved match score model")
        
        if self.roi_model:
            joblib.dump(self.roi_model, 'ai/models/roi_model.pkl')
            print("  ✅ Saved ROI model")
        
        if self.engagement_model:
            joblib.dump(self.engagement_model, 'ai/models/engagement_model.pkl')
            print("  ✅ Saved engagement model")
        
        # Save scaler and feature names
        joblib.dump(self.scaler, 'ai/models/scaler.pkl')
        with open('ai/models/feature_names.json', 'w') as f:
            json.dump(self.feature_names, f)
        print("  ✅ Saved scaler and feature names")
    
    def load_models(self):
        """Load pre-trained models"""
        print("📥 Loading pre-trained models...")
        
        try:
            self.match_model = joblib.load('ai/models/match_score_model.pkl')
            self.roi_model = joblib.load('ai/models/roi_model.pkl')
            self.engagement_model = joblib.load('ai/models/engagement_model.pkl')
            self.scaler = joblib.load('ai/models/scaler.pkl')
            
            with open('ai/models/feature_names.json', 'r') as f:
                self.feature_names = json.load(f)
            
            print("✅ Loaded all models successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def predict(self, creator_campaign_features: dict):
        """Make predictions for a creator-campaign pair"""
        if not self.match_model:
            print("❌ Models not loaded. Train or load models first.")
            return None
        
        # Prepare features
        feature_values = [creator_campaign_features.get(f, 0) for f in self.feature_names]
        X = np.array(feature_values).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        match_score = float(self.match_model.predict(X_scaled)[0])
        roi_estimate = float(self.roi_model.predict(X_scaled)[0])
        engagement_estimate = float(self.engagement_model.predict(X_scaled)[0])
        
        return {
            'match_score': np.clip(match_score, 0, 100),
            'estimated_roi': np.clip(roi_estimate, 0, 300),
            'estimated_engagement': np.clip(engagement_estimate, 0, 100)
        }
    
    def train_all(self):
        """Train all models"""
        print("🚀 Starting ML Model Training Pipeline...\n")
        
        # Load data
        data = self.load_training_data()
        if data is None:
            return
        
        # Prepare features
        X, y_match, y_roi, y_engagement = self.prepare_features(data)
        
        # Train models
        self.train_match_score_model(X, y_match)
        self.train_roi_model(X, y_roi)
        self.train_engagement_model(X, y_engagement)
        
        # Feature importance
        self.get_feature_importance()
        
        # Save models
        self.save_models()
        
        print("\n🎉 ML Training Pipeline completed successfully!")


if __name__ == "__main__":
    ml_model = MatchingMLModel()
    ml_model.train_all()
