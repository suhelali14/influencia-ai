"""
Deep Learning Models for Advanced Analysis
- Neural Networks for complex pattern recognition
- Multi-output predictions: success probability, engagement, sentiment
- Advanced feature learning
"""
import os
import sys
import pandas as pd
import numpy as np
import json

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    from tensorflow.keras.models import Sequential, Model
    TF_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow not installed. Deep learning features will be limited.")
    TF_AVAILABLE = False

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

class DeepLearningAnalysis:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_training_data(self):
        """Load processed training data"""
        print("📥 Loading training data for DL...")
        
        if not os.path.exists('ai/data/training_data.csv'):
            print("❌ Training data not found. Run ETL pipeline first.")
            return None
        
        data = pd.read_csv('ai/data/training_data.csv')
        print(f"✅ Loaded {len(data)} training samples")
        return data
    
    def prepare_features(self, data: pd.DataFrame):
        """Prepare features for deep learning"""
        print("🔧 Preparing features for neural network...")
        
        # Select feature columns
        feature_cols = [
            'category_match', 'followers_match', 'engagement_match', 'platform_match',
            'experience_score', 'overall_rating', 'num_categories', 'num_languages',
            'estimated_followers', 'estimated_engagement_rate', 'campaign_budget',
            'campaign_duration_days', 'budget_fit', 'versatility_score', 'success_rate'
        ]
        
        available_cols = [col for col in feature_cols if col in data.columns]
        self.feature_names = available_cols
        
        X = data[available_cols].fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create multiple targets
        y_success = data['outcome'].values  # Binary: success or not
        
        # Match score (0-100)
        y_match = (
            data['category_match'] * 30 +
            data['followers_match'] * 15 +
            data['engagement_match'] * 10 +
            data['platform_match'] * 10 +
            data['experience_score'] * 4 +
            data['overall_rating'] * 3 +
            data['budget_fit'] * 10
        ).clip(0, 100).values / 100  # Normalize to 0-1
        
        # Engagement prediction (0-1)
        y_engagement = (data['estimated_engagement_rate'] * 10).clip(0, 1).values
        
        print(f"✅ Prepared {X_scaled.shape[1]} features, {len(X_scaled)} samples")
        return X_scaled, y_success, y_match, y_engagement
    
    def build_multi_output_model(self, input_dim):
        """Build neural network with multiple outputs"""
        if not TF_AVAILABLE:
            print("❌ TensorFlow not available")
            return None
        
        print("🏗️  Building multi-output neural network...")
        
        # Input layer
        inputs = keras.Input(shape=(input_dim,), name='input')
        
        # Shared layers
        x = layers.Dense(128, activation='relu', name='shared_1')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        x = layers.Dense(64, activation='relu', name='shared_2')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        
        x = layers.Dense(32, activation='relu', name='shared_3')(x)
        
        # Output 1: Success probability (binary classification)
        success_output = layers.Dense(16, activation='relu')(x)
        success_output = layers.Dense(1, activation='sigmoid', name='success_prob')(success_output)
        
        # Output 2: Match score (regression, 0-1)
        match_output = layers.Dense(16, activation='relu')(x)
        match_output = layers.Dense(1, activation='sigmoid', name='match_score')(match_output)
        
        # Output 3: Engagement prediction (regression, 0-1)
        engagement_output = layers.Dense(16, activation='relu')(x)
        engagement_output = layers.Dense(1, activation='sigmoid', name='engagement')(engagement_output)
        
        # Create model
        model = Model(
            inputs=inputs,
            outputs=[success_output, match_output, engagement_output],
            name='creator_campaign_analyzer'
        )
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'success_prob': 'binary_crossentropy',
                'match_score': 'mse',
                'engagement': 'mse'
            },
            loss_weights={
                'success_prob': 1.0,
                'match_score': 0.8,
                'engagement': 0.6
            },
            metrics={
                'success_prob': ['accuracy', keras.metrics.AUC(name='auc')],
                'match_score': ['mae'],
                'engagement': ['mae']
            }
        )
        
        print("✅ Model built successfully")
        print(f"\n📊 Model Summary:")
        model.summary()
        
        return model
    
    def train_model(self, X, y_success, y_match, y_engagement):
        """Train the deep learning model"""
        if not TF_AVAILABLE:
            print("❌ TensorFlow not available")
            return None
        
        print("\n🤖 Training deep learning model...\n")
        
        # Split data
        X_train, X_test, y_s_train, y_s_test, y_m_train, y_m_test, y_e_train, y_e_test = train_test_split(
            X, y_success, y_match, y_engagement, test_size=0.2, random_state=42
        )
        
        # Build model
        model = self.build_multi_output_model(X.shape[1])
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        # Train
        history = model.fit(
            X_train,
            {
                'success_prob': y_s_train,
                'match_score': y_m_train,
                'engagement': y_e_train
            },
            validation_data=(
                X_test,
                {
                    'success_prob': y_s_test,
                    'match_score': y_m_test,
                    'engagement': y_e_test
                }
            ),
            epochs=100,
            batch_size=32,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        # Evaluate
        print("\n📊 Model Evaluation:")
        results = model.evaluate(
            X_test,
            {
                'success_prob': y_s_test,
                'match_score': y_m_test,
                'engagement': y_e_test
            },
            verbose=0
        )
        
        print(f"  Test Loss: {results[0]:.4f}")
        print(f"  Success Accuracy: {results[4]:.4f}")
        print(f"  Success AUC: {results[5]:.4f}")
        print(f"  Match Score MAE: {results[7]:.4f}")
        print(f"  Engagement MAE: {results[9]:.4f}")
        
        self.model = model
        return model, history
    
    def save_model(self):
        """Save the trained model"""
        if not TF_AVAILABLE or not self.model:
            print("❌ No model to save")
            return
        
        print("\n💾 Saving deep learning model...")
        
        os.makedirs('ai/models', exist_ok=True)
        
        self.model.save('ai/models/dl_analysis_model.h5')
        joblib.dump(self.scaler, 'ai/models/dl_scaler.pkl')
        
        with open('ai/models/dl_feature_names.json', 'w') as f:
            json.dump(self.feature_names, f)
        
        print("✅ Model saved successfully")
    
    def load_model(self):
        """Load pre-trained model"""
        if not TF_AVAILABLE:
            print("❌ TensorFlow not available")
            return False
        
        print("📥 Loading deep learning model...")
        
        try:
            self.model = keras.models.load_model('ai/models/dl_analysis_model.h5')
            self.scaler = joblib.load('ai/models/dl_scaler.pkl')
            
            with open('ai/models/dl_feature_names.json', 'r') as f:
                self.feature_names = json.load(f)
            
            print("✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def predict(self, creator_campaign_features: dict):
        """Make predictions for a creator-campaign pair"""
        if not TF_AVAILABLE or not self.model:
            print("❌ Model not available")
            return None
        
        # Prepare features
        feature_values = [creator_campaign_features.get(f, 0) for f in self.feature_names]
        X = np.array(feature_values).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled, verbose=0)
        
        return {
            'success_probability': float(predictions[0][0][0]),
            'match_score': float(predictions[1][0][0] * 100),  # Scale back to 0-100
            'predicted_engagement': float(predictions[2][0][0] * 100)  # Scale to percentage
        }
    
    def train_pipeline(self):
        """Run complete training pipeline"""
        if not TF_AVAILABLE:
            print("❌ TensorFlow not installed. Install with: pip install tensorflow")
            return
        
        print("🚀 Starting Deep Learning Training Pipeline...\n")
        
        # Load data
        data = self.load_training_data()
        if data is None:
            return
        
        # Prepare features
        X, y_success, y_match, y_engagement = self.prepare_features(data)
        
        # Train model
        model, history = self.train_model(X, y_success, y_match, y_engagement)
        
        # Save model
        self.save_model()
        
        print("\n🎉 Deep Learning Pipeline completed successfully!")


if __name__ == "__main__":
    dl_model = DeepLearningAnalysis()
    dl_model.train_pipeline()
