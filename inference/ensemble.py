"""
Ensemble Prediction System
Combines multiple models for robust creator-campaign matching:
- Two-Tower Deep Neural Network (40% weight)
- XGBoost Gradient Boosting (30% weight)  
- Neural Collaborative Filtering (20% weight)
- Semantic Matching (10% weight)
"""

import torch
import numpy as np
import joblib
import json
from typing import Dict, List, Tuple, Optional
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_models.two_tower import TwoTowerMatcher, create_default_config
from ml_models.ncf import NeuralCollaborativeFiltering
from ml_models.semantic_matcher import SemanticMatcher, HybridSemanticMatcher
from feature_engineering.feature_engineer import FeatureEngineer


class EnsemblePredictor:
    """
    Ensemble multiple models for robust predictions
    Combines deep learning, gradient boosting, and collaborative filtering
    """
    
    def __init__(self, model_paths: Optional[Dict[str, str]] = None, 
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize ensemble predictor
        
        Args:
            model_paths: Dictionary with paths to saved models
            weights: Model weights for ensemble (defaults: 40%, 30%, 20%, 10%)
        """
        self.model_paths = model_paths or {}
        self.weights = weights or {
            'two_tower': 0.40,
            'xgboost': 0.30,
            'ncf': 0.20,
            'semantic': 0.10
        }
        
        # Ensure weights sum to 1.0
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
        # Feature engineer
        self.feature_engineer = FeatureEngineer()
        
        # Load models
        self._load_models()
        
        print("Ensemble Predictor initialized")
        print(f"Model weights: {self.weights}")
    
    def _load_models(self):
        """Load all models"""
        # Two-Tower Model
        if 'two_tower' in self.model_paths:
            print("Loading Two-Tower model...")
            config = create_default_config()
            self.two_tower_model = TwoTowerMatcher(config)
            self.two_tower_model.load_state_dict(
                torch.load(self.model_paths['two_tower'], map_location='cpu')
            )
            self.two_tower_model.eval()
        else:
            self.two_tower_model = None
            print("Two-Tower model not loaded (path not provided)")
        
        # XGBoost Model
        if 'xgboost' in self.model_paths:
            print("Loading XGBoost model...")
            self.xgboost_model = joblib.load(self.model_paths['xgboost'])
        else:
            self.xgboost_model = None
            print("XGBoost model not loaded (path not provided)")
        
        # NCF Model
        if 'ncf' in self.model_paths:
            print("Loading NCF model...")
            # Load config to get num_creators and num_campaigns
            with open(self.model_paths['ncf'] + '.config.json', 'r') as f:
                ncf_config = json.load(f)
            
            self.ncf_model = NeuralCollaborativeFiltering(
                num_creators=ncf_config['num_creators'],
                num_campaigns=ncf_config['num_campaigns'],
                embedding_dim=ncf_config.get('embedding_dim', 64)
            )
            self.ncf_model.load_state_dict(
                torch.load(self.model_paths['ncf'], map_location='cpu')
            )
            self.ncf_model.eval()
        else:
            self.ncf_model = None
            print("NCF model not loaded (path not provided)")
        
        # Semantic Matcher
        print("Loading Semantic Matcher...")
        self.semantic_matcher = HybridSemanticMatcher()
        
        # Load feature names (for XGBoost)
        if 'feature_names' in self.model_paths:
            with open(self.model_paths['feature_names'], 'r') as f:
                self.feature_names = json.load(f)
        else:
            self.feature_names = None
    
    def predict(self, creator: Dict, campaign: Dict, 
                historical_data: Optional[pd.DataFrame] = None,
                brand: Optional[Dict] = None) -> Dict[str, float]:
        """
        Generate ensemble prediction
        
        Args:
            creator: Creator profile dictionary
            campaign: Campaign details dictionary
            historical_data: Historical performance data (optional)
            brand: Brand information (optional)
        
        Returns:
            Dictionary with predictions and confidence
        """
        scores = {}
        
        # 1. Two-Tower Prediction
        if self.two_tower_model is not None:
            two_tower_score = self._predict_two_tower(creator, campaign)
            scores['two_tower'] = two_tower_score
        else:
            scores['two_tower'] = 0.5  # Neutral if not available
        
        # 2. XGBoost Prediction
        if self.xgboost_model is not None and self.feature_names is not None:
            xgboost_score = self._predict_xgboost(creator, campaign, historical_data)
            scores['xgboost'] = xgboost_score
        else:
            scores['xgboost'] = 0.5
        
        # 3. NCF Prediction
        if self.ncf_model is not None:
            ncf_score = self._predict_ncf(creator, campaign)
            scores['ncf'] = ncf_score
        else:
            scores['ncf'] = 0.5
        
        # 4. Semantic Similarity
        semantic_result = self.semantic_matcher.hybrid_match_score(creator, campaign, brand)
        scores['semantic'] = semantic_result['overall_score']
        
        # Ensemble combination (weighted average)
        final_score = sum(
            self.weights[model] * scores[model]
            for model in self.weights.keys()
        )
        
        # Compute confidence (based on model agreement)
        confidence = self._compute_confidence(list(scores.values()))
        
        return {
            'match_score': final_score,
            'confidence': confidence,
            'model_breakdown': scores,
            'semantic_breakdown': semantic_result
        }
    
    def _predict_two_tower(self, creator: Dict, campaign: Dict) -> float:
        """Get prediction from Two-Tower model"""
        with torch.no_grad():
            # Prepare features (simplified - needs proper preprocessing)
            creator_numeric = torch.randn(1, 30)  # Placeholder
            creator_category = torch.tensor([0])  # Placeholder
            creator_platform = torch.tensor([0])
            creator_tier = torch.tensor([0])
            
            campaign_numeric = torch.randn(1, 15)
            campaign_category = torch.tensor([0])
            campaign_platform = torch.tensor([0])
            campaign_industry = torch.tensor([0])
            
            output = self.two_tower_model(
                creator_numeric, creator_category, creator_platform, creator_tier,
                campaign_numeric, campaign_category, campaign_platform, campaign_industry
            )
            
            return output['match_score'].item()
    
    def _predict_xgboost(self, creator: Dict, campaign: Dict, 
                        historical_data: Optional[pd.DataFrame]) -> float:
        """Get prediction from XGBoost model"""
        # Generate features
        creator_feats = self.feature_engineer.creator_features(creator, historical_data)
        campaign_feats = self.feature_engineer.campaign_features(campaign)
        interaction_feats = self.feature_engineer.interaction_features(creator, campaign)
        
        all_features = {**creator_feats, **campaign_feats, **interaction_feats}
        
        # Create feature vector in correct order
        feature_vector = [all_features.get(name, 0.0) for name in self.feature_names]
        
        # Predict probability
        prediction = self.xgboost_model.predict_proba([feature_vector])[0][1]
        
        return float(prediction)
    
    def _predict_ncf(self, creator: Dict, campaign: Dict) -> float:
        """Get prediction from NCF model"""
        with torch.no_grad():
            creator_id = torch.tensor([creator.get('creator_id', 0)])
            campaign_id = torch.tensor([campaign.get('campaign_id', 0)])
            
            prediction = self.ncf_model(creator_id, campaign_id)
            
            return prediction.item()
    
    def _compute_confidence(self, model_scores: List[float]) -> float:
        """
        Compute prediction confidence based on model agreement
        Low variance = high confidence
        
        Args:
            model_scores: List of scores from different models
        
        Returns:
            Confidence score (0-1)
        """
        if len(model_scores) < 2:
            return 1.0
        
        variance = np.var(model_scores)
        
        # Convert variance to confidence (0-1 scale)
        # High variance (0.25) -> low confidence (0.0)
        # Low variance (0.0) -> high confidence (1.0)
        confidence = 1 / (1 + 10 * variance)
        
        return float(confidence)
    
    def batch_predict(self, campaign: Dict, creators: List[Dict], 
                     top_k: int = 20, brand: Optional[Dict] = None) -> List[Tuple[int, float, Dict]]:
        """
        Efficiently rank many creators for a campaign
        
        Args:
            campaign: Campaign details
            creators: List of creator profiles
            top_k: Number of top matches to return
            brand: Brand information (optional)
        
        Returns:
            List of (creator_id, match_score, details) tuples
        """
        scores = []
        
        for creator in creators:
            prediction = self.predict(creator, campaign, brand=brand)
            scores.append((
                creator['creator_id'],
                prediction['match_score'],
                prediction
            ))
        
        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]
    
    def explain_prediction(self, creator: Dict, campaign: Dict, 
                          brand: Optional[Dict] = None) -> Dict:
        """
        Generate explanation for prediction
        
        Returns:
            Dictionary with explanation details
        """
        prediction = self.predict(creator, campaign, brand=brand)
        
        # Get model breakdown
        breakdown = prediction['model_breakdown']
        semantic = prediction['semantic_breakdown']
        
        # Identify strongest and weakest signals
        sorted_models = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)
        
        explanation = {
            'overall_score': prediction['match_score'],
            'confidence': prediction['confidence'],
            'strongest_signal': {
                'model': sorted_models[0][0],
                'score': sorted_models[0][1]
            },
            'weakest_signal': {
                'model': sorted_models[-1][0],
                'score': sorted_models[-1][1]
            },
            'model_agreement': self._assess_agreement(list(breakdown.values())),
            'semantic_factors': {
                'category_match': semantic.get('category_score', 0),
                'demographic_fit': semantic.get('demographic_score', 0),
                'content_relevance': semantic.get('semantic_score', 0)
            }
        }
        
        return explanation
    
    def _assess_agreement(self, scores: List[float]) -> str:
        """Assess model agreement level"""
        variance = np.var(scores)
        
        if variance < 0.01:
            return "Strong agreement"
        elif variance < 0.05:
            return "Moderate agreement"
        else:
            return "Mixed signals"


class LightweightEnsemble:
    """
    Lightweight ensemble for production use
    Only requires feature engineering and semantic matching
    No need for pre-trained models
    """
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.semantic_matcher = HybridSemanticMatcher()
    
    def predict(self, creator: Dict, campaign: Dict, 
                historical_data: Optional[pd.DataFrame] = None,
                brand: Optional[Dict] = None) -> Dict[str, float]:
        """
        Make prediction using only semantic matching and heuristics
        
        Args:
            creator: Creator profile
            campaign: Campaign details
            historical_data: Historical data (optional)
            brand: Brand info (optional)
        
        Returns:
            Prediction dictionary
        """
        # Semantic match
        semantic_result = self.semantic_matcher.hybrid_match_score(creator, campaign, brand)
        
        # Heuristic rules
        creator_feats = self.feature_engineer.creator_features(creator, historical_data)
        interaction_feats = self.feature_engineer.interaction_features(creator, campaign)
        
        # Requirement fit score
        requirement_score = (
            interaction_feats.get('interaction_meets_follower_requirement', 0) * 0.5 +
            interaction_feats.get('interaction_meets_engagement_requirement', 0) * 0.5
        )
        
        # Experience score
        experience_score = interaction_feats.get('interaction_experience_score', 0)
        
        # Quality score
        quality_score = interaction_feats.get('interaction_creator_success_rate', 0)
        
        # Combined score
        final_score = (
            0.4 * semantic_result['overall_score'] +
            0.3 * requirement_score +
            0.15 * experience_score +
            0.15 * quality_score
        )
        
        return {
            'match_score': final_score,
            'confidence': 0.75,  # Fixed confidence for lightweight
            'components': {
                'semantic': semantic_result['overall_score'],
                'requirements': requirement_score,
                'experience': experience_score,
                'quality': quality_score
            }
        }


if __name__ == '__main__':
    # Test lightweight ensemble (no pre-trained models needed)
    print("Testing Lightweight Ensemble...")
    
    ensemble = LightweightEnsemble()
    
    sample_creator = {
        'creator_id': 1,
        'bio': 'Fashion and lifestyle influencer',
        'categories': json.dumps(['Fashion', 'Lifestyle']),
        'platforms': json.dumps(['Instagram']),
        'followers': 50000,
        'engagement_rate': 0.045,
        'tier': 'micro',
        'total_campaigns': 20,
        'successful_campaigns': 17,
        'success_rate': 0.85,
        'overall_rating': 4.5,
        'total_earnings': 25000,
        'audience_age_18_24': 45,
        'audience_age_25_34': 35,
        'audience_female_pct': 70
    }
    
    sample_campaign = {
        'campaign_id': 1,
        'title': 'Summer Fashion Launch',
        'description': 'Promote new summer collection',
        'category': 'Fashion',
        'platform': 'Instagram',
        'industry': 'Fashion',
        'budget': 2000,
        'duration_days': 30,
        'deliverables': json.dumps(['Post', 'Story']),
        'min_followers': 30000,
        'min_engagement': 0.03,
        'target_age_group': '18-24',
        'target_gender': 'Female'
    }
    
    prediction = ensemble.predict(sample_creator, sample_campaign)
    
    print("\nPrediction Results:")
    print(f"Match Score: {prediction['match_score']:.4f}")
    print(f"Confidence: {prediction['confidence']:.4f}")
    print("\nComponent Scores:")
    for key, value in prediction['components'].items():
        print(f"  {key}: {value:.4f}")
