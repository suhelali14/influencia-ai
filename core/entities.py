"""
Industry-Grade Influencer-Brand Recommendation System

Architecture Overview:
=====================
This system implements a multi-stage recommendation pipeline inspired by 
industry practices from YouTube, Instagram, Alibaba, and Netflix.

Stage 1: CANDIDATE GENERATION (Fast, Coarse)
   - Approximate Nearest Neighbors (ANN) for embedding-based retrieval
   - Rule-based filtering (platform, budget, availability)
   - Collaborative filtering signals
   Output: ~100-500 candidates from millions

Stage 2: RANKING (Slower, Precise)  
   - Learning-to-rank model with rich features
   - Cross-features for interaction patterns
   - LLM-enhanced semantic scoring
   Output: Top 10-50 ranked candidates

Stage 3: RE-RANKING (Business Rules)
   - Diversity injection (category, tier, geography)
   - Freshness boost for new creators
   - Fairness constraints
   - Exploration vs exploitation (Thompson Sampling)
   Output: Final recommendations

Key Differences from Previous System:
=====================================
1. REAL outcome-based labels (not circular synthetic data)
2. Two-tower architecture for scalable candidate generation
3. Proper embedding storage with FAISS for fast retrieval
4. LLM integration for semantic understanding
5. Online feature computation for real-time signals
6. A/B testing framework for algorithm improvements
7. Monitoring and drift detection
8. Multi-objective optimization (engagement, ROI, fairness)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np


class RecommendationType(Enum):
    """Types of recommendations the system can generate"""
    CREATOR_FOR_CAMPAIGN = "creator_for_campaign"  # Find creators for a brand's campaign
    CAMPAIGN_FOR_CREATOR = "campaign_for_creator"  # Find campaigns for a creator
    SIMILAR_CREATORS = "similar_creators"          # Find similar creators
    TRENDING_CREATORS = "trending_creators"        # Currently hot creators


class MatchStage(Enum):
    """Stages in the recommendation pipeline"""
    CANDIDATE_GENERATION = "candidate_generation"
    RANKING = "ranking"
    RERANKING = "reranking"


@dataclass
class Creator:
    """Creator entity with all relevant features"""
    id: str
    name: str
    platform: str
    followers: int
    engagement_rate: float
    categories: List[str]
    location: str
    language: str
    avg_cost: float
    tier: str  # nano, micro, mid, macro, mega
    
    # Computed features (updated periodically)
    content_quality_score: float = 0.0
    audience_authenticity: float = 0.0
    growth_rate_30d: float = 0.0
    response_time_hours: float = 24.0
    completion_rate: float = 0.0
    avg_campaign_rating: float = 0.0
    total_campaigns: int = 0
    successful_campaigns: int = 0
    
    # Embedding (computed by encoder)
    embedding: Optional[np.ndarray] = None
    
    # Metadata
    last_active: Optional[datetime] = None
    created_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'platform': self.platform,
            'followers': self.followers,
            'engagement_rate': self.engagement_rate,
            'categories': self.categories,
            'location': self.location,
            'language': self.language,
            'avg_cost': self.avg_cost,
            'tier': self.tier,
            'content_quality_score': self.content_quality_score,
            'audience_authenticity': self.audience_authenticity,
            'growth_rate_30d': self.growth_rate_30d,
            'response_time_hours': self.response_time_hours,
            'completion_rate': self.completion_rate,
            'avg_campaign_rating': self.avg_campaign_rating,
            'total_campaigns': self.total_campaigns,
            'successful_campaigns': self.successful_campaigns,
        }


@dataclass 
class Campaign:
    """Campaign entity with all relevant features"""
    id: str
    brand_id: str
    brand_name: str
    title: str
    description: str
    platform: str
    categories: List[str]
    budget: float
    min_followers: int
    max_followers: int
    target_engagement_rate: float
    target_locations: List[str]
    target_languages: List[str]
    preferred_tiers: List[str]
    
    # Campaign requirements
    content_requirements: str = ""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Historical performance of brand
    brand_avg_rating: float = 0.0
    brand_payment_reliability: float = 1.0
    brand_total_campaigns: int = 0
    
    # Embedding (computed by encoder)
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'brand_id': self.brand_id,
            'brand_name': self.brand_name,
            'title': self.title,
            'description': self.description,
            'platform': self.platform,
            'categories': self.categories,
            'budget': self.budget,
            'min_followers': self.min_followers,
            'max_followers': self.max_followers,
            'target_engagement_rate': self.target_engagement_rate,
            'target_locations': self.target_locations,
            'target_languages': self.target_languages,
            'preferred_tiers': self.preferred_tiers,
        }


@dataclass
class MatchResult:
    """Result of matching a creator to a campaign"""
    creator_id: str
    campaign_id: str
    
    # Scores from each stage
    candidate_score: float = 0.0    # From candidate generation
    ranking_score: float = 0.0      # From ranking model
    final_score: float = 0.0        # After re-ranking
    
    # Score breakdown
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Predictions
    predicted_engagement: float = 0.0
    predicted_roi: float = 0.0
    predicted_success_probability: float = 0.0
    
    # Explanation
    match_reasons: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # LLM-generated explanation
    llm_explanation: str = ""
    
    # Metadata
    generated_at: datetime = field(default_factory=datetime.now)
    model_version: str = "2.0.0"
    
    def to_dict(self) -> Dict:
        return {
            'creator_id': self.creator_id,
            'campaign_id': self.campaign_id,
            'candidate_score': self.candidate_score,
            'ranking_score': self.ranking_score,
            'final_score': self.final_score,
            'score_breakdown': self.score_breakdown,
            'predicted_engagement': self.predicted_engagement,
            'predicted_roi': self.predicted_roi,
            'predicted_success_probability': self.predicted_success_probability,
            'match_reasons': self.match_reasons,
            'risks': self.risks,
            'recommendations': self.recommendations,
            'llm_explanation': self.llm_explanation,
            'model_version': self.model_version,
        }


@dataclass
class Interaction:
    """User interaction for collaborative filtering
    
    These are REAL signals that should drive the recommendation system:
    - Views: Brand viewed creator profile
    - Clicks: Brand clicked on creator in recommendations
    - Saves: Brand saved creator for later
    - Messages: Brand messaged creator
    - Invites: Brand invited creator to campaign
    - Accepts: Creator accepted invitation
    - Completes: Campaign completed successfully
    - Ratings: Post-campaign ratings (both directions)
    """
    id: str
    user_id: str  # Brand user ID or Creator user ID
    item_id: str  # Creator ID or Campaign ID
    interaction_type: str  # view, click, save, message, invite, accept, complete, rate
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)  # device, session, etc.
    outcome: Optional[float] = None  # For ratings, campaign success score, etc.


@dataclass
class FeatureVector:
    """Feature vector for ranking model"""
    # Creator features (normalized 0-1)
    creator_followers_normalized: float = 0.0
    creator_engagement_normalized: float = 0.0
    creator_quality_score: float = 0.0
    creator_authenticity: float = 0.0
    creator_growth_rate: float = 0.0
    creator_response_time_normalized: float = 0.0
    creator_completion_rate: float = 0.0
    creator_avg_rating: float = 0.0
    creator_success_rate: float = 0.0
    creator_experience_level: float = 0.0
    
    # Campaign/Brand features
    brand_avg_rating: float = 0.0
    brand_payment_reliability: float = 0.0
    budget_normalized: float = 0.0
    
    # Match features (the magic happens here)
    platform_match: float = 0.0
    category_similarity: float = 0.0  # Semantic similarity, not just exact match
    location_match: float = 0.0
    language_match: float = 0.0
    tier_match: float = 0.0
    budget_fit: float = 0.0  # How well creator cost fits budget
    followers_fit: float = 0.0  # How well followers match requirements
    engagement_fit: float = 0.0  # How well engagement matches target
    
    # Embedding similarity
    embedding_similarity: float = 0.0
    
    # Collaborative filtering signals
    cf_score: float = 0.0  # How similar brands liked this creator
    
    # Temporal features
    creator_recency: float = 0.0  # How recently active
    campaign_urgency: float = 0.0  # How soon campaign starts
    
    # Historical interaction features
    previous_interactions: float = 0.0
    previous_success: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input"""
        return np.array([
            self.creator_followers_normalized,
            self.creator_engagement_normalized,
            self.creator_quality_score,
            self.creator_authenticity,
            self.creator_growth_rate,
            self.creator_response_time_normalized,
            self.creator_completion_rate,
            self.creator_avg_rating,
            self.creator_success_rate,
            self.creator_experience_level,
            self.brand_avg_rating,
            self.brand_payment_reliability,
            self.budget_normalized,
            self.platform_match,
            self.category_similarity,
            self.location_match,
            self.language_match,
            self.tier_match,
            self.budget_fit,
            self.followers_fit,
            self.engagement_fit,
            self.embedding_similarity,
            self.cf_score,
            self.creator_recency,
            self.campaign_urgency,
            self.previous_interactions,
            self.previous_success,
        ], dtype=np.float32)
    
    @staticmethod
    def feature_names() -> List[str]:
        """Get feature names for model interpretability"""
        return [
            'creator_followers_normalized',
            'creator_engagement_normalized',
            'creator_quality_score',
            'creator_authenticity',
            'creator_growth_rate',
            'creator_response_time_normalized',
            'creator_completion_rate',
            'creator_avg_rating',
            'creator_success_rate',
            'creator_experience_level',
            'brand_avg_rating',
            'brand_payment_reliability',
            'budget_normalized',
            'platform_match',
            'category_similarity',
            'location_match',
            'language_match',
            'tier_match',
            'budget_fit',
            'followers_fit',
            'engagement_fit',
            'embedding_similarity',
            'cf_score',
            'creator_recency',
            'campaign_urgency',
            'previous_interactions',
            'previous_success',
        ]


# Configuration
@dataclass
class RecommendationConfig:
    """Configuration for the recommendation system"""
    # Candidate generation
    candidate_limit: int = 500
    ann_search_k: int = 100
    
    # Ranking
    ranking_limit: int = 50
    
    # Re-ranking
    final_limit: int = 20
    diversity_weight: float = 0.3
    exploration_rate: float = 0.1  # Thompson sampling exploration
    
    # Feature weights (for fallback rule-based scoring)
    category_weight: float = 0.25
    engagement_weight: float = 0.20
    platform_weight: float = 0.15
    budget_weight: float = 0.15
    quality_weight: float = 0.15
    history_weight: float = 0.10
    
    # Thresholds
    min_match_score: float = 30.0
    min_engagement_rate: float = 0.01
    
    # Model versions
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ranking_model_path: str = "models/ranking_model.joblib"
    
    # LLM settings
    use_llm_explanations: bool = True
    llm_model: str = "gemini-1.5-flash"
    
    # Caching
    cache_ttl_seconds: int = 3600
    embedding_cache_ttl: int = 86400  # 24 hours
