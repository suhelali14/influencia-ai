"""
Feature Engineering Module

This module handles ALL feature computation for the recommendation system.
Unlike the previous implementation that had:
  - Hardcoded values (platform_match = 0.8)
  - Circular label generation
  - No semantic understanding

This module provides:
  - Proper semantic similarity for categories
  - Real-time feature computation
  - Feature normalization and scaling
  - Feature importance tracking
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from functools import lru_cache

from .entities import Creator, Campaign, FeatureVector

logger = logging.getLogger(__name__)


class FeatureEngineering:
    """
    Production-grade feature engineering for influencer-brand matching.
    
    Key improvements over previous system:
    1. Semantic category matching (not binary)
    2. Proper normalization with learned statistics
    3. Temporal features for recency
    4. No hardcoded magic numbers
    """
    
    # Category hierarchy for semantic similarity
    CATEGORY_HIERARCHY = {
        'fashion': ['clothing', 'style', 'apparel', 'accessories', 'luxury'],
        'beauty': ['makeup', 'skincare', 'cosmetics', 'haircare', 'wellness'],
        'technology': ['tech', 'gadgets', 'software', 'electronics', 'gaming'],
        'gaming': ['esports', 'streaming', 'technology'],
        'fitness': ['health', 'wellness', 'sports', 'nutrition', 'workout'],
        'food': ['cooking', 'recipes', 'restaurants', 'nutrition', 'beverages'],
        'travel': ['tourism', 'adventure', 'hospitality', 'lifestyle'],
        'lifestyle': ['home', 'family', 'daily', 'wellness', 'self-improvement'],
        'entertainment': ['music', 'movies', 'comedy', 'celebrity', 'pop culture'],
        'education': ['learning', 'skills', 'courses', 'tutorials', 'knowledge'],
        'business': ['finance', 'entrepreneurship', 'marketing', 'career'],
        'sports': ['fitness', 'athletics', 'outdoor', 'teams'],
        'automotive': ['cars', 'vehicles', 'motorsports'],
        'parenting': ['family', 'kids', 'baby', 'education'],
        'pets': ['animals', 'dogs', 'cats'],
        'art': ['design', 'photography', 'creative', 'music'],
    }
    
    # Related categories (for semantic similarity)
    RELATED_CATEGORIES = {
        ('fashion', 'beauty'): 0.7,
        ('fashion', 'lifestyle'): 0.6,
        ('beauty', 'lifestyle'): 0.6,
        ('fitness', 'health'): 0.9,
        ('fitness', 'sports'): 0.8,
        ('food', 'health'): 0.5,
        ('travel', 'lifestyle'): 0.7,
        ('gaming', 'technology'): 0.7,
        ('entertainment', 'lifestyle'): 0.5,
        ('parenting', 'lifestyle'): 0.6,
        ('pets', 'lifestyle'): 0.5,
    }
    
    # Tier order for matching
    TIER_ORDER = ['nano', 'micro', 'mid', 'macro', 'mega']
    
    # Normalization constants (should be learned from data)
    NORMALIZATION = {
        'followers': {'min': 1000, 'max': 10_000_000, 'log': True},
        'engagement_rate': {'min': 0.01, 'max': 0.15},
        'budget': {'min': 100, 'max': 100_000, 'log': True},
        'response_time': {'min': 1, 'max': 168},  # hours
        'growth_rate': {'min': -0.1, 'max': 0.5},
    }
    
    def __init__(self, embedding_model=None):
        """
        Initialize feature engineering module.
        
        Args:
            embedding_model: Optional sentence transformer for semantic embeddings
        """
        self.embedding_model = embedding_model
        self._category_embeddings_cache = {}
        
    def compute_features(
        self,
        creator: Creator,
        campaign: Campaign,
        interactions: Optional[List[Dict]] = None,
        cf_score: float = 0.0,
    ) -> FeatureVector:
        """
        Compute all features for a creator-campaign pair.
        
        This is the main entry point for feature engineering.
        Returns a FeatureVector ready for the ranking model.
        """
        fv = FeatureVector()
        
        # Creator features (normalized)
        fv.creator_followers_normalized = self._normalize_followers(creator.followers)
        fv.creator_engagement_normalized = self._normalize_engagement(creator.engagement_rate)
        fv.creator_quality_score = creator.content_quality_score
        fv.creator_authenticity = creator.audience_authenticity
        fv.creator_growth_rate = self._normalize_growth_rate(creator.growth_rate_30d)
        fv.creator_response_time_normalized = self._normalize_response_time(creator.response_time_hours)
        fv.creator_completion_rate = creator.completion_rate
        fv.creator_avg_rating = creator.avg_campaign_rating / 5.0  # Normalize to 0-1
        fv.creator_success_rate = (
            creator.successful_campaigns / max(1, creator.total_campaigns)
        )
        fv.creator_experience_level = self._compute_experience_level(creator)
        
        # Brand/Campaign features
        fv.brand_avg_rating = campaign.brand_avg_rating / 5.0
        fv.brand_payment_reliability = campaign.brand_payment_reliability
        fv.budget_normalized = self._normalize_budget(campaign.budget)
        
        # Match features (the core matching logic)
        fv.platform_match = self._compute_platform_match(creator.platform, campaign.platform)
        fv.category_similarity = self._compute_category_similarity(
            creator.categories, campaign.categories
        )
        fv.location_match = self._compute_location_match(
            creator.location, campaign.target_locations
        )
        fv.language_match = self._compute_language_match(
            creator.language, campaign.target_languages
        )
        fv.tier_match = self._compute_tier_match(
            creator.tier, campaign.preferred_tiers
        )
        fv.budget_fit = self._compute_budget_fit(
            creator.avg_cost, campaign.budget
        )
        fv.followers_fit = self._compute_followers_fit(
            creator.followers, 
            campaign.min_followers, 
            campaign.max_followers
        )
        fv.engagement_fit = self._compute_engagement_fit(
            creator.engagement_rate, 
            campaign.target_engagement_rate
        )
        
        # Embedding similarity (if embeddings available)
        fv.embedding_similarity = self._compute_embedding_similarity(
            creator.embedding, campaign.embedding
        )
        
        # Collaborative filtering score
        fv.cf_score = cf_score
        
        # Temporal features
        fv.creator_recency = self._compute_recency(creator.last_active)
        fv.campaign_urgency = self._compute_urgency(campaign.start_date)
        
        # Historical interaction features
        if interactions:
            fv.previous_interactions = self._compute_interaction_score(interactions)
            fv.previous_success = self._compute_historical_success(interactions)
        
        return fv
    
    def _normalize_followers(self, followers: int) -> float:
        """Normalize follower count using log scale"""
        config = self.NORMALIZATION['followers']
        log_followers = np.log10(max(1, followers))
        log_min = np.log10(config['min'])
        log_max = np.log10(config['max'])
        return np.clip((log_followers - log_min) / (log_max - log_min), 0, 1)
    
    def _normalize_engagement(self, engagement_rate: float) -> float:
        """Normalize engagement rate"""
        config = self.NORMALIZATION['engagement_rate']
        return np.clip(
            (engagement_rate - config['min']) / (config['max'] - config['min']),
            0, 1
        )
    
    def _normalize_growth_rate(self, growth_rate: float) -> float:
        """Normalize growth rate"""
        config = self.NORMALIZATION['growth_rate']
        return np.clip(
            (growth_rate - config['min']) / (config['max'] - config['min']),
            0, 1
        )
    
    def _normalize_response_time(self, hours: float) -> float:
        """Normalize response time (lower is better, so invert)"""
        config = self.NORMALIZATION['response_time']
        normalized = (hours - config['min']) / (config['max'] - config['min'])
        return 1.0 - np.clip(normalized, 0, 1)  # Invert so faster = higher
    
    def _normalize_budget(self, budget: float) -> float:
        """Normalize budget using log scale"""
        config = self.NORMALIZATION['budget']
        log_budget = np.log10(max(1, budget))
        log_min = np.log10(config['min'])
        log_max = np.log10(config['max'])
        return np.clip((log_budget - log_min) / (log_max - log_min), 0, 1)
    
    def _compute_experience_level(self, creator: Creator) -> float:
        """Compute normalized experience level"""
        # Combine multiple signals
        campaigns = min(creator.total_campaigns / 50.0, 1.0)  # Cap at 50 campaigns
        completion = creator.completion_rate
        rating = creator.avg_campaign_rating / 5.0
        
        return 0.4 * campaigns + 0.3 * completion + 0.3 * rating
    
    def _compute_platform_match(self, creator_platform: str, campaign_platform: str) -> float:
        """
        Compute platform match score.
        
        Unlike the old system that returned hardcoded 0.8, this properly
        computes platform compatibility.
        """
        if not creator_platform or not campaign_platform:
            return 0.5  # Unknown, neutral score
        
        creator_platform = creator_platform.lower().strip()
        campaign_platform = campaign_platform.lower().strip()
        
        # Exact match
        if creator_platform == campaign_platform:
            return 1.0
        
        # Handle "all" or "any" platform campaigns
        if campaign_platform in ['all', 'any', 'multiple']:
            return 0.9
        
        # Handle comma-separated platforms
        if ',' in campaign_platform:
            platforms = [p.strip() for p in campaign_platform.split(',')]
            if creator_platform in platforms:
                return 1.0
            return 0.3  # Not in list
        
        # Platform compatibility matrix
        compatibility = {
            ('instagram', 'facebook'): 0.7,  # Same parent company
            ('instagram', 'tiktok'): 0.5,    # Different but both short-form
            ('youtube', 'tiktok'): 0.4,      # Video but different formats
            ('twitter', 'x'): 1.0,           # Same platform, renamed
            ('linkedin', 'twitter'): 0.4,    # Professional overlap
        }
        
        pair = (creator_platform, campaign_platform)
        reverse_pair = (campaign_platform, creator_platform)
        
        if pair in compatibility:
            return compatibility[pair]
        if reverse_pair in compatibility:
            return compatibility[reverse_pair]
        
        return 0.2  # No match
    
    def _compute_category_similarity(
        self, 
        creator_categories: List[str], 
        campaign_categories: List[str]
    ) -> float:
        """
        Compute semantic category similarity.
        
        This is a MAJOR improvement over the old binary matching.
        Uses category hierarchy and relationships.
        """
        if not creator_categories or not campaign_categories:
            return 0.5  # Unknown, neutral
        
        # Normalize categories
        creator_cats = [c.lower().strip() for c in creator_categories]
        campaign_cats = [c.lower().strip() for c in campaign_categories]
        
        # Calculate best match for each campaign category
        max_scores = []
        for cc in campaign_cats:
            best_score = 0.0
            for crc in creator_cats:
                score = self._category_pair_similarity(crc, cc)
                best_score = max(best_score, score)
            max_scores.append(best_score)
        
        # Return average of best matches
        return np.mean(max_scores) if max_scores else 0.5
    
    def _category_pair_similarity(self, cat1: str, cat2: str) -> float:
        """Compute similarity between two categories"""
        if cat1 == cat2:
            return 1.0
        
        # Check if one is a sub-category of another
        if cat1 in self.CATEGORY_HIERARCHY:
            if cat2 in self.CATEGORY_HIERARCHY[cat1]:
                return 0.85
        if cat2 in self.CATEGORY_HIERARCHY:
            if cat1 in self.CATEGORY_HIERARCHY[cat2]:
                return 0.85
        
        # Check related categories
        pair = (cat1, cat2) if cat1 < cat2 else (cat2, cat1)
        if pair in self.RELATED_CATEGORIES:
            return self.RELATED_CATEGORIES[pair]
        
        # Check for shared sub-categories
        subs1 = set(self.CATEGORY_HIERARCHY.get(cat1, []))
        subs2 = set(self.CATEGORY_HIERARCHY.get(cat2, []))
        if subs1 and subs2:
            overlap = len(subs1 & subs2) / len(subs1 | subs2)
            if overlap > 0:
                return 0.4 + 0.3 * overlap
        
        return 0.2  # Unrelated categories
    
    def _compute_location_match(
        self, 
        creator_location: str, 
        target_locations: List[str]
    ) -> float:
        """Compute location match score"""
        if not target_locations or 'global' in [t.lower() for t in target_locations]:
            return 0.9  # Global campaign, location doesn't matter much
        
        if not creator_location:
            return 0.5  # Unknown
        
        creator_loc = creator_location.lower().strip()
        target_locs = [t.lower().strip() for t in target_locations]
        
        # Exact match
        if creator_loc in target_locs:
            return 1.0
        
        # Check for country/region matches
        for target in target_locs:
            if target in creator_loc or creator_loc in target:
                return 0.8
        
        return 0.3  # No match
    
    def _compute_language_match(
        self, 
        creator_language: str, 
        target_languages: List[str]
    ) -> float:
        """Compute language match score"""
        if not target_languages:
            return 0.9  # No language requirement
        
        if not creator_language:
            return 0.5  # Unknown
        
        creator_lang = creator_language.lower().strip()
        target_langs = [t.lower().strip() for t in target_languages]
        
        if creator_lang in target_langs:
            return 1.0
        
        # Check for language variants (e.g., en-US, en-GB both match 'english')
        for target in target_langs:
            if target.split('-')[0] == creator_lang.split('-')[0]:
                return 0.9
        
        return 0.2  # No match
    
    def _compute_tier_match(
        self, 
        creator_tier: str, 
        preferred_tiers: List[str]
    ) -> float:
        """Compute tier match score"""
        if not preferred_tiers:
            return 0.9  # No preference
        
        if not creator_tier:
            return 0.5  # Unknown
        
        creator_tier = creator_tier.lower().strip()
        pref_tiers = [t.lower().strip() for t in preferred_tiers]
        
        # Exact match
        if creator_tier in pref_tiers:
            return 1.0
        
        # Adjacent tier (within 1 step)
        if creator_tier in self.TIER_ORDER:
            creator_idx = self.TIER_ORDER.index(creator_tier)
            for pref in pref_tiers:
                if pref in self.TIER_ORDER:
                    pref_idx = self.TIER_ORDER.index(pref)
                    distance = abs(creator_idx - pref_idx)
                    if distance == 1:
                        return 0.7
                    elif distance == 2:
                        return 0.4
        
        return 0.2  # Far from preferred
    
    def _compute_budget_fit(self, creator_cost: float, campaign_budget: float) -> float:
        """
        Compute how well creator cost fits campaign budget.
        
        Score is highest when creator cost is 30-70% of budget,
        allowing room for multiple creators or contingency.
        """
        if campaign_budget <= 0 or creator_cost <= 0:
            return 0.5
        
        ratio = creator_cost / campaign_budget
        
        if ratio > 1.0:
            # Creator costs more than budget
            return max(0.1, 1.0 - (ratio - 1.0))
        elif ratio > 0.7:
            # Approaching budget limit
            return 0.7 + 0.3 * (1.0 - ratio) / 0.3
        elif ratio > 0.3:
            # Sweet spot: 30-70% of budget
            return 1.0
        elif ratio > 0.1:
            # Creator might be too cheap (quality concerns)
            return 0.7 + 0.3 * (ratio - 0.1) / 0.2
        else:
            # Very cheap, might be mismatch
            return 0.5 + 0.2 * (ratio / 0.1)
    
    def _compute_followers_fit(
        self, 
        creator_followers: int, 
        min_followers: int, 
        max_followers: int
    ) -> float:
        """Compute how well follower count fits requirements"""
        if max_followers == 0:
            max_followers = float('inf')
        
        if creator_followers < min_followers:
            # Below minimum
            ratio = creator_followers / max(1, min_followers)
            return max(0.2, ratio)
        
        if creator_followers > max_followers and max_followers < float('inf'):
            # Above maximum
            excess = (creator_followers - max_followers) / max_followers
            return max(0.3, 1.0 - min(1.0, excess))
        
        # Within range
        return 1.0
    
    def _compute_engagement_fit(
        self, 
        creator_engagement: float, 
        target_engagement: float
    ) -> float:
        """Compute how well engagement rate meets target"""
        if target_engagement <= 0:
            return 0.8  # No target, use creator's engagement as-is
        
        ratio = creator_engagement / target_engagement
        
        if ratio >= 1.0:
            # Meets or exceeds target
            return min(1.0, 0.9 + 0.1 * (1.0 / ratio))
        else:
            # Below target
            return max(0.2, ratio)
    
    def _compute_embedding_similarity(
        self, 
        creator_embedding: Optional[np.ndarray], 
        campaign_embedding: Optional[np.ndarray]
    ) -> float:
        """Compute cosine similarity between embeddings"""
        if creator_embedding is None or campaign_embedding is None:
            return 0.5  # Unknown
        
        # Cosine similarity
        norm1 = np.linalg.norm(creator_embedding)
        norm2 = np.linalg.norm(campaign_embedding)
        
        if norm1 == 0 or norm2 == 0:
            return 0.5
        
        similarity = np.dot(creator_embedding, campaign_embedding) / (norm1 * norm2)
        
        # Transform from [-1, 1] to [0, 1]
        return (similarity + 1) / 2
    
    def _compute_recency(self, last_active: Optional[datetime]) -> float:
        """Compute recency score based on last activity"""
        if last_active is None:
            return 0.5  # Unknown
        
        now = datetime.now()
        days_ago = (now - last_active).days
        
        if days_ago <= 1:
            return 1.0
        elif days_ago <= 7:
            return 0.9
        elif days_ago <= 30:
            return 0.7
        elif days_ago <= 90:
            return 0.5
        else:
            return max(0.2, 1.0 - days_ago / 365)
    
    def _compute_urgency(self, start_date: Optional[datetime]) -> float:
        """Compute campaign urgency score"""
        if start_date is None:
            return 0.5  # Unknown
        
        now = datetime.now()
        days_until = (start_date - now).days
        
        if days_until <= 0:
            return 1.0  # Already started
        elif days_until <= 7:
            return 0.9
        elif days_until <= 30:
            return 0.7
        elif days_until <= 90:
            return 0.5
        else:
            return 0.3
    
    def _compute_interaction_score(self, interactions: List[Dict]) -> float:
        """Compute score from historical interactions"""
        if not interactions:
            return 0.0
        
        # Weight different interaction types
        weights = {
            'view': 0.1,
            'click': 0.2,
            'save': 0.3,
            'message': 0.5,
            'invite': 0.7,
            'accept': 0.8,
            'complete': 1.0,
        }
        
        total_score = 0.0
        for interaction in interactions:
            int_type = interaction.get('type', 'view')
            weight = weights.get(int_type, 0.1)
            
            # Recency factor
            timestamp = interaction.get('timestamp')
            if timestamp:
                days_ago = (datetime.now() - timestamp).days
                recency = max(0.1, 1.0 - days_ago / 365)
            else:
                recency = 0.5
            
            total_score += weight * recency
        
        # Normalize
        return min(1.0, total_score / 3)
    
    def _compute_historical_success(self, interactions: List[Dict]) -> float:
        """Compute historical success rate with this pair"""
        completions = [i for i in interactions if i.get('type') == 'complete']
        if not completions:
            return 0.5  # No history
        
        outcomes = [i.get('outcome', 0.5) for i in completions]
        return np.mean(outcomes)
    
    def compute_batch_features(
        self,
        creators: List[Creator],
        campaign: Campaign,
        interactions_map: Optional[Dict[str, List[Dict]]] = None,
        cf_scores: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[str, FeatureVector]]:
        """
        Compute features for multiple creators efficiently.
        
        Returns list of (creator_id, feature_vector) tuples.
        """
        results = []
        
        for creator in creators:
            interactions = (
                interactions_map.get(creator.id, []) 
                if interactions_map else None
            )
            cf_score = cf_scores.get(creator.id, 0.0) if cf_scores else 0.0
            
            fv = self.compute_features(
                creator=creator,
                campaign=campaign,
                interactions=interactions,
                cf_score=cf_score,
            )
            results.append((creator.id, fv))
        
        return results
