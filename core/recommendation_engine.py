"""
Recommendation Engine - Main Orchestrator

This is the main entry point for the recommendation system.
It orchestrates the three-stage pipeline:

1. CANDIDATE GENERATION (embeddings.py)
   - Fast approximate retrieval using FAISS
   - Rule-based filtering

2. RANKING (ranking.py) 
   - Feature engineering (feature_engineering.py)
   - Learning-to-rank model

3. RE-RANKING (reranking.py)
   - Diversity injection
   - Exploration vs exploitation
   - Business rules

4. ENHANCEMENT (llm_integration.py)
   - LLM-powered explanations
   - Risk identification
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
import numpy as np
import os
import time

from .entities import (
    Creator, Campaign, MatchResult, FeatureVector,
    RecommendationType, RecommendationConfig
)
from .feature_engineering import FeatureEngineering
from .embeddings import EmbeddingService, EmbeddingConfig
from .ranking import RankingModel, RankingModelConfig
from .reranking import ReRanker, BusinessRuleFilter
from .llm_integration import LLMExplainer, LLMMatchEnhancer

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Industry-grade recommendation engine for influencer-brand matching.
    
    This implements best practices from YouTube, Instagram, and Alibaba:
    - Two-tower architecture for candidate generation
    - Learning-to-rank for precise scoring
    - Thompson sampling for exploration
    - LLM enhancement for explanations
    """
    
    def __init__(self, config: Optional[RecommendationConfig] = None):
        self.config = config or RecommendationConfig()
        
        # Initialize components
        self.feature_engineering = FeatureEngineering()
        self.embedding_service = EmbeddingService(
            EmbeddingConfig(model_name=self.config.embedding_model)
        )
        self.ranking_model = RankingModel(
            RankingModelConfig(model_path=self.config.ranking_model_path)
        )
        self.reranker = ReRanker(self.config)
        self.llm_explainer = LLMExplainer(self.config.llm_model)
        self.llm_enhancer = LLMMatchEnhancer(self.config.llm_model)
        
        # Creator cache
        self._creators_cache: Dict[str, Creator] = {}
        self._index_built = False
        
        # Metrics
        self._request_count = 0
        self._total_latency_ms = 0
        
    def initialize(self, creators: List[Creator]) -> None:
        """
        Initialize the recommendation engine with creator data.
        
        This should be called on startup or when creators are updated.
        It:
        1. Pre-computes creator embeddings
        2. Builds FAISS index for fast retrieval
        3. Loads ranking model
        """
        logger.info(f"Initializing recommendation engine with {len(creators)} creators")
        start_time = time.time()
        
        # Cache creators
        for creator in creators:
            self._creators_cache[creator.id] = creator
        
        # Compute embeddings and build index
        for creator in creators:
            creator.embedding = self.embedding_service.encode_creator(creator)
        
        self.embedding_service.build_faiss_index(creators)
        self._index_built = True
        
        # Load ranking model
        model_loaded = self.ranking_model.load()
        if not model_loaded:
            logger.warning("Ranking model not found, will use rule-based scoring")
        
        elapsed = time.time() - start_time
        logger.info(f"Recommendation engine initialized in {elapsed:.2f}s")
    
    def recommend_creators_for_campaign(
        self,
        campaign: Campaign,
        limit: int = 20,
        exclusions: Optional[Set[str]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MatchResult]:
        """
        Get top creator recommendations for a campaign.
        
        This is the main API for brand users.
        
        Args:
            campaign: The campaign to find creators for
            limit: Maximum number of recommendations
            exclusions: Creator IDs to exclude
            filters: Additional filters (platform, tier, etc.)
            
        Returns:
            List of MatchResult objects, sorted by score
        """
        self._request_count += 1
        start_time = time.time()
        
        try:
            # Stage 1: Candidate Generation
            candidates = self._generate_candidates(
                campaign, 
                limit=self.config.candidate_limit,
                filters=filters
            )
            logger.debug(f"Generated {len(candidates)} candidates")
            
            if not candidates:
                return []
            
            # Stage 2: Ranking
            ranked = self._rank_candidates(candidates, campaign)
            logger.debug(f"Ranked {len(ranked)} candidates")
            
            # Apply business rule filters
            filtered = BusinessRuleFilter.apply_filters(
                ranked, campaign, exclusions
            )
            logger.debug(f"After filtering: {len(filtered)} candidates")
            
            # Stage 3: Re-ranking (diversity + exploration)
            reranked = self.reranker.rerank(
                filtered,
                campaign,
                limit=limit,
                diversity_weight=self.config.diversity_weight,
                exploration_rate=self.config.exploration_rate
            )
            
            # Extract match results and enhance
            results = []
            for creator, match_result in reranked:
                # Add explanation if enabled
                if self.config.use_llm_explanations:
                    match_result = self.llm_enhancer.enhance_match_result(
                        creator, campaign, match_result
                    )
                
                # Record impression for exploration tracking
                self.reranker.record_impression(creator.id)
                
                results.append(match_result)
            
            # Track latency
            elapsed_ms = (time.time() - start_time) * 1000
            self._total_latency_ms += elapsed_ms
            logger.info(f"Recommendations generated in {elapsed_ms:.1f}ms")
            
            return results
            
        except Exception as e:
            logger.error(f"Recommendation failed: {e}")
            raise
    
    def _generate_candidates(
        self,
        campaign: Campaign,
        limit: int = 500,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Creator, float]]:
        """
        Stage 1: Generate candidate creators using ANN search.
        
        Returns list of (creator, candidate_score) tuples.
        """
        # Encode campaign
        campaign.embedding = self.embedding_service.encode_campaign(campaign)
        
        candidates = []
        
        # Use FAISS for fast retrieval if available
        if self._index_built:
            similar = self.embedding_service.search_similar_creators(
                campaign, k=min(limit * 2, 1000)  # Get more for filtering
            )
            
            for creator_id, similarity in similar:
                creator = self._creators_cache.get(creator_id)
                if creator:
                    # Apply quick filters
                    if self._quick_filter(creator, campaign, filters):
                        candidates.append((creator, similarity))
        else:
            # Fallback: linear scan with embedding similarity
            for creator in self._creators_cache.values():
                if self._quick_filter(creator, campaign, filters):
                    creator.embedding = self.embedding_service.encode_creator(creator)
                    similarity = self.embedding_service.compute_similarity(
                        creator.embedding, campaign.embedding
                    )
                    candidates.append((creator, similarity))
        
        # Sort by similarity and limit
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:limit]
    
    def _quick_filter(
        self,
        creator: Creator,
        campaign: Campaign,
        filters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Quick filtering for candidate generation stage"""
        # Platform filter
        if campaign.platform and campaign.platform.lower() not in ['all', 'any']:
            platforms = [p.strip().lower() for p in campaign.platform.split(',')]
            if creator.platform.lower() not in platforms:
                return False
        
        # Follower range filter
        if creator.followers < campaign.min_followers * 0.5:  # Some tolerance
            return False
        if campaign.max_followers and creator.followers > campaign.max_followers * 2:
            return False
        
        # Additional filters
        if filters:
            if 'tier' in filters and creator.tier not in filters['tier']:
                return False
            if 'location' in filters and creator.location not in filters['location']:
                return False
        
        return True
    
    def _rank_candidates(
        self,
        candidates: List[Tuple[Creator, float]],
        campaign: Campaign
    ) -> List[Tuple[Creator, MatchResult]]:
        """
        Stage 2: Rank candidates using the ranking model.
        
        Returns list of (creator, match_result) tuples sorted by ranking score.
        """
        ranked = []
        
        # Batch feature computation
        creators = [c[0] for c in candidates]
        candidate_scores = {c[0].id: c[1] for c in candidates}
        
        feature_vectors = self.feature_engineering.compute_batch_features(
            creators, campaign
        )
        
        # Prepare feature matrix
        feature_matrix = np.array([fv.to_array() for _, fv in feature_vectors])
        
        # Get ranking scores
        ranking_scores = self.ranking_model.predict(feature_matrix)
        
        # Build results
        for i, (creator_id, fv) in enumerate(feature_vectors):
            creator = self._creators_cache[creator_id]
            
            match_result = MatchResult(
                creator_id=creator_id,
                campaign_id=campaign.id,
                candidate_score=candidate_scores[creator_id] * 100,
                ranking_score=float(ranking_scores[i] * 100),
            )
            
            # Add score breakdown
            match_result.score_breakdown = {
                'category_similarity': fv.category_similarity,
                'platform_match': fv.platform_match,
                'engagement_fit': fv.engagement_fit,
                'budget_fit': fv.budget_fit,
                'embedding_similarity': fv.embedding_similarity,
                'creator_quality': fv.creator_quality_score,
            }
            
            # Add predictions
            match_result.predicted_success_probability = float(ranking_scores[i])
            match_result.predicted_engagement = self._estimate_engagement(
                creator, campaign, ranking_scores[i]
            )
            match_result.predicted_roi = self._estimate_roi(
                creator, campaign, ranking_scores[i]
            )
            
            ranked.append((creator, match_result))
        
        # Sort by ranking score
        ranked.sort(key=lambda x: x[1].ranking_score, reverse=True)
        return ranked
    
    def _estimate_engagement(
        self,
        creator: Creator,
        campaign: Campaign,
        match_score: float
    ) -> float:
        """Estimate expected engagement for this match"""
        base_engagement = creator.followers * creator.engagement_rate
        
        # Adjust by match quality
        adjustment = 0.7 + 0.3 * match_score
        
        return base_engagement * adjustment
    
    def _estimate_roi(
        self,
        creator: Creator,
        campaign: Campaign,
        match_score: float
    ) -> float:
        """Estimate ROI for this collaboration"""
        if creator.avg_cost <= 0 or campaign.budget <= 0:
            return 1.0
        
        # Simplified ROI estimation
        expected_reach = creator.followers * (1 + creator.engagement_rate)
        cost_per_reach = creator.avg_cost / expected_reach
        
        # Benchmark CPM
        benchmark_cpm = 10.0  # $10 per 1000 impressions
        
        actual_cpm = (creator.avg_cost / expected_reach) * 1000
        
        # ROI relative to benchmark
        roi = benchmark_cpm / max(0.01, actual_cpm)
        
        # Adjust by match quality
        roi *= (0.8 + 0.4 * match_score)
        
        return min(5.0, max(0.1, roi))  # Cap at 5x
    
    def get_match_explanation(
        self,
        creator_id: str,
        campaign: Campaign,
        detailed: bool = False
    ) -> str:
        """Get detailed explanation for a specific match"""
        creator = self._creators_cache.get(creator_id)
        if not creator:
            return "Creator not found"
        
        # Generate match result
        fv = self.feature_engineering.compute_features(creator, campaign)
        score = self.ranking_model.predict_single(fv)
        
        match_result = MatchResult(
            creator_id=creator_id,
            campaign_id=campaign.id,
            ranking_score=score * 100,
        )
        
        return self.llm_explainer.generate_match_explanation(
            creator, campaign, match_result, detailed
        )
    
    def record_interaction(
        self,
        creator_id: str,
        campaign_id: str,
        interaction_type: str
    ) -> None:
        """
        Record user interaction for model improvement.
        
        Interaction types:
        - click: Brand clicked on creator
        - save: Brand saved creator for later
        - invite: Brand invited creator
        - accept: Creator accepted invitation
        - complete: Campaign completed
        """
        if interaction_type == 'click':
            self.reranker.record_click(creator_id)
        elif interaction_type == 'conversion':
            self.reranker.record_conversion(creator_id)
        
        logger.info(f"Recorded interaction: {interaction_type} for creator {creator_id}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get recommendation engine metrics"""
        avg_latency = (
            self._total_latency_ms / self._request_count 
            if self._request_count > 0 else 0
        )
        
        return {
            'request_count': self._request_count,
            'avg_latency_ms': avg_latency,
            'creators_indexed': len(self._creators_cache),
            'index_built': self._index_built,
            'model_version': self.ranking_model._model_version,
            'exploration_stats': self.reranker.get_exploration_stats(),
            'feature_importance': self.ranking_model.get_feature_importance(),
        }
    
    def update_creator(self, creator: Creator) -> None:
        """Update a single creator in the cache"""
        creator.embedding = self.embedding_service.encode_creator(creator)
        self._creators_cache[creator.id] = creator
        # Note: FAISS index would need to be rebuilt for this to take effect
    
    def batch_predict(
        self,
        creator_ids: List[str],
        campaign: Campaign
    ) -> Dict[str, float]:
        """
        Get match scores for specific creators.
        
        This is useful for batch evaluation or when the client
        already knows which creators to evaluate.
        """
        results = {}
        
        campaign.embedding = self.embedding_service.encode_campaign(campaign)
        
        for creator_id in creator_ids:
            creator = self._creators_cache.get(creator_id)
            if not creator:
                continue
            
            fv = self.feature_engineering.compute_features(creator, campaign)
            score = self.ranking_model.predict_single(fv)
            results[creator_id] = score * 100
        
        return results


# Singleton instance for the application
_engine_instance: Optional[RecommendationEngine] = None


def get_recommendation_engine() -> RecommendationEngine:
    """Get or create the singleton recommendation engine"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RecommendationEngine()
    return _engine_instance


def initialize_recommendation_engine(creators: List[Creator]) -> RecommendationEngine:
    """Initialize the recommendation engine with creator data"""
    engine = get_recommendation_engine()
    engine.initialize(creators)
    return engine
