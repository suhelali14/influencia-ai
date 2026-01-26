"""
Re-ranking Module - Business Rules & Diversity

This module applies post-processing to ranked candidates:
1. Diversity injection (avoid recommending similar creators)
2. Freshness boost (promote new creators)
3. Fairness constraints (balanced representation)
4. Exploration vs Exploitation (Thompson Sampling)
5. Business rule filters

This is crucial for a healthy marketplace:
- New creators get exposure
- Brands see diverse options
- The system explores to improve
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import logging
from datetime import datetime
import random

from .entities import Creator, Campaign, MatchResult, RecommendationConfig

logger = logging.getLogger(__name__)


class ReRanker:
    """
    Re-ranking module for recommendation diversity and exploration.
    
    This ensures recommendations are not just accurate but also:
    - Diverse (different categories, tiers, styles)
    - Fair (balanced exposure across creators)
    - Explorative (test new creators)
    """
    
    def __init__(self, config: Optional[RecommendationConfig] = None):
        self.config = config or RecommendationConfig()
        self._creator_impressions = defaultdict(int)  # Track exposure
        self._creator_clicks = defaultdict(int)  # Track engagement
        
    def rerank(
        self,
        candidates: List[Tuple[Creator, MatchResult]],
        campaign: Campaign,
        limit: int = 20,
        diversity_weight: float = 0.3,
        exploration_rate: float = 0.1
    ) -> List[Tuple[Creator, MatchResult]]:
        """
        Re-rank candidates with diversity and exploration.
        
        Args:
            candidates: List of (creator, match_result) tuples, sorted by ranking score
            campaign: The campaign being matched
            limit: Maximum number of results to return
            diversity_weight: Weight for diversity vs relevance (0-1)
            exploration_rate: Fraction of slots for exploration (0-1)
            
        Returns:
            Re-ranked list of (creator, match_result) tuples
        """
        if not candidates:
            return []
        
        # Separate exploration slots
        n_explore = max(1, int(limit * exploration_rate))
        n_exploit = limit - n_explore
        
        # Select exploitation candidates (top by score with diversity)
        exploit_candidates = self._select_diverse(
            candidates, 
            n_exploit, 
            diversity_weight
        )
        
        # Select exploration candidates (underexposed creators)
        remaining = [c for c in candidates if c not in exploit_candidates]
        explore_candidates = self._select_exploration(
            remaining, 
            n_explore
        )
        
        # Combine and interleave
        final = self._interleave(exploit_candidates, explore_candidates)
        
        # Apply fairness constraints
        final = self._apply_fairness(final, limit)
        
        # Update final scores
        for i, (creator, result) in enumerate(final):
            # Position-based score adjustment
            position_factor = 1.0 - (i / len(final)) * 0.2
            result.final_score = result.ranking_score * position_factor
        
        return final[:limit]
    
    def _select_diverse(
        self,
        candidates: List[Tuple[Creator, MatchResult]],
        n: int,
        diversity_weight: float
    ) -> List[Tuple[Creator, MatchResult]]:
        """
        Select diverse candidates using MMR-like algorithm.
        
        MMR (Maximal Marginal Relevance) balances relevance and diversity.
        """
        if not candidates or n <= 0:
            return []
        
        selected = []
        remaining = list(candidates)
        
        # Start with highest scoring candidate
        remaining.sort(key=lambda x: x[1].ranking_score, reverse=True)
        selected.append(remaining.pop(0))
        
        while len(selected) < n and remaining:
            # Score remaining candidates by MMR
            best_score = -float('inf')
            best_idx = 0
            
            for i, (creator, result) in enumerate(remaining):
                # Relevance score
                relevance = result.ranking_score
                
                # Diversity score (min similarity to already selected)
                diversity = self._compute_diversity(
                    creator, 
                    [s[0] for s in selected]
                )
                
                # MMR score
                mmr_score = (1 - diversity_weight) * relevance + diversity_weight * diversity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    def _compute_diversity(
        self, 
        creator: Creator, 
        selected_creators: List[Creator]
    ) -> float:
        """Compute diversity score of creator vs already selected creators"""
        if not selected_creators:
            return 1.0
        
        max_similarity = 0.0
        
        for selected in selected_creators:
            similarity = self._creator_similarity(creator, selected)
            max_similarity = max(max_similarity, similarity)
        
        # Diversity is inverse of max similarity
        return 1.0 - max_similarity
    
    def _creator_similarity(self, c1: Creator, c2: Creator) -> float:
        """Compute similarity between two creators"""
        similarity = 0.0
        n_factors = 0
        
        # Category overlap
        if c1.categories and c2.categories:
            overlap = len(set(c1.categories) & set(c2.categories))
            total = len(set(c1.categories) | set(c2.categories))
            similarity += overlap / total if total > 0 else 0
            n_factors += 1
        
        # Same tier
        if c1.tier == c2.tier:
            similarity += 0.5
        n_factors += 1
        
        # Same platform
        if c1.platform == c2.platform:
            similarity += 0.3
        n_factors += 1
        
        # Similar follower count (within 2x)
        follower_ratio = max(c1.followers, c2.followers) / max(1, min(c1.followers, c2.followers))
        if follower_ratio <= 2:
            similarity += 0.3
        n_factors += 1
        
        # Same location
        if c1.location and c2.location and c1.location.lower() == c2.location.lower():
            similarity += 0.2
        n_factors += 1
        
        return similarity / n_factors if n_factors > 0 else 0
    
    def _select_exploration(
        self,
        candidates: List[Tuple[Creator, MatchResult]],
        n: int
    ) -> List[Tuple[Creator, MatchResult]]:
        """
        Select exploration candidates using Thompson Sampling.
        
        Thompson Sampling balances exploration and exploitation by
        sampling from posterior distributions of creator quality.
        """
        if not candidates or n <= 0:
            return []
        
        # Score candidates by exploration value
        scored = []
        for creator, result in candidates:
            # Thompson Sampling: sample from Beta distribution
            # Beta(successes + 1, failures + 1)
            impressions = self._creator_impressions[creator.id]
            clicks = self._creator_clicks[creator.id]
            
            if impressions == 0:
                # New creator - high exploration value
                sampled_ctr = np.random.beta(1, 1)  # Uniform prior
                exploration_bonus = 0.3
            else:
                sampled_ctr = np.random.beta(clicks + 1, impressions - clicks + 1)
                exploration_bonus = 0.1 / (1 + np.log1p(impressions))
            
            # Combine ranking score with exploration value
            explore_score = result.ranking_score * 0.5 + sampled_ctr * 0.3 + exploration_bonus * 0.2
            scored.append((creator, result, explore_score))
        
        # Sort by exploration score and select top n
        scored.sort(key=lambda x: x[2], reverse=True)
        return [(c, r) for c, r, _ in scored[:n]]
    
    def _interleave(
        self,
        exploit: List[Tuple[Creator, MatchResult]],
        explore: List[Tuple[Creator, MatchResult]]
    ) -> List[Tuple[Creator, MatchResult]]:
        """Interleave exploitation and exploration candidates"""
        result = []
        exploit_idx = 0
        explore_idx = 0
        
        # Ratio: 4 exploit to 1 explore
        while exploit_idx < len(exploit) or explore_idx < len(explore):
            # Add 4 exploitation candidates
            for _ in range(4):
                if exploit_idx < len(exploit):
                    result.append(exploit[exploit_idx])
                    exploit_idx += 1
            
            # Add 1 exploration candidate
            if explore_idx < len(explore):
                result.append(explore[explore_idx])
                explore_idx += 1
        
        return result
    
    def _apply_fairness(
        self,
        candidates: List[Tuple[Creator, MatchResult]],
        limit: int
    ) -> List[Tuple[Creator, MatchResult]]:
        """
        Apply fairness constraints to ensure balanced representation.
        
        Constraints:
        - No single category dominates (max 50%)
        - At least 2 tiers represented (if available)
        - Mix of experience levels
        """
        if len(candidates) <= limit:
            return candidates
        
        result = []
        category_counts = defaultdict(int)
        tier_set = set()
        
        max_per_category = limit // 2
        
        for creator, match in candidates:
            if len(result) >= limit:
                break
            
            # Check category constraint
            main_category = creator.categories[0] if creator.categories else 'other'
            if category_counts[main_category] >= max_per_category:
                # Skip to add diversity
                continue
            
            result.append((creator, match))
            category_counts[main_category] += 1
            tier_set.add(creator.tier)
        
        # If we didn't fill the limit, add remaining
        for creator, match in candidates:
            if len(result) >= limit:
                break
            if (creator, match) not in result:
                result.append((creator, match))
        
        return result
    
    def record_impression(self, creator_id: str) -> None:
        """Record that a creator was shown to a brand"""
        self._creator_impressions[creator_id] += 1
    
    def record_click(self, creator_id: str) -> None:
        """Record that a brand clicked on a creator"""
        self._creator_clicks[creator_id] += 1
    
    def record_conversion(self, creator_id: str) -> None:
        """Record that a brand initiated collaboration with creator"""
        # Could be used for conversion optimization
        pass
    
    def get_exploration_stats(self) -> Dict[str, Any]:
        """Get statistics about exploration performance"""
        total_impressions = sum(self._creator_impressions.values())
        total_clicks = sum(self._creator_clicks.values())
        unique_creators = len(self._creator_impressions)
        
        return {
            'total_impressions': total_impressions,
            'total_clicks': total_clicks,
            'overall_ctr': total_clicks / max(1, total_impressions),
            'unique_creators_shown': unique_creators,
        }
    
    def reset_stats(self) -> None:
        """Reset exploration statistics"""
        self._creator_impressions.clear()
        self._creator_clicks.clear()


class BusinessRuleFilter:
    """
    Apply business rule filters to recommendations.
    
    These are hard constraints that must be satisfied.
    """
    
    @staticmethod
    def apply_filters(
        candidates: List[Tuple[Creator, MatchResult]],
        campaign: Campaign,
        exclusions: Optional[Set[str]] = None
    ) -> List[Tuple[Creator, MatchResult]]:
        """
        Apply business rule filters to candidates.
        
        Args:
            candidates: List of (creator, match_result) tuples
            campaign: The campaign being matched
            exclusions: Set of creator IDs to exclude
            
        Returns:
            Filtered list of candidates
        """
        filtered = []
        
        for creator, result in candidates:
            # Exclusion list
            if exclusions and creator.id in exclusions:
                continue
            
            # Platform match (hard constraint)
            if not BusinessRuleFilter._platform_compatible(creator, campaign):
                continue
            
            # Budget constraint (hard constraint)
            if creator.avg_cost > campaign.budget * 1.5:
                continue
            
            # Minimum quality threshold
            if creator.content_quality_score < 0.3:
                continue
            
            # Authenticity threshold
            if creator.audience_authenticity < 0.5:
                continue
            
            filtered.append((creator, result))
        
        return filtered
    
    @staticmethod
    def _platform_compatible(creator: Creator, campaign: Campaign) -> bool:
        """Check if creator's platform is compatible with campaign"""
        if not campaign.platform or campaign.platform.lower() in ['all', 'any']:
            return True
        
        campaign_platforms = [p.strip().lower() for p in campaign.platform.split(',')]
        return creator.platform.lower() in campaign_platforms
