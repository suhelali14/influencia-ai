"""
Comprehensive Feature Engineering Pipeline
Extracts 50+ features for ML models including:
- Behavioral features (past performance, engagement patterns)
- Temporal features (trends, seasonality)
- Contextual features (audience fit, budget alignment)
- Network features (influence, reach)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json


def parse_field(value, default=[]):
    """Parse JSON or comma-separated string fields"""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        if not value.strip():
            return default
        # Try JSON first
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            # Fall back to comma-separated
            return [item.strip() for item in value.split(',') if item.strip()]
    return default


class FeatureEngineer:
    """Extract and compute features for creator-campaign matching"""
    
    @staticmethod
    def creator_features(creator: Dict, historical_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Compute comprehensive creator features (30+ features)
        
        Args:
            creator: Creator profile dictionary
            historical_data: Time-series performance data
        
        Returns:
            Dictionary of engineered features
        """
        features = {}
        
        # === BASIC STATS ===
        features['follower_count'] = float(creator.get('followers', 0))
        features['following_count'] = float(creator.get('following', 0))
        features['total_posts'] = float(creator.get('total_posts', 0))
        features['account_age_days'] = float(creator.get('account_age_days', 0))
        
        # Follower-to-following ratio (indicator of influence)
        features['follower_following_ratio'] = features['follower_count'] / max(features['following_count'], 1)
        
        # === ENGAGEMENT METRICS ===
        base_engagement = float(creator.get('engagement_rate', 0))
        features['engagement_rate'] = base_engagement
        
        if historical_data is not None and len(historical_data) > 0:
            features['engagement_rate_7d'] = FeatureEngineer._compute_engagement(
                historical_data, window_days=7
            )
            features['engagement_rate_30d'] = FeatureEngineer._compute_engagement(
                historical_data, window_days=30
            )
            features['engagement_rate_90d'] = FeatureEngineer._compute_engagement(
                historical_data, window_days=90
            )
            
            # Engagement trend
            features['engagement_trend'] = FeatureEngineer._compute_trend(
                historical_data, metric='engagement_rate'
            )
        else:
            # Use base engagement if no historical data
            features['engagement_rate_7d'] = base_engagement
            features['engagement_rate_30d'] = base_engagement
            features['engagement_rate_90d'] = base_engagement
            features['engagement_trend'] = 0.0
        
        # === GROWTH METRICS ===
        if historical_data is not None and len(historical_data) > 0:
            features['follower_growth_7d'] = FeatureEngineer._compute_growth(
                historical_data, metric='followers', window_days=7
            )
            features['follower_growth_30d'] = FeatureEngineer._compute_growth(
                historical_data, metric='followers', window_days=30
            )
            features['follower_growth_rate'] = features['follower_growth_30d']
        else:
            features['follower_growth_7d'] = 0.0
            features['follower_growth_30d'] = 0.0
            features['follower_growth_rate'] = 0.0
        
        # === CONTENT QUALITY ===
        total_posts = max(features['total_posts'], 1)
        features['avg_likes_per_post'] = features['follower_count'] * base_engagement
        features['avg_comments_per_post'] = features['avg_likes_per_post'] * 0.1  # Estimate
        features['avg_shares_per_post'] = features['avg_likes_per_post'] * 0.05  # Estimate
        
        # === CONSISTENCY ===
        if historical_data is not None and len(historical_data) > 0:
            features['posting_frequency'] = FeatureEngineer._compute_posting_frequency(
                historical_data
            )
            features['posting_consistency_score'] = FeatureEngineer._compute_consistency(
                historical_data
            )
        else:
            # Estimate from total posts and account age
            features['posting_frequency'] = features['total_posts'] / max(features['account_age_days'], 1)
            features['posting_consistency_score'] = 0.5  # Neutral
        
        # === COLLABORATION HISTORY ===
        features['total_campaigns'] = float(creator.get('total_campaigns', 0))
        features['successful_campaigns'] = float(creator.get('successful_campaigns', 0))
        features['success_rate'] = float(creator.get('success_rate', 0))
        features['avg_campaign_rating'] = float(creator.get('overall_rating', 0))
        features['total_earnings'] = float(creator.get('total_earnings', 0))
        features['avg_earnings_per_campaign'] = features['total_earnings'] / max(features['total_campaigns'], 1)
        
        # === AUDIENCE DEMOGRAPHICS ===
        features['audience_age_18_24_pct'] = float(creator.get('audience_age_18_24', 0))
        features['audience_age_25_34_pct'] = float(creator.get('audience_age_25_34', 0))
        features['audience_age_35_44_pct'] = float(creator.get('audience_age_35_44', 0))
        features['audience_female_pct'] = float(creator.get('audience_female_pct', 50))
        features['audience_male_pct'] = float(creator.get('audience_male_pct', 50))
        
        # === REACH & INFLUENCE ===
        features['estimated_reach'] = features['follower_count'] * features['engagement_rate_30d']
        features['influence_score'] = FeatureEngineer._compute_influence_score(creator)
        
        # === VERSATILITY ===
        categories = parse_field(creator.get('categories'), [])
        platforms = parse_field(creator.get('platforms'), [])
        
        features['num_categories'] = float(len(categories))
        features['num_platforms'] = float(len(platforms))
        
        # === VERIFICATION & TRUST ===
        features['is_verified'] = float(creator.get('is_verified', False))
        
        # === TIER ENCODING (one-hot) ===
        tier = creator.get('tier', 'micro')
        features['tier_nano'] = 1.0 if tier == 'nano' else 0.0
        features['tier_micro'] = 1.0 if tier == 'micro' else 0.0
        features['tier_mid'] = 1.0 if tier == 'mid' else 0.0
        features['tier_macro'] = 1.0 if tier == 'macro' else 0.0
        features['tier_mega'] = 1.0 if tier == 'mega' else 0.0
        
        return features
    
    @staticmethod
    def campaign_features(campaign: Dict, brand_data: Optional[Dict] = None) -> Dict[str, float]:
        """
        Compute campaign features (15+ features)
        
        Args:
            campaign: Campaign details dictionary
            brand_data: Brand information (optional)
        
        Returns:
            Dictionary of engineered features
        """
        features = {}
        
        # === BASIC INFO ===
        features['budget_total'] = float(campaign.get('budget', 0))
        features['duration_days'] = float(campaign.get('duration_days', 30))
        features['budget_per_day'] = features['budget_total'] / max(features['duration_days'], 1)
        
        # === REQUIREMENTS ===
        features['min_followers_required'] = float(campaign.get('min_followers', 0))
        features['min_engagement_required'] = float(campaign.get('min_engagement', 0))
        
        # Parse deliverables
        deliverables = parse_field(campaign.get('deliverables'), [])
        features['num_deliverables'] = float(len(deliverables))
        
        features['content_rights_required'] = float(campaign.get('content_rights', False))
        
        # === BRAND REPUTATION ===
        if brand_data:
            features['brand_total_campaigns'] = float(brand_data.get('total_campaigns', 0))
            features['brand_avg_rating'] = float(brand_data.get('avg_rating', 0))
            features['brand_completion_rate'] = float(brand_data.get('completion_rate', 0))
        else:
            features['brand_total_campaigns'] = float(campaign.get('brand_total_campaigns', 0))
            features['brand_avg_rating'] = float(campaign.get('brand_avg_rating', 0))
            features['brand_completion_rate'] = float(campaign.get('brand_completion_rate', 0))
        
        # === COMPETITIVENESS ===
        features['applications_count'] = float(campaign.get('applications_count', 0))
        features['competition_level'] = min(features['applications_count'] / 100.0, 1.0)
        
        # === URGENCY ===
        start_date_str = campaign.get('start_date')
        if start_date_str:
            if isinstance(start_date_str, str):
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
            else:
                start_date = start_date_str
            days_until_start = (start_date - datetime.now()).days
            features['days_until_start'] = max(float(days_until_start), 0.0)
            features['is_urgent'] = 1.0 if days_until_start < 7 else 0.0
        else:
            features['days_until_start'] = 30.0
            features['is_urgent'] = 0.0
        
        return features
    
    @staticmethod
    def interaction_features(creator: Dict, campaign: Dict) -> Dict[str, float]:
        """
        Compute creator-campaign interaction features (10+ features)
        
        Args:
            creator: Creator profile
            campaign: Campaign details
        
        Returns:
            Dictionary of interaction features
        """
        features = {}
        
        # === MATCH SCORES ===
        creator_categories = parse_field(creator.get('categories'), [])
        creator_categories_set = set(creator_categories)
        
        campaign_category = campaign.get('category', '')
        features['category_exact_match'] = 1.0 if campaign_category in creator_categories_set else 0.0
        
        creator_platforms = parse_field(creator.get('platforms'), [])
        creator_platforms_set = set(creator_platforms)
        
        campaign_platform = campaign.get('platform', '')
        features['platform_match'] = 1.0 if campaign_platform in creator_platforms_set else 0.0
        
        # === BUDGET FIT ===
        creator_avg_earnings = float(creator.get('total_earnings', 0)) / max(float(creator.get('total_campaigns', 1)), 1.0)
        campaign_budget = float(campaign.get('budget', 0))
        
        features['budget_ratio'] = campaign_budget / max(creator_avg_earnings, 1.0)
        features['budget_fit_score'] = FeatureEngineer._sigmoid(
            features['budget_ratio'], midpoint=1.0, steepness=2
        )
        
        # === AUDIENCE OVERLAP ===
        # Age overlap (simplified - could be more sophisticated)
        creator_age_18_24 = float(creator.get('audience_age_18_24', 0))
        creator_age_25_34 = float(creator.get('audience_age_25_34', 0))
        
        # Assuming campaign targets young adults (could be extracted from campaign)
        features['audience_age_overlap'] = (creator_age_18_24 + creator_age_25_34) / 100.0
        
        # Gender overlap
        creator_female_pct = float(creator.get('audience_female_pct', 50))
        target_gender = campaign.get('target_gender', 'All')
        
        if target_gender == 'Female':
            features['audience_gender_overlap'] = creator_female_pct / 100.0
        elif target_gender == 'Male':
            features['audience_gender_overlap'] = (100 - creator_female_pct) / 100.0
        else:
            features['audience_gender_overlap'] = 1.0  # All genders
        
        # === REQUIREMENT FIT ===
        creator_followers = float(creator.get('followers', 0))
        min_followers = float(campaign.get('min_followers', 0))
        features['meets_follower_requirement'] = 1.0 if creator_followers >= min_followers else 0.0
        features['follower_excess_ratio'] = creator_followers / max(min_followers, 1.0)
        
        creator_engagement = float(creator.get('engagement_rate', 0))
        min_engagement = float(campaign.get('min_engagement', 0))
        features['meets_engagement_requirement'] = 1.0 if creator_engagement >= min_engagement else 0.0
        features['engagement_excess_ratio'] = creator_engagement / max(min_engagement, 0.001)
        
        # === EXPERIENCE FIT ===
        total_campaigns = float(creator.get('total_campaigns', 0))
        features['experience_score'] = min(total_campaigns / 10.0, 1.0)
        
        # === QUALITY INDICATORS ===
        features['creator_rating'] = float(creator.get('overall_rating', 0)) / 5.0  # Normalize to 0-1
        features['creator_success_rate'] = float(creator.get('success_rate', 0))
        
        return features
    
    @staticmethod
    def combined_features(creator: Dict, campaign: Dict, historical_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Generate all features: creator + campaign + interaction
        
        Returns:
            Complete feature dictionary with 50+ features
        """
        creator_feats = FeatureEngineer.creator_features(creator, historical_data)
        campaign_feats = FeatureEngineer.campaign_features(campaign)
        interaction_feats = FeatureEngineer.interaction_features(creator, campaign)
        
        # Combine all features
        all_features = {**creator_feats, **campaign_feats, **interaction_feats}
        
        # Add prefix to avoid name collisions
        return {
            **{f'creator_{k}': v for k, v in creator_feats.items()},
            **{f'campaign_{k}': v for k, v in campaign_feats.items()},
            **{f'interaction_{k}': v for k, v in interaction_feats.items()}
        }
    
    # ========== HELPER METHODS ==========
    
    @staticmethod
    def _compute_engagement(historical_data: pd.DataFrame, window_days: int) -> float:
        """Compute engagement rate for specific time window"""
        if len(historical_data) == 0:
            return 0.0
        
        cutoff_date = datetime.now() - timedelta(days=window_days)
        cutoff_str = cutoff_date.strftime('%Y-%m-%d')
        
        recent_data = historical_data[historical_data['date'] >= cutoff_str]
        
        if len(recent_data) == 0:
            return 0.0
        
        return float(recent_data['engagement_rate'].mean())
    
    @staticmethod
    def _compute_growth(historical_data: pd.DataFrame, metric: str, window_days: int) -> float:
        """Compute growth rate for a metric"""
        if len(historical_data) < 2:
            return 0.0
        
        cutoff_date = datetime.now() - timedelta(days=window_days)
        cutoff_str = cutoff_date.strftime('%Y-%m-%d')
        
        recent_data = historical_data[historical_data['date'] >= cutoff_str].sort_values('date')
        
        if len(recent_data) < 2:
            return 0.0
        
        start_value = float(recent_data.iloc[0][metric])
        end_value = float(recent_data.iloc[-1][metric])
        
        if start_value == 0:
            return 0.0
        
        growth_rate = (end_value - start_value) / start_value
        return float(growth_rate)
    
    @staticmethod
    def _compute_trend(historical_data: pd.DataFrame, metric: str) -> float:
        """Compute trend direction (-1 to 1)"""
        if len(historical_data) < 5:
            return 0.0
        
        sorted_data = historical_data.sort_values('date')
        values = sorted_data[metric].values
        
        # Simple linear trend
        x = np.arange(len(values))
        coefficients = np.polyfit(x, values, 1)
        slope = coefficients[0]
        
        # Normalize slope to -1 to 1 range
        mean_value = np.mean(values)
        if mean_value == 0:
            return 0.0
        
        normalized_slope = slope / mean_value
        return float(np.clip(normalized_slope, -1, 1))
    
    @staticmethod
    def _compute_posting_frequency(historical_data: pd.DataFrame) -> float:
        """Compute average posts per day"""
        if len(historical_data) == 0:
            return 0.0
        
        total_posts = historical_data['posts'].sum()
        num_days = len(historical_data)
        
        return float(total_posts / max(num_days, 1))
    
    @staticmethod
    def _compute_consistency(historical_data: pd.DataFrame) -> float:
        """
        Compute posting consistency score (0-1)
        Higher score = more consistent posting schedule
        """
        if len(historical_data) == 0:
            return 0.0
        
        posts_per_day = historical_data['posts'].values
        
        # Variance in posting frequency
        variance = np.var(posts_per_day)
        mean_posts = np.mean(posts_per_day)
        
        if mean_posts == 0:
            return 0.0
        
        # Coefficient of variation (lower = more consistent)
        cv = np.sqrt(variance) / mean_posts
        
        # Convert to 0-1 score (inverse relationship)
        consistency_score = 1 / (1 + cv)
        
        return float(consistency_score)
    
    @staticmethod
    def _compute_influence_score(creator: Dict) -> float:
        """
        Compute overall influence score (0-1)
        Weighted combination of follower count, engagement, and success rate
        """
        followers = float(creator.get('followers', 0))
        engagement = float(creator.get('engagement_rate', 0))
        success_rate = float(creator.get('success_rate', 0))
        
        # Log-scale for followers (normalize to 0-1)
        followers_score = min(np.log10(max(followers, 1)) / 7, 1.0)  # 10M followers = 1.0
        
        # Engagement already 0-1 range (percentage)
        engagement_score = min(engagement / 0.1, 1.0)  # 10% engagement = 1.0
        
        # Success rate already 0-1
        success_score = success_rate
        
        # Weighted average
        influence = (
            0.4 * followers_score +
            0.3 * engagement_score +
            0.3 * success_score
        )
        
        return float(influence)
    
    @staticmethod
    def _sigmoid(x: float, midpoint: float = 1.0, steepness: float = 1.0) -> float:
        """Sigmoid function for smooth transitions"""
        return float(1 / (1 + np.exp(-steepness * (x - midpoint))))


def features_to_vector(features: Dict[str, float], feature_names: List[str]) -> np.ndarray:
    """
    Convert feature dictionary to numpy vector in consistent order
    
    Args:
        features: Dictionary of feature name -> value
        feature_names: Ordered list of feature names
    
    Returns:
        Numpy array of feature values
    """
    return np.array([features.get(name, 0.0) for name in feature_names])


def save_feature_names(feature_names: List[str], filepath: str):
    """Save feature names for consistency"""
    with open(filepath, 'w') as f:
        json.dump(feature_names, f, indent=2)


def load_feature_names(filepath: str) -> List[str]:
    """Load feature names"""
    with open(filepath, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    # Example usage
    sample_creator = {
        'followers': 50000,
        'following': 1000,
        'engagement_rate': 0.045,
        'total_posts': 500,
        'account_age_days': 730,
        'tier': 'micro',
        'categories': json.dumps(['Fashion', 'Lifestyle']),
        'platforms': json.dumps(['Instagram', 'TikTok']),
        'total_campaigns': 25,
        'successful_campaigns': 20,
        'success_rate': 0.8,
        'overall_rating': 4.5,
        'total_earnings': 35000,
        'is_verified': True,
        'audience_age_18_24': 40,
        'audience_age_25_34': 35,
        'audience_age_35_44': 15,
        'audience_female_pct': 70,
        'audience_male_pct': 30
    }
    
    sample_campaign = {
        'budget': 2000,
        'duration_days': 30,
        'category': 'Fashion',
        'platform': 'Instagram',
        'min_followers': 10000,
        'min_engagement': 0.03,
        'deliverables': json.dumps(['Post', 'Story', 'Reel']),
        'content_rights': True,
        'brand_total_campaigns': 15,
        'brand_avg_rating': 4.2,
        'brand_completion_rate': 0.85,
        'applications_count': 45,
        'start_date': '2025-12-01'
    }
    
    # Generate features
    features = FeatureEngineer.combined_features(sample_creator, sample_campaign)
    
    print(f"Generated {len(features)} features:")
    for name, value in sorted(features.items())[:10]:
        print(f"  {name}: {value:.4f}")
    print(f"  ... and {len(features) - 10} more")
