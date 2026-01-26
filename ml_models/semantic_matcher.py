"""
Semantic Matching with Transformer Models
Uses sentence-transformers for efficient embedding generation and similarity computation

Based on BERT/RoBERTa architecture for understanding creator profiles and campaign descriptions
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
from sklearn.metrics.pairwise import cosine_similarity
import json


class SemanticMatcher:
    """
    Transformer-based semantic matching for campaign-creator fit
    Uses pre-trained sentence transformers optimized for semantic search
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: Optional[str] = None):
        """
        Initialize semantic matcher
        
        Args:
            model_name: Pre-trained sentence transformer model
                - 'all-MiniLM-L6-v2': Fast and efficient (384 dim)
                - 'all-mpnet-base-v2': Higher quality (768 dim)
                - 'multi-qa-mpnet-base-dot-v1': Optimized for Q&A
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        print(f"Loaded semantic model: {model_name}")
        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Device: {device}")
    
    def encode_creator_profile(self, creator: Dict) -> np.ndarray:
        """
        Create rich text representation of creator and encode to embedding
        
        Args:
            creator: Creator profile dictionary
        
        Returns:
            Embedding vector (numpy array)
        """
        # Parse JSON fields if needed
        categories = creator.get('categories', [])
        if isinstance(categories, str):
            # Try JSON first, fall back to comma-separated
            try:
                categories = json.loads(categories) if categories.strip() else []
            except (json.JSONDecodeError, ValueError):
                categories = [c.strip() for c in categories.split(',') if c.strip()]
        
        platforms = creator.get('platforms', [])
        if isinstance(platforms, str):
            try:
                platforms = json.loads(platforms) if platforms.strip() else []
            except (json.JSONDecodeError, ValueError):
                platforms = [p.strip() for p in platforms.split(',') if p.strip()]
        
        languages = creator.get('languages', [])
        if isinstance(languages, str):
            try:
                languages = json.loads(languages) if languages.strip() else []
            except (json.JSONDecodeError, ValueError):
                languages = [l.strip() for l in languages.split(',') if l.strip()]
        
        # Construct rich text profile
        text_parts = []
        
        # Bio
        bio = creator.get('bio', '')
        if bio:
            text_parts.append(f"Bio: {bio}")
        
        # Categories and expertise
        if categories:
            text_parts.append(f"Specializes in: {', '.join(categories)}")
        
        # Platforms
        if platforms:
            text_parts.append(f"Active on: {', '.join(platforms)}")
        
        # Audience size and engagement
        followers = creator.get('followers', 0)
        engagement = creator.get('engagement_rate', 0)
        tier = creator.get('tier', 'micro')
        
        text_parts.append(
            f"{tier.capitalize()} influencer with {self._format_number(followers)} followers "
            f"and {engagement*100:.1f}% engagement rate"
        )
        
        # Experience and track record
        total_campaigns = creator.get('total_campaigns', 0)
        success_rate = creator.get('success_rate', 0)
        rating = creator.get('overall_rating', 0)
        
        if total_campaigns > 0:
            text_parts.append(
                f"Completed {total_campaigns} campaigns with {success_rate*100:.0f}% success rate "
                f"and {rating:.1f}/5.0 average rating"
            )
        
        # Audience demographics
        age_18_24 = creator.get('audience_age_18_24', 0)
        age_25_34 = creator.get('audience_age_25_34', 0)
        female_pct = creator.get('audience_female_pct', 50)
        
        text_parts.append(
            f"Audience: {age_18_24}% ages 18-24, {age_25_34}% ages 25-34, "
            f"{female_pct}% female"
        )
        
        # Combine all parts
        full_text = " | ".join(text_parts)
        
        # Encode to embedding
        embedding = self.model.encode(full_text, convert_to_tensor=False)
        
        return embedding
    
    def encode_campaign(self, campaign: Dict, brand: Optional[Dict] = None) -> np.ndarray:
        """
        Create rich text representation of campaign and encode to embedding
        
        Args:
            campaign: Campaign details dictionary
            brand: Optional brand information
        
        Returns:
            Embedding vector (numpy array)
        """
        text_parts = []
        
        # Campaign title and description
        title = campaign.get('title', '')
        description = campaign.get('description', '')
        
        if title:
            text_parts.append(f"Campaign: {title}")
        if description:
            text_parts.append(f"Description: {description}")
        
        # Brand information
        if brand:
            brand_name = brand.get('company_name', '')
            brand_desc = brand.get('description', '')
            industry = brand.get('industry', '')
            
            if brand_name:
                text_parts.append(f"Brand: {brand_name}")
            if industry:
                text_parts.append(f"Industry: {industry}")
            if brand_desc:
                text_parts.append(f"About: {brand_desc}")
        else:
            # Use campaign-level brand info
            industry = campaign.get('industry', '')
            if industry:
                text_parts.append(f"Industry: {industry}")
        
        # Campaign details
        category = campaign.get('category', '')
        platform = campaign.get('platform', '')
        budget = campaign.get('budget', 0)
        duration = campaign.get('duration_days', 0)
        
        text_parts.append(
            f"{category} campaign on {platform} with ${self._format_number(budget)} budget "
            f"over {duration} days"
        )
        
        # Requirements and deliverables
        deliverables = campaign.get('deliverables', [])
        if isinstance(deliverables, str):
            deliverables = json.loads(deliverables)
        
        if deliverables:
            text_parts.append(f"Deliverables: {', '.join(deliverables)}")
        
        min_followers = campaign.get('min_followers', 0)
        min_engagement = campaign.get('min_engagement', 0)
        
        text_parts.append(
            f"Requirements: Minimum {self._format_number(min_followers)} followers, "
            f"{min_engagement*100:.1f}% engagement rate"
        )
        
        # Target audience
        target_age = campaign.get('target_age_group', '')
        target_gender = campaign.get('target_gender', 'All')
        
        if target_age or target_gender != 'All':
            text_parts.append(f"Target audience: {target_gender}, ages {target_age}")
        
        # Campaign goals (if available)
        goals = campaign.get('campaign_goals', '')
        if goals:
            text_parts.append(f"Goals: {goals}")
        
        # Combine all parts
        full_text = " | ".join(text_parts)
        
        # Encode to embedding
        embedding = self.model.encode(full_text, convert_to_tensor=False)
        
        return embedding
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
        
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        # Ensure 2D arrays for sklearn
        emb1 = embedding1.reshape(1, -1)
        emb2 = embedding2.reshape(1, -1)
        
        similarity = cosine_similarity(emb1, emb2)[0][0]
        
        # Convert from [-1, 1] to [0, 1] range
        similarity = (similarity + 1) / 2
        
        return float(similarity)
    
    def match_creator_to_campaign(self, creator: Dict, campaign: Dict, 
                                  brand: Optional[Dict] = None) -> float:
        """
        Compute semantic match score between creator and campaign
        
        Args:
            creator: Creator profile
            campaign: Campaign details
            brand: Optional brand information
        
        Returns:
            Match score (0-1)
        """
        creator_emb = self.encode_creator_profile(creator)
        campaign_emb = self.encode_campaign(campaign, brand)
        
        return self.compute_similarity(creator_emb, campaign_emb)
    
    def find_top_matches(self, campaign: Dict, creators: List[Dict], 
                        brand: Optional[Dict] = None, top_k: int = 20) -> List[Tuple[int, float]]:
        """
        Find top K semantically similar creators for a campaign
        
        Args:
            campaign: Campaign details
            creators: List of creator profiles
            brand: Optional brand information
            top_k: Number of top matches to return
        
        Returns:
            List of (creator_id, similarity_score) tuples, sorted by score
        """
        # Encode campaign once
        campaign_emb = self.encode_campaign(campaign, brand)
        
        # Compute similarities for all creators
        scores = []
        for creator in creators:
            creator_emb = self.encode_creator_profile(creator)
            similarity = self.compute_similarity(creator_emb, campaign_emb)
            scores.append((creator['creator_id'], similarity))
        
        # Sort by similarity (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]
    
    def batch_encode_creators(self, creators: List[Dict]) -> np.ndarray:
        """
        Efficiently encode multiple creators
        
        Args:
            creators: List of creator profiles
        
        Returns:
            (num_creators, embedding_dim) array of embeddings
        """
        texts = []
        for creator in creators:
            # Create text representation (simplified for batch processing)
            categories = creator.get('categories', [])
            if isinstance(categories, str):
                categories = json.loads(categories)
            
            bio = creator.get('bio', '')
            followers = creator.get('followers', 0)
            
            text = f"{bio} | {', '.join(categories)} | {self._format_number(followers)} followers"
            texts.append(text)
        
        # Batch encode
        embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
        
        return embeddings
    
    def batch_encode_campaigns(self, campaigns: List[Dict]) -> np.ndarray:
        """
        Efficiently encode multiple campaigns
        
        Args:
            campaigns: List of campaign details
        
        Returns:
            (num_campaigns, embedding_dim) array of embeddings
        """
        texts = []
        for campaign in campaigns:
            title = campaign.get('title', '')
            description = campaign.get('description', '')
            category = campaign.get('category', '')
            
            text = f"{title} | {description} | {category}"
            texts.append(text)
        
        # Batch encode
        embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
        
        return embeddings
    
    @staticmethod
    def _format_number(num: float) -> str:
        """Format large numbers with K/M suffix"""
        if num >= 1_000_000:
            return f"{num/1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num/1_000:.1f}K"
        else:
            return str(int(num))


class HybridSemanticMatcher(SemanticMatcher):
    """
    Enhanced semantic matcher that combines multiple signals:
    1. Text similarity (BERT embeddings)
    2. Category overlap
    3. Demographic alignment
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: Optional[str] = None):
        super().__init__(model_name, device)
        
        # Weights for hybrid scoring
        self.text_weight = 0.6
        self.category_weight = 0.25
        self.demographic_weight = 0.15
    
    def hybrid_match_score(self, creator: Dict, campaign: Dict, 
                          brand: Optional[Dict] = None) -> Dict[str, float]:
        """
        Compute hybrid match score combining multiple signals
        
        Returns:
            Dictionary with overall score and component scores
        """
        # 1. Semantic similarity
        semantic_score = self.match_creator_to_campaign(creator, campaign, brand)
        
        # 2. Category overlap
        creator_categories = creator.get('categories', [])
        if isinstance(creator_categories, str):
            # Try JSON first, fall back to comma-separated
            try:
                creator_categories = json.loads(creator_categories) if creator_categories.strip() else []
            except (json.JSONDecodeError, ValueError):
                creator_categories = [c.strip() for c in creator_categories.split(',') if c.strip()]
        creator_categories_set = set(creator_categories)
        
        campaign_category = campaign.get('category', '')
        category_score = 1.0 if campaign_category in creator_categories_set else 0.0
        
        # 3. Demographic alignment
        # Age alignment
        target_age = campaign.get('target_age_group', '')
        creator_age_18_24 = creator.get('audience_age_18_24', 0) / 100
        creator_age_25_34 = creator.get('audience_age_25_34', 0) / 100
        
        if '18-24' in target_age:
            age_score = creator_age_18_24
        elif '25-34' in target_age:
            age_score = creator_age_25_34
        else:
            age_score = 0.5  # Neutral
        
        # Gender alignment
        target_gender = campaign.get('target_gender', 'All')
        creator_female_pct = creator.get('audience_female_pct', 50) / 100
        
        if target_gender == 'Female':
            gender_score = creator_female_pct
        elif target_gender == 'Male':
            gender_score = 1 - creator_female_pct
        else:
            gender_score = 1.0  # All genders
        
        demographic_score = (age_score + gender_score) / 2
        
        # Combine scores
        overall_score = (
            self.text_weight * semantic_score +
            self.category_weight * category_score +
            self.demographic_weight * demographic_score
        )
        
        return {
            'overall_score': overall_score,
            'semantic_score': semantic_score,
            'category_score': category_score,
            'demographic_score': demographic_score
        }


if __name__ == '__main__':
    # Example usage
    matcher = SemanticMatcher()
    
    sample_creator = {
        'creator_id': 1,
        'bio': 'Fashion and lifestyle content creator passionate about sustainable fashion',
        'categories': json.dumps(['Fashion', 'Lifestyle', 'Sustainability']),
        'platforms': json.dumps(['Instagram', 'TikTok']),
        'followers': 75000,
        'engagement_rate': 0.05,
        'tier': 'micro',
        'total_campaigns': 30,
        'success_rate': 0.85,
        'overall_rating': 4.7,
        'audience_age_18_24': 45,
        'audience_age_25_34': 35,
        'audience_female_pct': 75
    }
    
    sample_campaign = {
        'title': 'Summer Fashion Collection Launch',
        'description': 'Promote our new sustainable summer clothing line',
        'category': 'Fashion',
        'platform': 'Instagram',
        'industry': 'Fashion',
        'budget': 3000,
        'duration_days': 30,
        'deliverables': json.dumps(['Post', 'Story', 'Reel']),
        'min_followers': 50000,
        'min_engagement': 0.03,
        'target_age_group': '18-24',
        'target_gender': 'Female'
    }
    
    # Compute match score
    score = matcher.match_creator_to_campaign(sample_creator, sample_campaign)
    print(f"Semantic Match Score: {score:.4f}")
    
    # Try hybrid matcher
    hybrid_matcher = HybridSemanticMatcher()
    hybrid_scores = hybrid_matcher.hybrid_match_score(sample_creator, sample_campaign)
    
    print("\nHybrid Match Scores:")
    for key, value in hybrid_scores.items():
        print(f"  {key}: {value:.4f}")
