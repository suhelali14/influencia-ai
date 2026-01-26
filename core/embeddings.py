"""
Embedding Service - Two-Tower Architecture

This module implements a proper two-tower model for candidate generation:
- Creator Tower: Encodes creator profiles into dense embeddings
- Campaign Tower: Encodes campaign requirements into dense embeddings

Benefits:
1. Embeddings can be pre-computed and cached
2. Approximate Nearest Neighbor (ANN) search for fast retrieval
3. Semantic understanding of content, not just keyword matching
4. Scales to millions of creators with sub-millisecond retrieval

Reference implementations:
- YouTube DNN: https://dl.acm.org/doi/10.1145/2959100.2959190
- Instagram Explore: https://ai.facebook.com/blog/powered-by-ai-instagrams-explore-recommender-system/
- Alibaba i2i: https://arxiv.org/abs/1803.02349
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import json
import os

from .entities import Creator, Campaign

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding service"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    batch_size: int = 32
    cache_dir: str = "embeddings_cache"
    use_gpu: bool = False


class EmbeddingService:
    """
    Two-Tower embedding service for creator-campaign matching.
    
    The key insight is that creator and campaign embeddings live in the
    same vector space, so cosine similarity between them indicates match quality.
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._model = None
        self._faiss_index = None
        self._creator_id_map = {}  # Maps faiss index -> creator_id
        self._embeddings_cache = {}
        
    def _load_model(self):
        """Lazy load the embedding model"""
        if self._model is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.config.model_name}")
            self._model = SentenceTransformer(self.config.model_name)
            
            if self.config.use_gpu:
                self._model = self._model.cuda()
            
            logger.info("Embedding model loaded successfully")
        except ImportError:
            logger.warning("sentence-transformers not installed, using fallback")
            self._model = None
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self._model = None
    
    def encode_creator(self, creator: Creator) -> np.ndarray:
        """
        Encode creator profile into embedding vector.
        
        The text representation is carefully crafted to capture:
        - Primary content categories
        - Platform and audience size
        - Performance metrics
        - Content style signals
        """
        cache_key = f"creator_{creator.id}"
        if cache_key in self._embeddings_cache:
            return self._embeddings_cache[cache_key]
        
        # Build rich text representation
        text = self._build_creator_text(creator)
        
        # Get embedding
        embedding = self._encode_text(text)
        
        # Enhance with structured features
        enhanced = self._enhance_creator_embedding(embedding, creator)
        
        self._embeddings_cache[cache_key] = enhanced
        return enhanced
    
    def encode_campaign(self, campaign: Campaign) -> np.ndarray:
        """
        Encode campaign requirements into embedding vector.
        
        The text representation captures:
        - Campaign goals and requirements
        - Target audience characteristics
        - Brand context
        """
        cache_key = f"campaign_{campaign.id}"
        if cache_key in self._embeddings_cache:
            return self._embeddings_cache[cache_key]
        
        # Build rich text representation
        text = self._build_campaign_text(campaign)
        
        # Get embedding
        embedding = self._encode_text(text)
        
        # Enhance with structured features
        enhanced = self._enhance_campaign_embedding(embedding, campaign)
        
        self._embeddings_cache[cache_key] = enhanced
        return enhanced
    
    def _build_creator_text(self, creator: Creator) -> str:
        """Build text representation of creator for embedding"""
        parts = []
        
        # Categories (most important for semantic matching)
        if creator.categories:
            categories_str = ", ".join(creator.categories)
            parts.append(f"Content creator specializing in {categories_str}")
        
        # Platform and reach
        tier_descriptions = {
            'nano': 'emerging',
            'micro': 'growing',
            'mid': 'established',
            'macro': 'popular',
            'mega': 'celebrity'
        }
        tier_desc = tier_descriptions.get(creator.tier, 'active')
        parts.append(f"{tier_desc.capitalize()} {creator.platform} creator with {creator.followers:,} followers")
        
        # Engagement and quality signals
        if creator.engagement_rate > 0.05:
            parts.append("High engagement rate with active audience")
        elif creator.engagement_rate > 0.02:
            parts.append("Good engagement with responsive followers")
        
        if creator.content_quality_score > 0.8:
            parts.append("Premium quality content production")
        
        if creator.audience_authenticity > 0.9:
            parts.append("Authentic organic audience")
        
        # Experience
        if creator.total_campaigns > 20:
            parts.append("Experienced in brand collaborations")
        elif creator.total_campaigns > 5:
            parts.append("Proven track record with brands")
        
        # Performance
        if creator.avg_campaign_rating >= 4.5:
            parts.append("Highly rated by previous partners")
        
        # Location/Language
        if creator.location:
            parts.append(f"Based in {creator.location}")
        if creator.language:
            parts.append(f"Content in {creator.language}")
        
        return ". ".join(parts) + "."
    
    def _build_campaign_text(self, campaign: Campaign) -> str:
        """Build text representation of campaign for embedding"""
        parts = []
        
        # Campaign basics
        parts.append(f"{campaign.brand_name} brand campaign")
        parts.append(campaign.title)
        
        # Target categories
        if campaign.categories:
            categories_str = ", ".join(campaign.categories)
            parts.append(f"Looking for creators in {categories_str}")
        
        # Platform
        parts.append(f"For {campaign.platform} platform")
        
        # Requirements
        if campaign.description:
            parts.append(campaign.description[:200])  # Truncate long descriptions
        
        # Target audience
        tier_prefs = campaign.preferred_tiers
        if tier_prefs:
            parts.append(f"Seeking {', '.join(tier_prefs)} tier influencers")
        
        if campaign.min_followers > 0:
            parts.append(f"Minimum {campaign.min_followers:,} followers required")
        
        if campaign.target_engagement_rate > 0:
            parts.append(f"Target engagement rate: {campaign.target_engagement_rate:.1%}")
        
        # Location
        if campaign.target_locations:
            parts.append(f"Target markets: {', '.join(campaign.target_locations)}")
        
        # Language
        if campaign.target_languages:
            parts.append(f"Content in {', '.join(campaign.target_languages)}")
        
        return ". ".join(parts) + "."
    
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding using sentence transformer"""
        self._load_model()
        
        if self._model is None:
            # Fallback: simple bag-of-words style embedding
            return self._fallback_embedding(text)
        
        try:
            embedding = self._model.encode(
                text, 
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Encoding failed: {e}")
            return self._fallback_embedding(text)
    
    def _fallback_embedding(self, text: str) -> np.ndarray:
        """
        Fallback embedding when model isn't available.
        Uses simple hashing trick for consistent embeddings.
        """
        np.random.seed(hash(text) % 2**32)
        embedding = np.random.randn(self.config.embedding_dim).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def _enhance_creator_embedding(
        self, 
        text_embedding: np.ndarray, 
        creator: Creator
    ) -> np.ndarray:
        """
        Enhance text embedding with structured features.
        
        This hybrid approach combines semantic understanding from
        text with explicit structured features.
        """
        # Create structured feature vector
        structured = np.array([
            np.log10(max(1, creator.followers)) / 8,  # Normalize to ~0-1
            creator.engagement_rate * 10,  # Scale up
            creator.content_quality_score,
            creator.audience_authenticity,
            creator.completion_rate,
            creator.avg_campaign_rating / 5.0,
            min(creator.total_campaigns / 50, 1.0),
            creator.growth_rate_30d + 0.5,  # Shift to positive
        ], dtype=np.float32)
        
        # Concatenate (or could use a learned projection)
        # For simplicity, we weight the text embedding higher
        enhanced = np.concatenate([
            text_embedding * 0.85,
            structured * 0.15 / len(structured) * len(text_embedding)
        ])
        
        # Re-normalize
        enhanced = enhanced / np.linalg.norm(enhanced)
        return enhanced
    
    def _enhance_campaign_embedding(
        self, 
        text_embedding: np.ndarray, 
        campaign: Campaign
    ) -> np.ndarray:
        """Enhance text embedding with campaign structured features"""
        # Create structured feature vector
        structured = np.array([
            np.log10(max(1, campaign.budget)) / 6,
            np.log10(max(1, campaign.min_followers)) / 8,
            np.log10(max(1, campaign.max_followers or 10_000_000)) / 8,
            campaign.target_engagement_rate * 10,
            campaign.brand_avg_rating / 5.0,
            campaign.brand_payment_reliability,
            min(campaign.brand_total_campaigns / 50, 1.0),
            0.5,  # Placeholder
        ], dtype=np.float32)
        
        # Match the enhancement from creators
        enhanced = np.concatenate([
            text_embedding * 0.85,
            structured * 0.15 / len(structured) * len(text_embedding)
        ])
        
        enhanced = enhanced / np.linalg.norm(enhanced)
        return enhanced
    
    def compute_similarity(
        self, 
        creator_embedding: np.ndarray, 
        campaign_embedding: np.ndarray
    ) -> float:
        """Compute cosine similarity between embeddings"""
        return float(np.dot(creator_embedding, campaign_embedding))
    
    def batch_encode_creators(
        self, 
        creators: List[Creator]
    ) -> Dict[str, np.ndarray]:
        """Encode multiple creators efficiently"""
        results = {}
        
        # Batch encode texts for efficiency
        texts = [self._build_creator_text(c) for c in creators]
        
        self._load_model()
        
        if self._model is not None:
            try:
                embeddings = self._model.encode(
                    texts,
                    batch_size=self.config.batch_size,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                
                for i, creator in enumerate(creators):
                    enhanced = self._enhance_creator_embedding(
                        embeddings[i], creator
                    )
                    results[creator.id] = enhanced
                    self._embeddings_cache[f"creator_{creator.id}"] = enhanced
                
            except Exception as e:
                logger.error(f"Batch encoding failed: {e}")
                # Fall back to individual encoding
                for creator in creators:
                    results[creator.id] = self.encode_creator(creator)
        else:
            for creator in creators:
                results[creator.id] = self.encode_creator(creator)
        
        return results
    
    def build_faiss_index(self, creators: List[Creator]) -> None:
        """
        Build FAISS index for fast approximate nearest neighbor search.
        
        This enables sub-millisecond candidate retrieval from millions of creators.
        """
        try:
            import faiss
        except ImportError:
            logger.warning("FAISS not installed, ANN search unavailable")
            return
        
        logger.info(f"Building FAISS index for {len(creators)} creators")
        
        # Encode all creators
        embeddings = self.batch_encode_creators(creators)
        
        # Build matrix
        embedding_dim = next(iter(embeddings.values())).shape[0]
        matrix = np.zeros((len(creators), embedding_dim), dtype=np.float32)
        
        for i, creator in enumerate(creators):
            matrix[i] = embeddings[creator.id]
            self._creator_id_map[i] = creator.id
        
        # Create index (using Inner Product for cosine similarity on normalized vectors)
        self._faiss_index = faiss.IndexFlatIP(embedding_dim)
        self._faiss_index.add(matrix)
        
        logger.info(f"FAISS index built with {self._faiss_index.ntotal} vectors")
    
    def search_similar_creators(
        self, 
        campaign: Campaign, 
        k: int = 100
    ) -> List[Tuple[str, float]]:
        """
        Find top-k most similar creators for a campaign using ANN.
        
        Returns list of (creator_id, similarity_score) tuples.
        """
        if self._faiss_index is None:
            logger.warning("FAISS index not built")
            return []
        
        # Encode campaign
        campaign_embedding = self.encode_campaign(campaign)
        
        # Search
        query = campaign_embedding.reshape(1, -1).astype(np.float32)
        scores, indices = self._faiss_index.search(query, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx >= 0:  # Valid index
                creator_id = self._creator_id_map.get(int(idx))
                if creator_id:
                    results.append((creator_id, float(scores[0][i])))
        
        return results
    
    def save_embeddings(self, path: str) -> None:
        """Save embeddings cache to disk"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable = {
            k: v.tolist() for k, v in self._embeddings_cache.items()
        }
        
        with open(path, 'w') as f:
            json.dump(serializable, f)
        
        logger.info(f"Saved {len(serializable)} embeddings to {path}")
    
    def load_embeddings(self, path: str) -> None:
        """Load embeddings cache from disk"""
        if not os.path.exists(path):
            logger.warning(f"Embeddings file not found: {path}")
            return
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        self._embeddings_cache = {
            k: np.array(v, dtype=np.float32) for k, v in data.items()
        }
        
        logger.info(f"Loaded {len(self._embeddings_cache)} embeddings from {path}")
    
    def clear_cache(self) -> None:
        """Clear embeddings cache"""
        self._embeddings_cache.clear()
        logger.info("Embeddings cache cleared")
