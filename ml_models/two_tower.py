"""
Two-Tower Deep Neural Network for Creator-Campaign Matching
Based on Google's research for large-scale recommendation systems

Architecture:
- Creator Tower: Encodes creator features into dense embeddings
- Campaign Tower: Encodes campaign features into dense embeddings
- Interaction Layer: Combines embeddings via dot product + MLP
- Multi-task Learning: Predicts match score, success probability, and ROI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List


class CreatorTower(nn.Module):
    """Encodes creator features into dense embedding"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        
        # Categorical embeddings
        self.category_emb = nn.Embedding(
            config.get('num_categories', 20),
            config.get('category_dim', 32)
        )
        self.platform_emb = nn.Embedding(
            config.get('num_platforms', 6),
            config.get('platform_dim', 16)
        )
        self.tier_emb = nn.Embedding(
            config.get('num_tiers', 5),
            config.get('tier_dim', 16)
        )
        
        # Calculate input dimension for FC layers
        # Embeddings: category (32) + platform (16) + tier (16) = 64
        # Numeric features: ~30 from FeatureEngineer
        embedding_dim = (
            config.get('category_dim', 32) +
            config.get('platform_dim', 16) +
            config.get('tier_dim', 16)
        )
        numeric_dim = config.get('creator_numeric_features', 30)
        total_input_dim = embedding_dim + numeric_dim
        
        # Batch normalization for numeric features
        self.batch_norm = nn.BatchNorm1d(numeric_dim)
        
        # Dense layers with residual connections
        hidden_dim = config.get('hidden_dim', 512)
        embedding_output_dim = config.get('embedding_dim', 128)
        
        self.fc1 = nn.Linear(total_input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(config.get('dropout', 0.3))
        
        self.fc2 = nn.Linear(hidden_dim, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(256, embedding_output_dim)
        
    def forward(self, numeric_features: torch.Tensor, 
                category_idx: torch.Tensor, 
                platform_idx: torch.Tensor,
                tier_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            numeric_features: (batch_size, numeric_dim) - normalized numeric features
            category_idx: (batch_size,) - category index
            platform_idx: (batch_size,) - platform index
            tier_idx: (batch_size,) - tier index
        
        Returns:
            (batch_size, embedding_dim) - L2 normalized embedding
        """
        # Embed categorical features
        cat_emb = self.category_emb(category_idx)  # (batch, category_dim)
        plat_emb = self.platform_emb(platform_idx)  # (batch, platform_dim)
        tier_emb = self.tier_emb(tier_idx)  # (batch, tier_dim)
        
        # Normalize numeric features
        numeric_norm = self.batch_norm(numeric_features)  # (batch, numeric_dim)
        
        # Concatenate all features
        combined = torch.cat([cat_emb, plat_emb, tier_emb, numeric_norm], dim=1)
        
        # Dense layers with activations
        x = self.fc1(combined)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        embedding = self.fc3(x)
        
        # L2 normalize for cosine similarity
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding


class CampaignTower(nn.Module):
    """Encodes campaign features into dense embedding"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        
        # Categorical embeddings
        self.category_emb = nn.Embedding(
            config.get('num_categories', 20),
            config.get('category_dim', 32)
        )
        self.platform_emb = nn.Embedding(
            config.get('num_platforms', 6),
            config.get('platform_dim', 16)
        )
        self.industry_emb = nn.Embedding(
            config.get('num_industries', 15),
            config.get('industry_dim', 16)
        )
        
        # Calculate input dimension
        embedding_dim = (
            config.get('category_dim', 32) +
            config.get('platform_dim', 16) +
            config.get('industry_dim', 16)
        )
        numeric_dim = config.get('campaign_numeric_features', 15)
        total_input_dim = embedding_dim + numeric_dim
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(numeric_dim)
        
        # Dense layers
        hidden_dim = config.get('hidden_dim', 512)
        embedding_output_dim = config.get('embedding_dim', 128)
        
        self.fc1 = nn.Linear(total_input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(config.get('dropout', 0.3))
        
        self.fc2 = nn.Linear(hidden_dim, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(256, embedding_output_dim)
        
    def forward(self, numeric_features: torch.Tensor,
                category_idx: torch.Tensor,
                platform_idx: torch.Tensor,
                industry_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            numeric_features: (batch_size, numeric_dim)
            category_idx: (batch_size,)
            platform_idx: (batch_size,)
            industry_idx: (batch_size,)
        
        Returns:
            (batch_size, embedding_dim) - L2 normalized embedding
        """
        # Embed categorical features
        cat_emb = self.category_emb(category_idx)
        plat_emb = self.platform_emb(platform_idx)
        ind_emb = self.industry_emb(industry_idx)
        
        # Normalize numeric
        numeric_norm = self.batch_norm(numeric_features)
        
        # Concatenate
        combined = torch.cat([cat_emb, plat_emb, ind_emb, numeric_norm], dim=1)
        
        # Dense layers
        x = self.fc1(combined)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        embedding = self.fc3(x)
        
        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding


class TwoTowerMatcher(nn.Module):
    """
    Complete two-tower model for creator-campaign matching
    
    Outputs:
    - Match score (0-1): Probability of successful match
    - Success probability (0-1): Probability campaign succeeds
    - Estimated ROI (regression): Expected return on investment
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.creator_tower = CreatorTower(config)
        self.campaign_tower = CampaignTower(config)
        
        embedding_dim = config.get('embedding_dim', 128)
        
        # Interaction layer (dot product + MLP)
        # Input: concatenated embeddings (128*2) + dot product (1) = 257
        interaction_input_dim = embedding_dim * 2 + 1
        
        self.interaction_mlp = nn.Sequential(
            nn.Linear(interaction_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        
        # Multi-task heads
        self.match_score_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.success_prob_head = nn.Sequential(
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.roi_head = nn.Linear(32, 1)  # Regression (no activation)
        
    def forward(self, 
                creator_numeric: torch.Tensor,
                creator_category: torch.Tensor,
                creator_platform: torch.Tensor,
                creator_tier: torch.Tensor,
                campaign_numeric: torch.Tensor,
                campaign_category: torch.Tensor,
                campaign_platform: torch.Tensor,
                campaign_industry: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through both towers and interaction layer
        
        Returns:
            Dictionary with predictions and embeddings
        """
        # Get embeddings from both towers
        creator_emb = self.creator_tower(
            creator_numeric, creator_category, creator_platform, creator_tier
        )
        campaign_emb = self.campaign_tower(
            campaign_numeric, campaign_category, campaign_platform, campaign_industry
        )
        
        # Compute dot product (cosine similarity since L2 normalized)
        dot_product = (creator_emb * campaign_emb).sum(dim=1, keepdim=True)
        
        # Concatenate for interaction layer
        interaction_input = torch.cat([creator_emb, campaign_emb, dot_product], dim=1)
        
        # Interaction features
        interaction_features = self.interaction_mlp(interaction_input)
        
        # Multi-task predictions
        match_score = self.match_score_head(interaction_features)
        success_prob = self.success_prob_head(interaction_features)
        estimated_roi = self.roi_head(interaction_features)
        
        return {
            'match_score': match_score.squeeze(-1),
            'success_probability': success_prob.squeeze(-1),
            'estimated_roi': estimated_roi.squeeze(-1),
            'creator_embedding': creator_emb,
            'campaign_embedding': campaign_emb,
            'dot_product': dot_product.squeeze(-1)
        }
    
    def get_creator_embedding(self,
                             creator_numeric: torch.Tensor,
                             creator_category: torch.Tensor,
                             creator_platform: torch.Tensor,
                             creator_tier: torch.Tensor) -> torch.Tensor:
        """Get creator embedding only (for retrieval)"""
        with torch.no_grad():
            return self.creator_tower(
                creator_numeric, creator_category, creator_platform, creator_tier
            )
    
    def get_campaign_embedding(self,
                              campaign_numeric: torch.Tensor,
                              campaign_category: torch.Tensor,
                              campaign_platform: torch.Tensor,
                              campaign_industry: torch.Tensor) -> torch.Tensor:
        """Get campaign embedding only (for retrieval)"""
        with torch.no_grad():
            return self.campaign_tower(
                campaign_numeric, campaign_category, campaign_platform, campaign_industry
            )


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function with weighted combination
    Supports uncertainty weighting (learns optimal task weights)
    """
    
    def __init__(self, learn_weights: bool = False):
        super().__init__()
        self.learn_weights = learn_weights
        
        if learn_weights:
            # Learnable uncertainty parameters (log variance)
            self.log_var_match = nn.Parameter(torch.zeros(1))
            self.log_var_success = nn.Parameter(torch.zeros(1))
            self.log_var_roi = nn.Parameter(torch.zeros(1))
        else:
            # Fixed weights
            self.weight_match = 1.0
            self.weight_success = 1.0
            self.weight_roi = 0.5
        
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss
        
        Args:
            predictions: Dict with 'match_score', 'success_probability', 'estimated_roi'
            targets: Dict with 'match_label', 'success_label', 'roi_value'
        
        Returns:
            (total_loss, individual_losses)
        """
        # Individual losses
        loss_match = self.bce(predictions['match_score'], targets['match_label'])
        loss_success = self.bce(predictions['success_probability'], targets['success_label'])
        loss_roi = self.mse(predictions['estimated_roi'], targets['roi_value'])
        
        if self.learn_weights:
            # Uncertainty-weighted multi-task loss
            # Based on "Multi-Task Learning Using Uncertainty to Weigh Losses"
            total_loss = (
                torch.exp(-self.log_var_match) * loss_match + self.log_var_match +
                torch.exp(-self.log_var_success) * loss_success + self.log_var_success +
                torch.exp(-self.log_var_roi) * loss_roi + self.log_var_roi
            )
        else:
            # Fixed weighted sum
            total_loss = (
                self.weight_match * loss_match +
                self.weight_success * loss_success +
                self.weight_roi * loss_roi
            )
        
        individual_losses = {
            'match_loss': loss_match.item(),
            'success_loss': loss_success.item(),
            'roi_loss': loss_roi.item(),
            'total_loss': total_loss.item()
        }
        
        if self.learn_weights:
            individual_losses['weight_match'] = torch.exp(-self.log_var_match).item()
            individual_losses['weight_success'] = torch.exp(-self.log_var_success).item()
            individual_losses['weight_roi'] = torch.exp(-self.log_var_roi).item()
        
        return total_loss, individual_losses


def create_default_config() -> Dict:
    """Create default model configuration"""
    return {
        # Vocabulary sizes
        'num_categories': 20,
        'num_platforms': 6,
        'num_tiers': 5,
        'num_industries': 15,
        
        # Embedding dimensions
        'category_dim': 32,
        'platform_dim': 16,
        'tier_dim': 16,
        'industry_dim': 16,
        'embedding_dim': 128,  # Final embedding size
        
        # Network architecture
        'hidden_dim': 512,
        'dropout': 0.3,
        
        # Feature dimensions (from FeatureEngineer)
        'creator_numeric_features': 30,
        'campaign_numeric_features': 15,
    }


if __name__ == '__main__':
    # Test model
    config = create_default_config()
    model = TwoTowerMatcher(config)
    
    # Sample batch
    batch_size = 32
    
    # Creator features
    creator_numeric = torch.randn(batch_size, 30)
    creator_category = torch.randint(0, 20, (batch_size,))
    creator_platform = torch.randint(0, 6, (batch_size,))
    creator_tier = torch.randint(0, 5, (batch_size,))
    
    # Campaign features
    campaign_numeric = torch.randn(batch_size, 15)
    campaign_category = torch.randint(0, 20, (batch_size,))
    campaign_platform = torch.randint(0, 6, (batch_size,))
    campaign_industry = torch.randint(0, 15, (batch_size,))
    
    # Forward pass
    outputs = model(
        creator_numeric, creator_category, creator_platform, creator_tier,
        campaign_numeric, campaign_category, campaign_platform, campaign_industry
    )
    
    print("Model output shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
