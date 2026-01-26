"""
Neural Collaborative Filtering (NCF)
Based on "Neural Collaborative Filtering" (He et al., 2017)

Combines:
- Generalized Matrix Factorization (GMF): Element-wise product of embeddings
- Multi-Layer Perceptron (MLP): Deep interaction learning
- Fusion Layer: Combines both paths for final prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class NeuralCollaborativeFiltering(nn.Module):
    """
    Neural Collaborative Filtering for creator-campaign matching
    
    Learns latent factors for creators and campaigns from historical interactions
    Predicts match probability based on learned embeddings
    """
    
    def __init__(self, num_creators: int, num_campaigns: int, 
                 embedding_dim: int = 64, mlp_layers: list = [128, 64, 32]):
        super().__init__()
        
        self.num_creators = num_creators
        self.num_campaigns = num_campaigns
        self.embedding_dim = embedding_dim
        
        # === GMF Path (Matrix Factorization) ===
        self.creator_gmf_embedding = nn.Embedding(num_creators, embedding_dim)
        self.campaign_gmf_embedding = nn.Embedding(num_campaigns, embedding_dim)
        
        # === MLP Path ===
        self.creator_mlp_embedding = nn.Embedding(num_creators, embedding_dim)
        self.campaign_mlp_embedding = nn.Embedding(num_campaigns, embedding_dim)
        
        # MLP layers
        mlp_input_dim = embedding_dim * 2
        self.mlp_layers = nn.ModuleList()
        
        prev_dim = mlp_input_dim
        for layer_dim in mlp_layers:
            self.mlp_layers.append(nn.Linear(prev_dim, layer_dim))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(0.2))
            prev_dim = layer_dim
        
        self.mlp_output_dim = mlp_layers[-1]
        
        # === Fusion Layer ===
        # Combine GMF output (embedding_dim) and MLP output (mlp_layers[-1])
        fusion_input_dim = embedding_dim + self.mlp_output_dim
        self.fusion = nn.Linear(fusion_input_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings with Xavier initialization"""
        nn.init.xavier_uniform_(self.creator_gmf_embedding.weight)
        nn.init.xavier_uniform_(self.campaign_gmf_embedding.weight)
        nn.init.xavier_uniform_(self.creator_mlp_embedding.weight)
        nn.init.xavier_uniform_(self.campaign_mlp_embedding.weight)
        
        # Initialize linear layers
        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.fusion.weight)
        nn.init.zeros_(self.fusion.bias)
    
    def forward(self, creator_ids: torch.Tensor, campaign_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            creator_ids: (batch_size,) - Creator ID tensor
            campaign_ids: (batch_size,) - Campaign ID tensor
        
        Returns:
            (batch_size,) - Predicted match probability (0-1)
        """
        # === GMF Path ===
        creator_gmf = self.creator_gmf_embedding(creator_ids)  # (batch, embedding_dim)
        campaign_gmf = self.campaign_gmf_embedding(campaign_ids)  # (batch, embedding_dim)
        
        # Element-wise product (Hadamard product)
        gmf_output = creator_gmf * campaign_gmf  # (batch, embedding_dim)
        
        # === MLP Path ===
        creator_mlp = self.creator_mlp_embedding(creator_ids)  # (batch, embedding_dim)
        campaign_mlp = self.campaign_mlp_embedding(campaign_ids)  # (batch, embedding_dim)
        
        # Concatenate creator and campaign embeddings
        mlp_input = torch.cat([creator_mlp, campaign_mlp], dim=1)  # (batch, embedding_dim*2)
        
        # Pass through MLP layers
        mlp_output = mlp_input
        for layer in self.mlp_layers:
            mlp_output = layer(mlp_output)
        
        # === Fusion Layer ===
        # Concatenate GMF and MLP outputs
        fusion_input = torch.cat([gmf_output, mlp_output], dim=1)
        
        # Final prediction
        prediction = torch.sigmoid(self.fusion(fusion_input))
        
        return prediction.squeeze(-1)
    
    def get_creator_embedding(self, creator_ids: torch.Tensor, mode: str = 'gmf') -> torch.Tensor:
        """
        Get creator embedding for retrieval
        
        Args:
            creator_ids: Creator IDs
            mode: 'gmf', 'mlp', or 'both'
        
        Returns:
            Creator embeddings
        """
        with torch.no_grad():
            if mode == 'gmf':
                return self.creator_gmf_embedding(creator_ids)
            elif mode == 'mlp':
                return self.creator_mlp_embedding(creator_ids)
            else:  # both
                gmf = self.creator_gmf_embedding(creator_ids)
                mlp = self.creator_mlp_embedding(creator_ids)
                return torch.cat([gmf, mlp], dim=1)
    
    def get_campaign_embedding(self, campaign_ids: torch.Tensor, mode: str = 'gmf') -> torch.Tensor:
        """
        Get campaign embedding for retrieval
        
        Args:
            campaign_ids: Campaign IDs
            mode: 'gmf', 'mlp', or 'both'
        
        Returns:
            Campaign embeddings
        """
        with torch.no_grad():
            if mode == 'gmf':
                return self.campaign_gmf_embedding(campaign_ids)
            elif mode == 'mlp':
                return self.campaign_mlp_embedding(campaign_ids)
            else:  # both
                gmf = self.campaign_gmf_embedding(campaign_ids)
                mlp = self.campaign_mlp_embedding(campaign_ids)
                return torch.cat([gmf, mlp], dim=1)


class NCFTrainer:
    """Training utilities for NCF model with negative sampling"""
    
    def __init__(self, model: NeuralCollaborativeFiltering, 
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5):
        self.model = model
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.BCELoss()
        
    def train_step(self, positive_samples: Dict[str, torch.Tensor], 
                   negative_samples: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Train for one step with positive and negative samples
        
        Args:
            positive_samples: Dict with 'creator_ids' and 'campaign_ids' (actual matches)
            negative_samples: Dict with 'creator_ids' and 'campaign_ids' (random non-matches)
        
        Returns:
            Dictionary with loss values
        """
        self.model.train()
        
        # Positive samples (label = 1)
        creator_ids_pos = positive_samples['creator_ids']
        campaign_ids_pos = positive_samples['campaign_ids']
        predictions_pos = self.model(creator_ids_pos, campaign_ids_pos)
        loss_pos = self.criterion(predictions_pos, torch.ones_like(predictions_pos))
        
        # Negative samples (label = 0)
        creator_ids_neg = negative_samples['creator_ids']
        campaign_ids_neg = negative_samples['campaign_ids']
        predictions_neg = self.model(creator_ids_neg, campaign_ids_neg)
        loss_neg = self.criterion(predictions_neg, torch.zeros_like(predictions_neg))
        
        # Total loss
        loss = loss_pos + loss_neg
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': loss.item(),
            'pos_loss': loss_pos.item(),
            'neg_loss': loss_neg.item(),
            'avg_pos_pred': predictions_pos.mean().item(),
            'avg_neg_pred': predictions_neg.mean().item()
        }
    
    @staticmethod
    def generate_negative_samples(positive_creator_ids: torch.Tensor,
                                 positive_campaign_ids: torch.Tensor,
                                 num_creators: int,
                                 num_campaigns: int,
                                 num_negatives: int = 4) -> Dict[str, torch.Tensor]:
        """
        Generate negative samples for each positive sample
        
        Args:
            positive_creator_ids: Tensor of creator IDs from positive samples
            positive_campaign_ids: Tensor of campaign IDs from positive samples
            num_creators: Total number of creators
            num_campaigns: Total number of campaigns
            num_negatives: Number of negative samples per positive
        
        Returns:
            Dictionary with negative creator_ids and campaign_ids
        """
        batch_size = len(positive_creator_ids)
        
        # For each positive sample, generate N negative samples
        neg_creator_ids = []
        neg_campaign_ids = []
        
        for i in range(batch_size):
            creator_id = positive_creator_ids[i].item()
            campaign_id = positive_campaign_ids[i].item()
            
            for _ in range(num_negatives):
                # Strategy: Keep creator, sample random campaign (or vice versa)
                if torch.rand(1).item() < 0.5:
                    # Same creator, different campaign
                    neg_campaign = torch.randint(0, num_campaigns, (1,))
                    neg_creator_ids.append(creator_id)
                    neg_campaign_ids.append(neg_campaign.item())
                else:
                    # Same campaign, different creator
                    neg_creator = torch.randint(0, num_creators, (1,))
                    neg_creator_ids.append(neg_creator.item())
                    neg_campaign_ids.append(campaign_id)
        
        return {
            'creator_ids': torch.tensor(neg_creator_ids, dtype=torch.long),
            'campaign_ids': torch.tensor(neg_campaign_ids, dtype=torch.long)
        }


class BayesianPersonalizedRanking(nn.Module):
    """
    BPR loss for implicit feedback
    Alternative to NCF with binary cross-entropy
    
    Based on "BPR: Bayesian Personalized Ranking from Implicit Feedback" (Rendle et al., 2009)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
        """
        BPR loss: maximize difference between positive and negative scores
        
        Args:
            pos_scores: Scores for positive items
            neg_scores: Scores for negative items
        
        Returns:
            BPR loss (lower is better)
        """
        # BPR loss: -log(sigmoid(pos_score - neg_score))
        diff = pos_scores - neg_scores
        loss = -torch.log(torch.sigmoid(diff) + 1e-10).mean()
        
        return loss


if __name__ == '__main__':
    # Test NCF model
    num_creators = 10000
    num_campaigns = 5000
    batch_size = 256
    
    model = NeuralCollaborativeFiltering(
        num_creators=num_creators,
        num_campaigns=num_campaigns,
        embedding_dim=64,
        mlp_layers=[128, 64, 32]
    )
    
    print(f"NCF Model initialized")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Sample batch
    creator_ids = torch.randint(0, num_creators, (batch_size,))
    campaign_ids = torch.randint(0, num_campaigns, (batch_size,))
    
    # Forward pass
    predictions = model(creator_ids, campaign_ids)
    
    print(f"\nPrediction shape: {predictions.shape}")
    print(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
    print(f"Mean prediction: {predictions.mean():.4f}")
    
    # Test trainer
    trainer = NCFTrainer(model)
    
    positive_samples = {
        'creator_ids': creator_ids[:32],
        'campaign_ids': campaign_ids[:32]
    }
    
    negative_samples = NCFTrainer.generate_negative_samples(
        positive_samples['creator_ids'],
        positive_samples['campaign_ids'],
        num_creators,
        num_campaigns,
        num_negatives=4
    )
    
    losses = trainer.train_step(positive_samples, negative_samples)
    
    print("\nTraining step losses:")
    for key, value in losses.items():
        print(f"  {key}: {value:.4f}")
