"""
Advanced ETL Pipeline for Creator-Campaign Matching
- Extracts data from PostgreSQL database
- Transforms and engineers sophisticated features
- Loads processed data for ML model training
"""
import os
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime
import json
from typing import Dict, Tuple
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Database connection from environment variables
DB_URL = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost:5432/influencia')
engine = create_engine(DB_URL)

# Create data directory if not exists
os.makedirs('ai/data', exist_ok=True)
os.makedirs('ai/models', exist_ok=True)

class ETLPipeline:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def extract_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Extract data from database"""
        print("📥 Extracting data from database...")
        
        # Extract creators with user info
        creators_query = """
            SELECT c.*, u.email, u.first_name, u.last_name, u.created_at as user_created_at
            FROM creators c
            JOIN users u ON c.user_id = u.id
        """
        creators = pd.read_sql(creators_query, engine)
        
        # Extract campaigns
        campaigns_query = """
            SELECT c.*, b.company_name as brand_name
            FROM campaigns c
            JOIN brands b ON c.brand_id = b.id
        """
        campaigns = pd.read_sql(campaigns_query, engine)
        
        # Extract collaborations with outcomes
        collaborations_query = """
            SELECT col.*, 
                   cr.bio as creator_bio, cr.location as creator_location,
                   cr.total_campaigns, cr.overall_rating, cr.categories as creator_categories,
                   cam.title as campaign_title, cam.platform, cam.category as campaign_category,
                   cam.budget, cam.status as campaign_status
            FROM collaborations col
            JOIN creators cr ON col.creator_id = cr.id
            JOIN campaigns cam ON col.campaign_id = cam.id
        """
        collaborations = pd.read_sql(collaborations_query, engine)
        
        print(f"✅ Extracted {len(creators)} creators, {len(campaigns)} campaigns, {len(collaborations)} collaborations")
        return creators, campaigns, collaborations
    
    def engineer_creator_features(self, creators: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering for creators"""
        print("🔧 Engineering creator features...")
        
        # Experience level (numeric)
        creators['experience_score'] = creators['total_campaigns'].apply(
            lambda x: 5 if x >= 50 else 4 if x >= 20 else 3 if x >= 10 else 2 if x >= 1 else 1
        )
        
        # Rating features
        creators['is_highly_rated'] = (creators['overall_rating'] >= 4.5).astype(int)
        creators['is_low_rated'] = (creators['overall_rating'] < 3.0).astype(int)
        
        # Category diversity
        creators['num_categories'] = creators['categories'].apply(
            lambda x: len(json.loads(x)) if isinstance(x, str) else len(x) if isinstance(x, list) else 0
        )
        
        # Languages diversity
        creators['num_languages'] = creators['languages'].apply(
            lambda x: len(json.loads(x)) if isinstance(x, str) else len(x) if isinstance(x, list) else 0
        )
        
        # Social media presence (estimate based on campaigns)
        creators['estimated_followers'] = creators['total_campaigns'] * 10000
        creators['estimated_engagement_rate'] = np.clip(
            (creators['overall_rating'] / 5.0) * 0.05 + np.random.uniform(0.01, 0.03, len(creators)),
            0.01, 0.10
        )
        
        # Account age (days since creation)
        creators['account_age_days'] = (datetime.now() - pd.to_datetime(creators['user_created_at'])).dt.days
        
        # Success rate proxy
        creators['success_rate'] = np.clip(creators['overall_rating'] / 5.0, 0, 1)
        
        # Versatility score
        creators['versatility_score'] = creators['num_categories'] * creators['num_languages'] * 0.1
        
        print(f"✅ Engineered {creators.shape[1]} features for creators")
        return creators
    
    def engineer_campaign_features(self, campaigns: pd.DataFrame) -> pd.DataFrame:
        """Advanced feature engineering for campaigns"""
        print("🔧 Engineering campaign features...")
        
        # Budget categories
        campaigns['budget_category'] = pd.cut(
            campaigns['budget'], 
            bins=[0, 1000, 5000, 10000, 50000, np.inf],
            labels=['micro', 'small', 'medium', 'large', 'enterprise']
        )
        
        # Campaign duration
        campaigns['duration_days'] = (
            pd.to_datetime(campaigns['end_date']) - pd.to_datetime(campaigns['start_date'])
        ).dt.days
        
        # Requirements complexity
        campaigns['has_follower_req'] = campaigns['requirements'].apply(
            lambda x: 'min_followers' in json.loads(x) if isinstance(x, str) else False
        )
        campaigns['has_engagement_req'] = campaigns['requirements'].apply(
            lambda x: 'min_engagement_rate' in json.loads(x) if isinstance(x, str) else False
        )
        
        # Target audience specificity
        campaigns['target_specificity_score'] = campaigns['target_audience'].apply(
            lambda x: len(json.loads(x)) if isinstance(x, str) else 0
        )
        
        # Budget per day
        campaigns['budget_per_day'] = campaigns['budget'] / campaigns['duration_days'].replace(0, 1)
        
        print(f"✅ Engineered {campaigns.shape[1]} features for campaigns")
        return campaigns
    
    def create_training_dataset(self, creators: pd.DataFrame, campaigns: pd.DataFrame, 
                                collaborations: pd.DataFrame) -> pd.DataFrame:
        """Create unified training dataset with creator-campaign pairs"""
        print("🔧 Creating training dataset...")
        
        training_data = []
        
        # Create positive examples (actual collaborations)
        for _, collab in collaborations.iterrows():
            creator = creators[creators['id'] == collab['creator_id']].iloc[0]
            campaign = campaigns[campaigns['id'] == collab['campaign_id']].iloc[0]
            
            row = self._create_feature_row(creator, campaign, collab)
            row['outcome'] = 1 if collab['status'] in ['accepted', 'completed'] else 0
            row['collaboration_id'] = collab['id']
            training_data.append(row)
        
        # Create negative examples (random non-collaborations)
        num_negatives = len(collaborations) * 2
        for _ in range(num_negatives):
            creator = creators.sample(1).iloc[0]
            campaign = campaigns.sample(1).iloc[0]
            
            # Check if this pair already exists
            existing = collaborations[
                (collaborations['creator_id'] == creator['id']) & 
                (collaborations['campaign_id'] == campaign['id'])
            ]
            
            if len(existing) == 0:
                row = self._create_feature_row(creator, campaign, None)
                row['outcome'] = 0
                row['collaboration_id'] = None
                training_data.append(row)
        
        df = pd.DataFrame(training_data)
        print(f"✅ Created training dataset with {len(df)} samples")
        return df
    
    def _create_feature_row(self, creator: pd.Series, campaign: pd.Series, 
                           collab: pd.Series = None) -> Dict:
        """Create feature row for creator-campaign pair"""
        
        # Category match
        creator_cats = json.loads(creator['categories']) if isinstance(creator['categories'], str) else creator['categories']
        campaign_cat = campaign['category']
        category_match = 1.0 if campaign_cat in creator_cats else 0.0
        
        # Extract requirements
        requirements = json.loads(campaign['requirements']) if isinstance(campaign['requirements'], str) else campaign['requirements']
        min_followers = requirements.get('min_followers', 0)
        min_engagement = requirements.get('min_engagement_rate', 0)
        
        # Followers match
        followers_match = 1.0 if creator['estimated_followers'] >= min_followers else 0.5
        
        # Engagement match
        engagement_match = 1.0 if creator['estimated_engagement_rate'] >= min_engagement else 0.5
        
        # Platform match (simplified)
        platform_match = 0.8  # Assume most creators support most platforms
        
        # Budget fit
        estimated_cost = (creator['estimated_followers'] / 1000) * creator['overall_rating'] * 10
        budget_fit = min(campaign['budget'] / estimated_cost, 2.0) if estimated_cost > 0 else 1.0
        
        return {
            'creator_id': creator['id'],
            'campaign_id': campaign['id'],
            'category_match': category_match,
            'followers_match': followers_match,
            'engagement_match': engagement_match,
            'platform_match': platform_match,
            'experience_score': creator['experience_score'],
            'overall_rating': creator['overall_rating'],
            'num_categories': creator['num_categories'],
            'num_languages': creator['num_languages'],
            'account_age_days': creator['account_age_days'],
            'estimated_followers': creator['estimated_followers'],
            'estimated_engagement_rate': creator['estimated_engagement_rate'],
            'campaign_budget': campaign['budget'],
            'campaign_duration_days': campaign['duration_days'],
            'budget_fit': budget_fit,
            'versatility_score': creator['versatility_score'],
            'success_rate': creator['success_rate']
        }
    
    def save_processed_data(self, creators: pd.DataFrame, campaigns: pd.DataFrame, 
                           training_data: pd.DataFrame):
        """Save processed data to CSV"""
        print("💾 Saving processed data...")
        
        creators.to_csv('ai/data/creators_processed.csv', index=False)
        campaigns.to_csv('ai/data/campaigns_processed.csv', index=False)
        training_data.to_csv('ai/data/training_data.csv', index=False)
        
        print("✅ Saved all processed data")
    
    def run_pipeline(self):
        """Run complete ETL pipeline"""
        print("🚀 Starting ETL Pipeline...")
        
        # Extract
        creators, campaigns, collaborations = self.extract_data()
        
        # Transform
        creators = self.engineer_creator_features(creators)
        campaigns = self.engineer_campaign_features(campaigns)
        
        # Create training dataset
        training_data = self.create_training_dataset(creators, campaigns, collaborations)
        
        # Load
        self.save_processed_data(creators, campaigns, training_data)
        
        print("🎉 ETL Pipeline completed successfully!")
        return creators, campaigns, training_data


if __name__ == "__main__":
    pipeline = ETLPipeline()
    pipeline.run_pipeline()
