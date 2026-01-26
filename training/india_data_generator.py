"""
India-Specific Training Data Generator
Generates realistic Indian influencer marketing data with:
- Indian creator demographics and preferences
- Regional language content
- India-specific platforms (ShareChat, Moj, Josh)
- Indian brand categories and budgets
- Cultural and regional preferences
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# India-specific configurations
INDIAN_CITIES = [
    'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata',
    'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow', 'Chandigarh', 'Kochi'
]

INDIAN_LANGUAGES = [
    'Hindi', 'English', 'Tamil', 'Telugu', 'Marathi', 'Bengali',
    'Kannada', 'Malayalam', 'Gujarati', 'Punjabi'
]

INDIAN_CATEGORIES = [
    'Fashion & Lifestyle', 'Technology & Gadgets', 'Food & Cooking',
    'Entertainment & Comedy', 'Education & Skills', 'Finance & Investment',
    'Health & Fitness', 'Travel & Tourism', 'Gaming', 'Beauty & Makeup',
    'Devotional & Spirituality', 'Regional Content', 'News & Current Affairs',
    'Music & Dance', 'Sports & Cricket'
]

INDIAN_PLATFORMS = [
    'Instagram', 'YouTube', 'Facebook', 'Twitter', 'LinkedIn',
    'ShareChat', 'Moj', 'Josh', 'Chingari', 'TikTok (historical)'
]

INDIAN_BRANDS = [
    'Flipkart', 'Amazon India', 'Myntra', 'Nykaa', 'Swiggy', 'Zomato',
    'PhonePe', 'Paytm', 'CRED', 'Dream11', 'Byju\'s', 'Unacademy',
    'Tata', 'Reliance', 'Asian Paints', 'Nestle India', 'HUL',
    'Boat', 'Noise', 'Realme', 'OnePlus', 'Xiaomi', 'Mamaearth',
    'Sugar Cosmetics', 'boAt Lifestyle', 'Wow Skin Science'
]

class IndiaDataGenerator:
    """Generate India-specific influencer marketing data"""
    
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        self.current_date = datetime(2025, 11, 13)
    
    def generate_indian_creators(self, n=10000):
        """Generate Indian creator profiles"""
        print(f"Generating {n} Indian creator profiles...")
        
        creators = []
        for i in range(n):
            # Follower distribution (log-normal, India-specific)
            followers = int(np.random.lognormal(mean=8.5, sigma=2.2))
            followers = max(500, min(followers, 15_000_000))
            
            # Tier based on followers
            if followers < 10_000:
                tier = 'nano'
            elif followers < 50_000:
                tier = 'micro'
            elif followers < 500_000:
                tier = 'mid'
            elif followers < 1_000_000:
                tier = 'macro'
            else:
                tier = 'mega'
            
            # Engagement rate (higher for smaller creators)
            base_engagement = np.random.beta(3, 30) * 100
            tier_multipliers = {'nano': 1.5, 'micro': 1.3, 'mid': 1.0, 'macro': 0.8, 'mega': 0.6}
            engagement = base_engagement * tier_multipliers[tier]
            
            # Indian-specific attributes
            primary_language = np.random.choice(INDIAN_LANGUAGES, p=[
                0.40, 0.30, 0.08, 0.07, 0.05, 0.04, 0.02, 0.02, 0.01, 0.01
            ])
            
            city = np.random.choice(INDIAN_CITIES)
            
            # Categories (1-3 per creator)
            num_categories = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
            categories = np.random.choice(INDIAN_CATEGORIES, size=num_categories, replace=False).tolist()
            
            # Platforms (1-4 per creator)
            num_platforms = np.random.choice([1, 2, 3, 4], p=[0.3, 0.4, 0.2, 0.1])
            platforms = np.random.choice(INDIAN_PLATFORMS, size=num_platforms, replace=False).tolist()
            
            # Age demographics (Indian audience)
            age_18_24 = np.random.beta(5, 3) * 100  # Younger skew
            age_25_34 = np.random.beta(4, 4) * 100
            age_35_44 = max(0, 100 - age_18_24 - age_25_34)
            
            # Gender (varies by category)
            if 'Beauty & Makeup' in categories or 'Fashion & Lifestyle' in categories:
                female_pct = np.random.beta(7, 3) * 100
            elif 'Gaming' in categories or 'Technology & Gadgets' in categories:
                female_pct = np.random.beta(3, 7) * 100
            else:
                female_pct = np.random.beta(5, 5) * 100
            
            creator = {
                'creator_id': f'IN_CR_{i+1:05d}',
                'name': f'Creator_{i+1}',
                'follower_count': followers,
                'tier': tier,
                'engagement_rate': round(engagement, 2),
                'categories': ','.join(categories),
                'platforms': ','.join(platforms),
                'primary_language': primary_language,
                'city': city,
                'audience_age_18_24': round(age_18_24, 1),
                'audience_age_25_34': round(age_25_34, 1),
                'audience_age_35_44': round(age_35_44, 1),
                'audience_female_pct': round(female_pct, 1),
                'audience_male_pct': round(100 - female_pct, 1),
                'total_campaigns': np.random.poisson(lam=5),
                'successful_campaigns': 0,  # Will calculate
                'success_rate': 0,
                'overall_rating': round(np.random.beta(8, 2) * 5, 2),
                'is_verified': np.random.random() < (0.7 if tier in ['macro', 'mega'] else 0.2),
                'total_earnings': 0,
                'avg_response_time_hours': np.random.exponential(scale=12),
                'completion_rate': round(np.random.beta(9, 2) * 100, 1),
                'created_at': (self.current_date - timedelta(days=np.random.randint(30, 1095))).isoformat()
            }
            
            creator['successful_campaigns'] = int(creator['total_campaigns'] * creator['completion_rate'] / 100)
            creator['success_rate'] = round(creator['successful_campaigns'] / max(1, creator['total_campaigns']) * 100, 1)
            creator['total_earnings'] = creator['successful_campaigns'] * np.random.lognormal(mean=8.5, sigma=1.5)
            
            creators.append(creator)
        
        df = pd.DataFrame(creators)
        print(f"✓ Generated {len(df)} creators")
        print(f"  Tier distribution: {df['tier'].value_counts().to_dict()}")
        print(f"  Top languages: {df['primary_language'].value_counts().head(3).to_dict()}")
        print(f"  Avg engagement: {df['engagement_rate'].mean():.2f}%")
        
        return df
    
    def generate_indian_campaigns(self, n=5000):
        """Generate Indian brand campaigns"""
        print(f"\nGenerating {n} Indian campaigns...")
        
        campaigns = []
        for i in range(n):
            # Budget in INR (power law distribution)
            budget_inr = np.random.pareto(a=1.5) * 20000 + 5000
            budget_inr = min(budget_inr, 5_000_000)
            
            category = np.random.choice(INDIAN_CATEGORIES)
            platform = np.random.choice(INDIAN_PLATFORMS[:5])  # Focus on major platforms
            brand = np.random.choice(INDIAN_BRANDS)
            
            # Campaign duration
            duration = np.random.choice([7, 14, 21, 30, 45, 60], p=[0.2, 0.3, 0.2, 0.2, 0.07, 0.03])
            
            # Target demographics
            target_age = np.random.choice(['18-24', '25-34', '35-44', '18-34'])
            target_gender = np.random.choice(['All', 'Female', 'Male'], p=[0.6, 0.25, 0.15])
            
            # Min requirements based on budget
            if budget_inr < 25000:
                min_followers = np.random.randint(5000, 25000)
                min_engagement = round(np.random.uniform(2.0, 5.0), 1)
            elif budget_inr < 100000:
                min_followers = np.random.randint(25000, 100000)
                min_engagement = round(np.random.uniform(3.0, 6.0), 1)
            else:
                min_followers = np.random.randint(100000, 1000000)
                min_engagement = round(np.random.uniform(2.0, 4.0), 1)
            
            # Deliverables
            num_deliverables = np.random.choice([1, 2, 3, 4], p=[0.3, 0.4, 0.2, 0.1])
            deliverable_types = ['Post', 'Story', 'Reel', 'Video', 'Live Session', 'Review']
            deliverables = np.random.choice(deliverable_types, size=num_deliverables, replace=False).tolist()
            
            campaign = {
                'campaign_id': f'IN_CMP_{i+1:05d}',
                'title': f'{brand} - {category} Campaign',
                'description': f'Promote {brand} products to {target_age} audience',
                'brand_name': brand,
                'category': category,
                'platform': platform,
                'budget': round(budget_inr, 2),
                'duration_days': duration,
                'target_age_group': target_age,
                'target_gender': target_gender,
                'min_followers': min_followers,
                'min_engagement': min_engagement,
                'deliverables': ','.join(deliverables),
                'status': np.random.choice(['active', 'draft', 'completed'], p=[0.4, 0.2, 0.4]),
                'brand_reputation': round(np.random.beta(7, 2) * 5, 2),
                'created_at': (self.current_date - timedelta(days=np.random.randint(1, 180))).isoformat()
            }
            
            campaigns.append(campaign)
        
        df = pd.DataFrame(campaigns)
        print(f"✓ Generated {len(df)} campaigns")
        print(f"  Budget range: ₹{df['budget'].min():.0f} - ₹{df['budget'].max():.0f}")
        print(f"  Avg budget: ₹{df['budget'].mean():.0f}")
        print(f"  Top brands: {df['brand_name'].value_counts().head(3).to_dict()}")
        
        return df
    
    def generate_interactions(self, creators_df, campaigns_df, n=100000):
        """Generate creator-campaign interactions with India-specific match scores"""
        print(f"\nGenerating {n} interactions...")
        
        interactions = []
        for i in range(n):
            creator = creators_df.sample(1).iloc[0]
            campaign = campaigns_df.sample(1).iloc[0]
            
            # Calculate India-specific match score
            match_score = self._calculate_india_match_score(creator, campaign)
            
            # Success probability based on match and Indian market dynamics
            success_prob = match_score * 0.6 + np.random.beta(3, 2) * 0.4
            is_successful = np.random.random() < success_prob
            
            # ROI (higher for regional content and vernacular creators)
            language_boost = 1.3 if creator['primary_language'] in ['Hindi', 'Tamil', 'Telugu'] else 1.0
            base_roi = np.random.lognormal(mean=1.2, sigma=0.6) * language_boost
            roi = base_roi if is_successful else base_roi * 0.3
            
            interaction = {
                'interaction_id': f'IN_INT_{i+1:06d}',
                'creator_id': creator['creator_id'],
                'campaign_id': campaign['campaign_id'],
                'match_score': round(match_score, 4),
                'applied': np.random.random() < match_score * 0.8,
                'accepted': is_successful,
                'success_probability': round(success_prob, 4),
                'estimated_roi': round(roi, 2),
                'actual_roi': round(roi * np.random.uniform(0.8, 1.2), 2) if is_successful else 0,
                'created_at': campaign['created_at']
            }
            
            interactions.append(interaction)
        
        df = pd.DataFrame(interactions)
        print(f"✓ Generated {len(df)} interactions")
        print(f"  Match rate (>0.5): {(df['match_score'] > 0.5).sum() / len(df) * 100:.1f}%")
        print(f"  Application rate: {df['applied'].sum() / len(df) * 100:.1f}%")
        print(f"  Success rate: {df['accepted'].sum() / len(df) * 100:.1f}%")
        print(f"  Avg ROI: {df[df['actual_roi'] > 0]['actual_roi'].mean():.2f}x")
        
        return df
    
    def _calculate_india_match_score(self, creator, campaign):
        """Calculate match score with India-specific factors"""
        score = 0.0
        
        # Category match (35% weight)
        creator_cats = set(creator['categories'].split(','))
        campaign_cat = campaign['category']
        if campaign_cat in creator_cats:
            score += 0.35
        
        # Platform match (20% weight)
        creator_platforms = set(creator['platforms'].split(','))
        if campaign['platform'] in creator_platforms:
            score += 0.20
        
        # Follower requirement (15% weight)
        if creator['follower_count'] >= campaign['min_followers']:
            score += 0.15
        
        # Engagement requirement (15% weight)
        if creator['engagement_rate'] >= campaign['min_engagement']:
            score += 0.15
        
        # Budget fit (10% weight) - India-specific pricing
        expected_price = creator['follower_count'] * 0.5  # ₹0.5 per follower rough estimate
        budget_fit = min(1.0, campaign['budget'] / max(expected_price, 1000))
        score += 0.10 * budget_fit
        
        # Regional language bonus (5% weight)
        if creator['primary_language'] in ['Hindi', 'English']:
            score += 0.05
        
        return min(1.0, score + np.random.normal(0, 0.05))
    
    def generate_all(self, creators=10000, campaigns=5000, interactions=100000):
        """Generate complete India-specific dataset"""
        print("=" * 60)
        print("INDIA-SPECIFIC TRAINING DATA GENERATION")
        print("=" * 60)
        
        # Generate data
        creators_df = self.generate_indian_creators(creators)
        campaigns_df = self.generate_indian_campaigns(campaigns)
        interactions_df = self.generate_interactions(creators_df, campaigns_df, interactions)
        
        # Save to CSV
        print("\nSaving to CSV files...")
        creators_df.to_csv('data/raw/india_creators.csv', index=False)
        campaigns_df.to_csv('data/raw/india_campaigns.csv', index=False)
        interactions_df.to_csv('data/raw/india_interactions.csv', index=False)
        
        print("\n✓ India-specific data generation complete!")
        print(f"  Creators: {len(creators_df)}")
        print(f"  Campaigns: {len(campaigns_df)}")
        print(f"  Interactions: {len(interactions_df)}")
        print(f"\nFiles saved to data/raw/")
        
        return creators_df, campaigns_df, interactions_df


if __name__ == '__main__':
    generator = IndiaDataGenerator(random_seed=42)
    generator.generate_all(creators=15000, campaigns=7000, interactions=150000)
