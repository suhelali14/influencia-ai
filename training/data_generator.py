"""
Advanced Training Data Generator for Influencer Marketing Platform
Generates realistic synthetic data with proper statistical distributions
Based on industry benchmarks and research papers
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import random
from typing import Dict, List, Tuple

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Industry data based on research
CATEGORIES = [
    'Fashion', 'Beauty', 'Fitness', 'Gaming', 'Technology', 'Food', 'Travel',
    'Lifestyle', 'Music', 'Sports', 'Education', 'Finance', 'Health', 'Art',
    'Entertainment', 'Business', 'Parenting', 'Pets', 'DIY', 'Photography'
]

PLATFORMS = ['Instagram', 'YouTube', 'TikTok', 'Twitter', 'Facebook', 'LinkedIn']

LANGUAGES = ['English', 'Spanish', 'French', 'German', 'Portuguese', 'Italian', 'Japanese', 'Korean', 'Arabic', 'Hindi']

INDUSTRIES = [
    'E-commerce', 'Fashion', 'Technology', 'Food & Beverage', 'Beauty & Cosmetics',
    'Health & Wellness', 'Entertainment', 'Finance', 'Education', 'Travel & Tourism',
    'Automotive', 'Real Estate', 'Gaming', 'Sports', 'Home & Garden'
]

AGE_GROUPS = ['13-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+']

class AdvancedDataGenerator:
    """Generate realistic synthetic training data"""
    
    def __init__(self, num_creators=10000, num_campaigns=5000, num_interactions=100000):
        self.num_creators = num_creators
        self.num_campaigns = num_campaigns
        self.num_interactions = num_interactions
        
    def generate_all_data(self) -> Dict[str, pd.DataFrame]:
        """Generate complete dataset"""
        print("Generating creators...")
        creators_df = self.generate_creators()
        
        print("Generating campaigns...")
        campaigns_df = self.generate_campaigns()
        
        print("Generating interactions...")
        interactions_df = self.generate_interactions(creators_df, campaigns_df)
        
        print("Generating historical performance...")
        historical_df = self.generate_historical_data(creators_df)
        
        return {
            'creators': creators_df,
            'campaigns': campaigns_df,
            'interactions': interactions_df,
            'historical': historical_df
        }
    
    def generate_creators(self) -> pd.DataFrame:
        """
        Generate realistic creator profiles
        Follower distribution: Log-normal (mimics real social media)
        Engagement rate: Beta distribution
        """
        creators = []
        
        for i in range(self.num_creators):
            # Follower count: Log-normal distribution
            # Mean: 50k followers, highly skewed (few mega influencers)
            followers = int(np.random.lognormal(mean=10.5, sigma=1.5))
            followers = max(1000, min(followers, 10_000_000))  # Clamp between 1k-10M
            
            # Categorize influencer tier
            if followers < 10_000:
                tier = 'nano'
                base_engagement = 0.08  # 8% (nano influencers have higher engagement)
            elif followers < 50_000:
                tier = 'micro'
                base_engagement = 0.06  # 6%
            elif followers < 500_000:
                tier = 'mid'
                base_engagement = 0.04  # 4%
            elif followers < 1_000_000:
                tier = 'macro'
                base_engagement = 0.025  # 2.5%
            else:
                tier = 'mega'
                base_engagement = 0.015  # 1.5% (mega influencers have lower engagement)
            
            # Engagement rate: Beta distribution around base rate
            engagement_rate = np.random.beta(2, 5) * base_engagement * 2
            engagement_rate = max(0.005, min(engagement_rate, 0.15))  # 0.5% - 15%
            
            # Following count (realistic ratio)
            following = int(followers * np.random.uniform(0.1, 2.0) if tier in ['nano', 'micro'] else followers * np.random.uniform(0.01, 0.5))
            
            # Total posts
            total_posts = int(np.random.gamma(shape=3, scale=100))
            total_posts = max(10, min(total_posts, 5000))
            
            # Account age (in days)
            account_age_days = int(np.random.gamma(shape=2, scale=365))
            account_age_days = max(30, min(account_age_days, 3650))  # 1 month to 10 years
            
            # Categories (1-3 categories per creator)
            num_categories = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
            categories = random.sample(CATEGORIES, num_categories)
            
            # Platforms (1-2 primary platforms)
            num_platforms = np.random.choice([1, 2], p=[0.6, 0.4])
            platforms = random.sample(PLATFORMS, num_platforms)
            
            # Languages
            num_languages = np.random.choice([1, 2], p=[0.8, 0.2])
            languages = random.sample(LANGUAGES, num_languages)
            
            # Campaign history (based on tier and account age)
            base_campaigns = int(account_age_days / 365 * {'nano': 2, 'micro': 5, 'mid': 10, 'macro': 20, 'mega': 30}[tier])
            total_campaigns = max(0, int(np.random.poisson(base_campaigns)))
            
            # Success rate (better for experienced creators)
            experience_factor = min(total_campaigns / 50, 1.0)
            success_rate = np.random.beta(2, 2) * 0.6 + 0.3 + (experience_factor * 0.1)  # 30-100%
            successful_campaigns = int(total_campaigns * success_rate)
            
            # Overall rating (correlated with success rate)
            overall_rating = success_rate * 5 * np.random.uniform(0.9, 1.1)
            overall_rating = max(1.0, min(overall_rating, 5.0))
            
            # Earnings (based on tier and campaigns)
            avg_earnings_per_campaign = {
                'nano': np.random.uniform(100, 500),
                'micro': np.random.uniform(500, 2000),
                'mid': np.random.uniform(2000, 10000),
                'macro': np.random.uniform(10000, 50000),
                'mega': np.random.uniform(50000, 500000)
            }[tier]
            total_earnings = successful_campaigns * avg_earnings_per_campaign * np.random.uniform(0.8, 1.2)
            
            # Verification (more likely for larger accounts)
            verification_prob = min(followers / 1_000_000, 0.9)
            is_verified = np.random.random() < verification_prob
            
            # Audience demographics (realistic distributions)
            age_distribution = self._generate_age_distribution(categories)
            gender_distribution = self._generate_gender_distribution(categories)
            
            # Bio (simulated)
            bio_length = int(np.random.gamma(shape=2, scale=30))
            bio = f"{'Passionate about ' + ' & '.join(categories[:2])}. {tier.capitalize()} influencer."
            
            creator = {
                'creator_id': i + 1,
                'followers': followers,
                'following': following,
                'engagement_rate': round(engagement_rate, 4),
                'total_posts': total_posts,
                'account_age_days': account_age_days,
                'tier': tier,
                'categories': json.dumps(categories),
                'platforms': json.dumps(platforms),
                'languages': json.dumps(languages),
                'total_campaigns': total_campaigns,
                'successful_campaigns': successful_campaigns,
                'success_rate': round(success_rate, 3),
                'overall_rating': round(overall_rating, 2),
                'total_earnings': round(total_earnings, 2),
                'is_verified': is_verified,
                'bio': bio,
                'audience_age_18_24': age_distribution.get('18-24', 0),
                'audience_age_25_34': age_distribution.get('25-34', 0),
                'audience_age_35_44': age_distribution.get('35-44', 0),
                'audience_female_pct': gender_distribution['female'],
                'audience_male_pct': gender_distribution['male'],
            }
            
            creators.append(creator)
        
        return pd.DataFrame(creators)
    
    def generate_campaigns(self) -> pd.DataFrame:
        """Generate realistic campaign data"""
        campaigns = []
        
        for i in range(self.num_campaigns):
            # Category
            category = random.choice(CATEGORIES)
            
            # Platform
            platform = random.choice(PLATFORMS)
            
            # Industry
            industry = random.choice(INDUSTRIES)
            
            # Budget: Power law distribution (few high-budget campaigns)
            budget = int(np.random.pareto(a=1.5) * 1000 + 500)
            budget = max(500, min(budget, 500_000))  # $500 - $500k
            
            # Duration (7-90 days)
            duration_days = int(np.random.gamma(shape=2, scale=15))
            duration_days = max(7, min(duration_days, 90))
            
            # Start date (within next 90 days)
            days_until_start = int(np.random.uniform(0, 90))
            start_date = datetime.now() + timedelta(days=days_until_start)
            end_date = start_date + timedelta(days=duration_days)
            
            # Requirements
            min_followers = int(np.random.choice([1000, 5000, 10000, 50000, 100000], p=[0.3, 0.3, 0.2, 0.15, 0.05]))
            min_engagement = round(np.random.uniform(0.01, 0.05), 3)
            
            num_deliverables = np.random.choice([1, 2, 3, 4, 5], p=[0.2, 0.3, 0.3, 0.15, 0.05])
            deliverables = random.sample(['Post', 'Story', 'Reel', 'Video', 'Blog'], num_deliverables)
            
            content_rights = np.random.random() < 0.4
            
            # Target audience
            target_age = random.choice(AGE_GROUPS)
            target_gender = random.choice(['Male', 'Female', 'All'])
            
            # Brand info
            brand_total_campaigns = int(np.random.poisson(10))
            brand_avg_rating = round(np.random.beta(5, 2) * 5, 2)
            brand_completion_rate = round(np.random.beta(8, 2), 3)
            
            # Competition
            applications_count = int(np.random.gamma(shape=2, scale=20))
            
            campaign = {
                'campaign_id': i + 1,
                'category': category,
                'platform': platform,
                'industry': industry,
                'budget': budget,
                'duration_days': duration_days,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'min_followers': min_followers,
                'min_engagement': min_engagement,
                'deliverables': json.dumps(deliverables),
                'content_rights': content_rights,
                'target_age_group': target_age,
                'target_gender': target_gender,
                'brand_total_campaigns': brand_total_campaigns,
                'brand_avg_rating': brand_avg_rating,
                'brand_completion_rate': brand_completion_rate,
                'applications_count': applications_count,
            }
            
            campaigns.append(campaign)
        
        return pd.DataFrame(campaigns)
    
    def generate_interactions(self, creators_df: pd.DataFrame, campaigns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate creator-campaign interaction data
        Includes match labels, success outcomes, and ROI
        """
        interactions = []
        
        for _ in range(self.num_interactions):
            creator = creators_df.sample(1).iloc[0]
            campaign = campaigns_df.sample(1).iloc[0]
            
            # Calculate match score based on multiple factors
            match_score = self._calculate_realistic_match_score(creator, campaign)
            
            # Binary match label (threshold: 0.6)
            matched = match_score > 0.6
            
            # If matched, calculate success and ROI
            if matched:
                # Success probability (higher for better matches)
                success_prob = match_score * np.random.beta(5, 2)
                successful = np.random.random() < success_prob
                
                # ROI (if successful)
                if successful:
                    # Base ROI depends on match quality and creator tier
                    tier_roi_multiplier = {
                        'nano': 8.0,
                        'micro': 6.0,
                        'mid': 4.5,
                        'macro': 3.5,
                        'mega': 2.5
                    }[creator['tier']]
                    
                    base_roi = tier_roi_multiplier * match_score
                    roi = base_roi * np.random.lognormal(0, 0.3)  # Add variance
                    roi = max(0.5, min(roi, 15.0))  # Clamp between 0.5x - 15x
                else:
                    roi = np.random.uniform(-0.5, 0.8)  # Failed campaigns
            else:
                successful = False
                roi = 0.0
                success_prob = 0.0
            
            interaction = {
                'creator_id': creator['creator_id'],
                'campaign_id': campaign['campaign_id'],
                'match_score': round(match_score, 4),
                'matched': matched,
                'successful': successful,
                'success_probability': round(success_prob, 4),
                'roi': round(roi, 2),
                'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 365))
            }
            
            interactions.append(interaction)
        
        return pd.DataFrame(interactions)
    
    def generate_historical_data(self, creators_df: pd.DataFrame) -> pd.DataFrame:
        """Generate historical performance data for creators"""
        historical = []
        
        # Sample 1000 creators for historical data
        sampled_creators = creators_df.sample(min(1000, len(creators_df)))
        
        for _, creator in sampled_creators.iterrows():
            # Generate 30-365 days of historical data
            num_days = int(np.random.uniform(30, 365))
            
            for day in range(num_days):
                date = datetime.now() - timedelta(days=day)
                
                # Simulate metrics with trends
                base_followers = creator['followers']
                daily_growth = np.random.normal(0, base_followers * 0.001)  # 0.1% daily variance
                followers = max(1000, int(base_followers + daily_growth * (num_days - day)))
                
                # Engagement varies
                base_engagement = creator['engagement_rate']
                engagement = max(0.001, base_engagement * np.random.lognormal(0, 0.2))
                
                # Post metrics
                posts_today = np.random.poisson(0.5)  # ~0.5 posts per day
                
                if posts_today > 0:
                    likes = int(followers * engagement * np.random.uniform(0.8, 1.2))
                    comments = int(likes * np.random.uniform(0.05, 0.15))
                    shares = int(likes * np.random.uniform(0.02, 0.08))
                else:
                    likes = comments = shares = 0
                
                historical.append({
                    'creator_id': creator['creator_id'],
                    'date': date.strftime('%Y-%m-%d'),
                    'followers': followers,
                    'engagement_rate': round(engagement, 4),
                    'posts': posts_today,
                    'likes': likes,
                    'comments': comments,
                    'shares': shares
                })
        
        return pd.DataFrame(historical)
    
    def _calculate_realistic_match_score(self, creator, campaign) -> float:
        """Calculate realistic match score based on multiple factors"""
        score = 0.5  # Base score
        
        # Category match (most important)
        creator_categories = json.loads(creator['categories'])
        if campaign['category'] in creator_categories:
            score += 0.25
        
        # Platform match
        creator_platforms = json.loads(creator['platforms'])
        if campaign['platform'] in creator_platforms:
            score += 0.15
        
        # Follower requirement
        if creator['followers'] >= campaign['min_followers']:
            follower_ratio = creator['followers'] / max(campaign['min_followers'], 1)
            score += min(0.1 * np.log10(follower_ratio + 1), 0.1)
        else:
            score -= 0.2  # Penalty for not meeting requirement
        
        # Engagement requirement
        if creator['engagement_rate'] >= campaign['min_engagement']:
            score += 0.1
        else:
            score -= 0.15
        
        # Budget fit (creator's average earnings vs campaign budget)
        avg_earnings = creator['total_earnings'] / max(creator['total_campaigns'], 1)
        budget_ratio = campaign['budget'] / max(avg_earnings, 1)
        if 0.5 <= budget_ratio <= 2.0:  # Good fit
            score += 0.1
        elif budget_ratio < 0.5 or budget_ratio > 3.0:  # Poor fit
            score -= 0.1
        
        # Experience bonus
        if creator['total_campaigns'] > 10:
            score += 0.05
        
        # Rating bonus
        if creator['overall_rating'] > 4.0:
            score += 0.05
        
        # Add some randomness
        score += np.random.normal(0, 0.1)
        
        return max(0.0, min(score, 1.0))
    
    def _generate_age_distribution(self, categories: List[str]) -> Dict[str, float]:
        """Generate realistic age distribution based on categories"""
        # Default distribution
        distribution = {
            '18-24': 0.30,
            '25-34': 0.35,
            '35-44': 0.20,
            '45-54': 0.10,
            '55-64': 0.05
        }
        
        # Adjust based on categories
        if 'Gaming' in categories or 'Music' in categories:
            distribution['18-24'] = 0.50
            distribution['25-34'] = 0.30
        elif 'Finance' in categories or 'Business' in categories:
            distribution['25-34'] = 0.40
            distribution['35-44'] = 0.30
        elif 'Health' in categories or 'Parenting' in categories:
            distribution['35-44'] = 0.35
            distribution['45-54'] = 0.25
        
        # Normalize
        total = sum(distribution.values())
        return {k: round(v / total, 3) for k, v in distribution.items()}
    
    def _generate_gender_distribution(self, categories: List[str]) -> Dict[str, float]:
        """Generate realistic gender distribution based on categories"""
        # Default: 50-50
        female_pct = 50.0
        
        # Adjust based on categories
        if 'Fashion' in categories or 'Beauty' in categories:
            female_pct = np.random.uniform(65, 85)
        elif 'Gaming' in categories or 'Technology' in categories:
            female_pct = np.random.uniform(25, 45)
        elif 'Fitness' in categories or 'Food' in categories:
            female_pct = np.random.uniform(55, 70)
        else:
            female_pct = np.random.uniform(45, 55)
        
        return {
            'female': round(female_pct, 1),
            'male': round(100 - female_pct, 1)
        }


def main():
    """Generate and save training data"""
    print("=" * 60)
    print("Advanced Training Data Generator")
    print("Generating realistic synthetic data for ML training...")
    print("=" * 60)
    
    generator = AdvancedDataGenerator(
        num_creators=10000,
        num_campaigns=5000,
        num_interactions=100000
    )
    
    datasets = generator.generate_all_data()
    
    # Save to CSV
    print("\nSaving datasets...")
    datasets['creators'].to_csv('data/raw/creators_full.csv', index=False)
    datasets['campaigns'].to_csv('data/raw/campaigns_full.csv', index=False)
    datasets['interactions'].to_csv('data/raw/interactions_full.csv', index=False)
    datasets['historical'].to_csv('data/raw/historical_performance.csv', index=False)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    
    print(f"\nCreators: {len(datasets['creators']):,}")
    print(f"  - Nano (<10k): {len(datasets['creators'][datasets['creators']['tier'] == 'nano']):,}")
    print(f"  - Micro (10k-50k): {len(datasets['creators'][datasets['creators']['tier'] == 'micro']):,}")
    print(f"  - Mid (50k-500k): {len(datasets['creators'][datasets['creators']['tier'] == 'mid']):,}")
    print(f"  - Macro (500k-1M): {len(datasets['creators'][datasets['creators']['tier'] == 'macro']):,}")
    print(f"  - Mega (>1M): {len(datasets['creators'][datasets['creators']['tier'] == 'mega']):,}")
    
    print(f"\nCampaigns: {len(datasets['campaigns']):,}")
    print(f"  - Avg Budget: ${datasets['campaigns']['budget'].mean():,.2f}")
    print(f"  - Median Budget: ${datasets['campaigns']['budget'].median():,.2f}")
    
    print(f"\nInteractions: {len(datasets['interactions']):,}")
    print(f"  - Matched: {datasets['interactions']['matched'].sum():,} ({datasets['interactions']['matched'].mean()*100:.1f}%)")
    print(f"  - Successful: {datasets['interactions']['successful'].sum():,} ({datasets['interactions']['successful'].mean()*100:.1f}%)")
    print(f"  - Avg Match Score: {datasets['interactions']['match_score'].mean():.3f}")
    print(f"  - Avg ROI (successful): {datasets['interactions'][datasets['interactions']['successful']]['roi'].mean():.2f}x")
    
    print(f"\nHistorical Records: {len(datasets['historical']):,}")
    
    print("\n" + "=" * 60)
    print("✅ Data generation complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
