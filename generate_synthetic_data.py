"""
Generate Synthetic Training Data for ML Models
Creates realistic creator-campaign pairs with labels for training
"""
import pandas as pd
import numpy as np
import json
import os

np.random.seed(42)

# Create data directory
os.makedirs('ai/data', exist_ok=True)

def generate_training_data(n_samples=1000):
    """Generate synthetic training data"""
    print(f"🎲 Generating {n_samples} synthetic training samples...")
    
    data = []
    
    # Categories for matching
    categories = ['fashion', 'lifestyle', 'tech', 'gaming', 'fitness', 'food', 
                  'beauty', 'travel', 'entertainment', 'education']
    platforms = ['instagram', 'youtube', 'tiktok', 'twitter']
    
    for i in range(n_samples):
        # Creator features
        creator_category = np.random.choice(categories)
        total_campaigns = np.random.randint(0, 100)
        overall_rating = np.random.uniform(2.5, 5.0)
        num_categories = np.random.randint(1, 6)
        num_languages = np.random.randint(1, 5)
        estimated_followers = np.random.choice([5000, 10000, 50000, 100000, 500000, 1000000])
        estimated_engagement_rate = np.random.uniform(0.01, 0.10)
        account_age_days = np.random.randint(30, 1825)  # 1 month to 5 years
        experience_score = min(5, max(1, 1 + total_campaigns // 10))
        
        # Campaign features
        campaign_category = np.random.choice(categories)
        campaign_budget = np.random.choice([1000, 5000, 10000, 20000, 50000])
        campaign_duration_days = np.random.randint(7, 90)
        min_followers = np.random.choice([0, 10000, 50000, 100000])
        min_engagement = np.random.uniform(0.02, 0.06)
        
        # Calculate matching features
        category_match = 1.0 if creator_category == campaign_category else 0.0
        followers_match = 1.0 if estimated_followers >= min_followers else 0.5
        engagement_match = 1.0 if estimated_engagement_rate >= min_engagement else 0.5
        platform_match = np.random.uniform(0.6, 1.0)
        
        # Budget fit
        estimated_cost = (estimated_followers / 1000) * overall_rating * 10
        budget_fit = min(campaign_budget / estimated_cost, 2.0) if estimated_cost > 0 else 1.0
        
        # Derived features
        success_rate = overall_rating / 5.0
        versatility_score = num_categories * num_languages * 0.1
        
        # Calculate base score for label
        base_score = (
            category_match * 30 +
            followers_match * 15 +
            engagement_match * 10 +
            platform_match * 10 +
            experience_score * 4 +
            overall_rating * 3 +
            budget_fit * 10 +
            success_rate * 10
        )
        
        # Add some randomness
        score = base_score + np.random.uniform(-10, 10)
        score = np.clip(score, 0, 100)
        
        # Outcome based on score with probability
        outcome_probability = score / 100
        outcome = 1 if np.random.random() < outcome_probability else 0
        
        data.append({
            'creator_id': f'creator_{i % 100}',
            'campaign_id': f'campaign_{i % 50}',
            'category_match': category_match,
            'followers_match': followers_match,
            'engagement_match': engagement_match,
            'platform_match': platform_match,
            'experience_score': experience_score,
            'overall_rating': overall_rating,
            'num_categories': num_categories,
            'num_languages': num_languages,
            'account_age_days': account_age_days,
            'estimated_followers': estimated_followers,
            'estimated_engagement_rate': estimated_engagement_rate,
            'campaign_budget': campaign_budget,
            'campaign_duration_days': campaign_duration_days,
            'budget_fit': budget_fit,
            'versatility_score': versatility_score,
            'success_rate': success_rate,
            'outcome': outcome
        })
    
    df = pd.DataFrame(data)
    
    print(f"✅ Generated {len(df)} samples")
    print(f"   Positive outcomes: {df['outcome'].sum()} ({df['outcome'].mean()*100:.1f}%)")
    print(f"   Negative outcomes: {(1-df['outcome']).sum()} ({(1-df['outcome'].mean())*100:.1f}%)")
    
    return df

def main():
    """Generate and save synthetic training data"""
    print("🚀 Starting Synthetic Data Generation...")
    
    # Generate data
    training_data = generate_training_data(n_samples=2000)
    
    # Save
    output_file = 'ai/data/training_data.csv'
    training_data.to_csv(output_file, index=False)
    
    print(f"💾 Saved training data to {output_file}")
    print(f"\n📊 Dataset Statistics:")
    print(training_data.describe())
    
    print("\n🎉 Synthetic data generation completed!")

if __name__ == "__main__":
    main()
