"""
AI Prediction Service - Integration Layer
Provides unified interface for all AI/ML models
Can be called from backend or as standalone service
"""
import os
import sys
import json
import numpy as np
from typing import Dict, Optional

# Add ai directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_matching import MatchingMLModel
from gemini_report import GeminiReportGenerator

try:
    from dl_analysis import DeepLearningAnalysis
    DL_AVAILABLE = True
except:
    DL_AVAILABLE = False
    print("⚠️  Deep Learning models not available")


class AIPredictionService:
    """
    Unified AI service for creator-campaign matching and analysis
    """
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        self.ml_model = None
        self.dl_model = None
        self.gemini = GeminiReportGenerator(api_key=gemini_api_key)
        self._load_models()
    
    def _load_models(self):
        """Load all AI models"""
        print("🤖 Loading AI models...")
        
        # Load ML models
        try:
            self.ml_model = MatchingMLModel()
            if self.ml_model.load_models():
                print("  ✅ ML models loaded")
            else:
                print("  ⚠️  ML models not found - predictions will use fallback")
                self.ml_model = None
        except Exception as e:
            print(f"  ⚠️  Could not load ML models: {e}")
            self.ml_model = None
        
        # Load DL models
        if DL_AVAILABLE:
            try:
                self.dl_model = DeepLearningAnalysis()
                if self.dl_model.load_model():
                    print("  ✅ DL models loaded")
                else:
                    print("  ⚠️  DL models not found - using ML only")
                    self.dl_model = None
            except Exception as e:
                print(f"  ⚠️  Could not load DL models: {e}")
                self.dl_model = None
    
    def _safe_int(self, value, default=0) -> int:
        """Safely convert value to int"""
        try:
            return int(value) if value is not None else default
        except (ValueError, TypeError):
            print(f"⚠️  Warning: Could not convert '{value}' to int, using default {default}")
            return default
    
    def _safe_float(self, value, default=0.0) -> float:
        """Safely convert value to float"""
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            print(f"⚠️  Warning: Could not convert '{value}' to float, using default {default}")
            return default
    
    def calculate_match_score(self, creator_data: Dict, campaign_data: Dict) -> float:
        """Calculate match score using rule-based + ML approach"""
        
        # Extract features
        features = self._extract_features(creator_data, campaign_data)
        
        # Use ML model if available
        if self.ml_model:
            try:
                predictions = self.ml_model.predict(features)
                return float(predictions['match_score'])
            except Exception as e:
                print(f"ML prediction failed: {e}, using fallback")
        
        # Fallback: rule-based scoring
        return self._calculate_rule_based_score(features)
    
    def _extract_features(self, creator_data: Dict, campaign_data: Dict) -> Dict:
        """Extract features for ML models"""
        
        # Category match
        creator_cats = creator_data.get('categories', [])
        campaign_cat = campaign_data.get('category', '')
        category_match = 1.0 if campaign_cat in creator_cats else 0.0
        
        # Requirements
        requirements = campaign_data.get('requirements', {})
        min_followers = requirements.get('min_followers', 0)
        min_engagement = requirements.get('min_engagement_rate', 0)
        
        # Ensure numeric types for calculations
        total_campaigns = creator_data.get('total_campaigns', 0)
        try:
            total_campaigns = int(total_campaigns) if total_campaigns is not None else 0
        except (ValueError, TypeError):
            total_campaigns = 0
            
        estimated_followers = creator_data.get('estimated_followers', total_campaigns * 10000)
        try:
            estimated_followers = float(estimated_followers) if estimated_followers is not None else (total_campaigns * 10000)
        except (ValueError, TypeError):
            estimated_followers = total_campaigns * 10000
            
        estimated_engagement = creator_data.get('estimated_engagement_rate', 0.05)
        try:
            estimated_engagement = float(estimated_engagement) if estimated_engagement is not None else 0.05
        except (ValueError, TypeError):
            estimated_engagement = 0.05
        
        followers_match = 1.0 if estimated_followers >= min_followers else 0.5
        engagement_match = 1.0 if estimated_engagement >= min_engagement else 0.5
        
        # Platform match
        platform_match = 0.8  # Simplified
        
        # Experience
        experience_score = min(5, max(1, 1 + total_campaigns // 10))
        
        # Rating - ensure it's a float
        overall_rating = creator_data.get('overall_rating', 3.0)
        try:
            overall_rating = float(overall_rating) if overall_rating is not None else 3.0
        except (ValueError, TypeError):
            overall_rating = 3.0
        
        # Budget fit
        budget = campaign_data.get('budget', 10000)
        try:
            budget = float(budget) if budget is not None else 10000.0
        except (ValueError, TypeError):
            budget = 10000.0
            
        estimated_cost = (estimated_followers / 1000) * overall_rating * 10
        budget_fit = min(2.0, budget / estimated_cost) if estimated_cost > 0 else 1.0
        
        # Other features
        num_categories = len(creator_cats) if creator_cats else 1
        num_languages = len(creator_data.get('languages', ['en']))
        account_age_days = creator_data.get('account_age_days', 365)
        success_rate = overall_rating / 5.0
        versatility_score = num_categories * num_languages * 0.1
        
        # Calculate campaign duration safely
        duration_days = campaign_data.get('duration_days', 30)
        if duration_days is None:
            # Calculate from start_date and end_date if available
            start_date = campaign_data.get('start_date')
            end_date = campaign_data.get('end_date')
            if start_date and end_date:
                try:
                    from datetime import datetime
                    if isinstance(start_date, str):
                        start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                    if isinstance(end_date, str):
                        end_date = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                    duration_days = (end_date - start_date).days
                except:
                    duration_days = 30
            else:
                duration_days = 30
        try:
            duration_days = int(duration_days) if duration_days is not None else 30
        except (ValueError, TypeError):
            duration_days = 30
        
        return {
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
            'estimated_engagement_rate': estimated_engagement,
            'campaign_budget': budget,
            'campaign_duration_days': duration_days,
            'budget_fit': budget_fit,
            'versatility_score': versatility_score,
            'success_rate': success_rate
        }
    
    def _calculate_rule_based_score(self, features: Dict) -> float:
        """Fallback rule-based scoring"""
        score = (
            features['category_match'] * 30 +
            features['followers_match'] * 15 +
            features['engagement_match'] * 10 +
            features['platform_match'] * 10 +
            features['experience_score'] * 4 +
            features['overall_rating'] * 3 +
            features['budget_fit'] * 10 +
            features['success_rate'] * 10
        )
        return min(100, max(0, score))
    
    def get_comprehensive_analysis(self, creator_data: Dict, campaign_data: Dict) -> Dict:
        """Get comprehensive AI-powered analysis"""
        print("🔍 Generating comprehensive AI analysis...")
        
        # Extract features
        features = self._extract_features(creator_data, campaign_data)
        
        # ML Predictions
        ml_predictions = {}
        if self.ml_model:
            try:
                ml_predictions = self.ml_model.predict(features)
            except Exception as e:
                print(f"ML prediction error: {e}")
                ml_predictions = {
                    'match_score': self._calculate_rule_based_score(features),
                    'estimated_roi': 100,
                    'estimated_engagement': 5.0
                }
        else:
            ml_predictions = {
                'match_score': self._calculate_rule_based_score(features),
                'estimated_roi': 100,
                'estimated_engagement': 5.0
            }
        
        # DL Predictions
        dl_predictions = {}
        if self.dl_model and DL_AVAILABLE:
            try:
                dl_predictions = self.dl_model.predict(features)
            except Exception as e:
                print(f"DL prediction error: {e}")
                dl_predictions = {
                    'success_probability': 0.7,
                    'match_score': ml_predictions['match_score'],
                    'predicted_engagement': ml_predictions['estimated_engagement']
                }
        else:
            # Fallback DL predictions
            match_score = ml_predictions['match_score']
            dl_predictions = {
                'success_probability': match_score / 120,  # 0-1 scale
                'match_score': match_score,
                'predicted_engagement': ml_predictions['estimated_engagement']
            }
        
        # Generate strengths and concerns
        strengths = self._generate_strengths(creator_data, features, ml_predictions)
        concerns = self._generate_concerns(creator_data, features, ml_predictions)
        reasons = self._generate_reasons(features)
        
        # Audience overlap
        audience_overlap = self._calculate_audience_overlap(creator_data, campaign_data)
        
        # Budget fit
        budget_fit = self._assess_budget_fit(features['budget_fit'])
        
        # Experience level
        experience_level = self._get_experience_level(
            self._safe_int(creator_data.get('total_campaigns', 0))
        )
        
        analysis = {
            'match_score': ml_predictions['match_score'],
            'ml_predictions': ml_predictions,
            'dl_predictions': dl_predictions,
            'strengths': strengths,
            'concerns': concerns,
            'reasons': reasons,
            'audience_overlap': audience_overlap,
            'budget_fit': budget_fit,
            'experience_level': experience_level,
            'features': features
        }
        
        print("✅ Analysis complete")
        return analysis
    
    def generate_ai_report(self, creator_data: Dict, campaign_data: Dict, 
                          analysis: Dict = None) -> Dict:
        """Generate AI-powered comprehensive report using Gemini"""
        print("📄 Generating AI report with Gemini...")
        
        if not analysis:
            analysis = self.get_comprehensive_analysis(creator_data, campaign_data)
        
        # Prepare data for Gemini
        report_data = {
            'creator_name': f"{creator_data.get('user', {}).get('first_name', '')} {creator_data.get('user', {}).get('last_name', '')}".strip(),
            'campaign_title': campaign_data.get('title', 'Campaign'),
            'match_score': analysis['match_score'],
            'ml_predictions': analysis['ml_predictions'],
            'dl_predictions': analysis['dl_predictions'],
            'creator_stats': {
                'total_campaigns': creator_data.get('total_campaigns', 0),
                'overall_rating': creator_data.get('overall_rating', 0),
                'categories': creator_data.get('categories', []),
                'estimated_followers': analysis['features']['estimated_followers'],
                'location': creator_data.get('location', 'N/A')
            },
            'campaign_details': {
                'platform': campaign_data.get('platform', 'N/A'),
                'category': campaign_data.get('category', 'N/A'),
                'budget': campaign_data.get('budget', 0),
                'duration_days': analysis['features']['campaign_duration_days']
            }
        }
        
        # Generate comprehensive report
        report = self.gemini.generate_comprehensive_report(report_data)
        
        # Add quick summary
        report['quick_summary'] = self.gemini.generate_quick_summary(report_data)
        
        # Add recommendations
        report['recommendations'] = self.gemini.generate_recommendations(report_data)
        
        # Add risk assessment
        report['risk_assessment'] = self.gemini.generate_risk_assessment(report_data)
        
        # Add analysis data
        report['analysis'] = analysis
        
        print("✅ AI report generated")
        return report
    
    def generate_creator_report(self, creator_data: Dict, campaign_data: Dict, 
                               collaboration_data: Dict = None) -> Dict:
        """Generate AI-powered report from CREATOR perspective using Gemini"""
        print("🎨 Generating creator-focused AI report with Gemini...")
        
        # Get analysis first
        analysis = self.get_comprehensive_analysis(creator_data, campaign_data)
        
        # Prepare data for Gemini - CREATOR PERSPECTIVE with FULL CONTEXT
        report_data = {
            'creator_name': f"{creator_data.get('user', {}).get('first_name', '')} {creator_data.get('user', {}).get('last_name', '')}".strip() or "You",
            'campaign_title': campaign_data.get('title', 'Campaign'),
            'brand_name': campaign_data.get('brand', {}).get('company_name', 'Brand'),
            'match_score': analysis['match_score'],
            'ml_predictions': analysis['ml_predictions'],
            'dl_predictions': analysis['dl_predictions'],
            
            # Creator statistics for context
            'creator_stats': {
                'total_campaigns': creator_data.get('total_campaigns', 0),
                'overall_rating': creator_data.get('overall_rating', 0),
                'categories': creator_data.get('categories', []),
                'languages': creator_data.get('languages', []),
                'estimated_followers': analysis['features'].get('estimated_followers', 0),
                'estimated_engagement_rate': analysis['features'].get('estimated_engagement_rate', 0),
                'location': creator_data.get('location', 'N/A'),
                'bio': creator_data.get('bio', ''),
                'experience_level': analysis.get('experience_level', 'Intermediate'),
            },
            
            # Complete campaign details
            'campaign_details': {
                'platform': campaign_data.get('platform', 'N/A'),
                'category': campaign_data.get('category', 'N/A'),
                'budget': campaign_data.get('budget', 0) if not collaboration_data else collaboration_data.get('proposed_budget', campaign_data.get('budget', 0)),
                'duration_days': analysis['features']['campaign_duration_days'],
                'start_date': campaign_data.get('start_date', 'N/A'),
                'end_date': campaign_data.get('end_date', 'N/A'),
                'brand_description': campaign_data.get('description', 'N/A'),
                'requirements': campaign_data.get('requirements', ''),
                'deliverables': campaign_data.get('deliverables', ''),
            },
            
            # Brand information
            'brand_details': {
                'company_name': campaign_data.get('brand', {}).get('company_name', 'Brand'),
                'description': campaign_data.get('brand', {}).get('description', 'N/A'),
                'website': campaign_data.get('brand', {}).get('website', 'N/A'),
                'industry': campaign_data.get('category', 'N/A'),
            },
            
            # Analysis features for deeper context
            'analysis_features': {
                'audience_overlap': self._safe_int(analysis['features'].get('category_match', 0) * 100),
                'budget_fit': analysis.get('budget_fit', 'Unknown'),
                'experience_score': analysis['features'].get('experience_score', 0),
                'versatility_score': analysis['features'].get('versatility_score', 0),
            },
            
            # Collaboration specific data (if exists)
            'collaboration_message': collaboration_data.get('message', '') if collaboration_data else '',
            'collaboration_deadline': collaboration_data.get('deadline', 'N/A') if collaboration_data else 'N/A',
        }
        
        # Generate creator-focused report using Gemini
        report = self.gemini.generate_creator_focused_report(report_data)
        
        # Add quick insights
        report['quick_insights'] = self.gemini.generate_creator_quick_insights(report_data)
        
        # Update analysis data with Gemini-generated strengths and concerns
        analysis['strengths'] = report.get('strengths', analysis['strengths'])
        analysis['concerns'] = report.get('concerns', analysis['concerns'])
        
        # Add analysis data
        report['analysis'] = analysis
        report['collaboration'] = collaboration_data
        
        print("✅ Creator-focused AI report generated")
        return report
    
    def _generate_strengths(self, creator_data: Dict, features: Dict, predictions: Dict) -> list:
        """Generate list of strengths"""
        strengths = []
        
        # Safe numeric conversions
        total_campaigns = self._safe_int(creator_data.get('total_campaigns', 0))
        overall_rating = self._safe_float(creator_data.get('overall_rating', 0))
        estimated_roi = self._safe_float(predictions.get('estimated_roi', 0))
        
        if total_campaigns >= 20:
            strengths.append(f"Proven track record with {total_campaigns} campaigns")
        
        if overall_rating >= 4.5:
            strengths.append(f"Excellent creator rating ({overall_rating}/5.0)")
        
        if features['category_match'] == 1.0:
            strengths.append("Perfect category match for this campaign")
        
        if features['estimated_followers'] >= 100000:
            strengths.append(f"Strong reach with {features['estimated_followers']:,} followers")
        
        if estimated_roi >= 150:
            strengths.append(f"High estimated ROI of {estimated_roi:.0f}%")
        
        if features['versatility_score'] >= 0.3:
            strengths.append("Versatile creator with multi-category expertise")
        
        return strengths[:5]  # Return top 5
    
    def _generate_concerns(self, creator_data: Dict, features: Dict, predictions: Dict) -> list:
        """Generate list of concerns"""
        concerns = []
        
        # Safe numeric conversions
        total_campaigns = self._safe_int(creator_data.get('total_campaigns', 0))
        overall_rating = self._safe_float(creator_data.get('overall_rating', 0))
        
        if features['budget_fit'] > 1.5:
            concerns.append("Creator may be premium-priced for this budget")
        
        if total_campaigns < 5:
            concerns.append("Limited campaign experience")
        
        if features['category_match'] < 1.0:
            concerns.append("Category alignment could be stronger")
        
        if features['followers_match'] < 1.0:
            concerns.append("May not meet follower requirements")
        
        if overall_rating < 4.0:
            concerns.append("Below-average creator rating")
        
        return concerns[:3]  # Return top 3
    
    def _generate_reasons(self, features: Dict) -> list:
        """Generate reasons for match"""
        reasons = []
        
        if features['category_match'] == 1.0:
            reasons.append("Perfect category alignment")
        
        if features['followers_match'] == 1.0:
            reasons.append("Meets follower requirements")
        
        if features['engagement_match'] == 1.0:
            reasons.append("Strong engagement rate")
        
        if features['experience_score'] >= 4:
            reasons.append("Highly experienced creator")
        
        if features['overall_rating'] >= 4.5:
            reasons.append("Excellent track record and ratings")
        
        if features['budget_fit'] <= 1.0 and features['budget_fit'] >= 0.8:
            reasons.append("Good budget fit")
        
        return reasons
    
    def _calculate_audience_overlap(self, creator_data: Dict, campaign_data: Dict) -> float:
        """Calculate audience overlap percentage"""
        overlap = 50  # Base overlap
        
        # Category match
        if campaign_data.get('category') in creator_data.get('categories', []):
            overlap += 20
        
        # Location match
        target_audience = campaign_data.get('target_audience', {})
        if creator_data.get('location') in target_audience.get('locations', []):
            overlap += 15
        
        # Additional factors
        overlap += np.random.randint(-10, 10)  # Random variance
        
        return max(0, min(100, overlap))
    
    def _assess_budget_fit(self, budget_fit_score: float) -> str:
        """Assess budget fit"""
        if budget_fit_score >= 1.2:
            return "Excellent Fit"
        elif budget_fit_score >= 0.8:
            return "Good Fit"
        elif budget_fit_score >= 0.5:
            return "Moderate Fit"
        else:
            return "Premium Option"
    
    def _get_experience_level(self, total_campaigns: int) -> str:
        """Get experience level"""
        if total_campaigns >= 50:
            return "Expert"
        elif total_campaigns >= 20:
            return "Advanced"
        elif total_campaigns >= 10:
            return "Intermediate"
        elif total_campaigns >= 1:
            return "Beginner"
        else:
            return "New"


# Singleton instance
_ai_service = None

def get_ai_service(gemini_api_key: str = None) -> AIPredictionService:
    """Get or create AI service instance"""
    global _ai_service
    if _ai_service is None:
        _ai_service = AIPredictionService(gemini_api_key=gemini_api_key)
    return _ai_service


if __name__ == "__main__":
    # Test the service
    print("🧪 Testing AI Prediction Service...\n")
    
    service = get_ai_service()
    
    # Sample data
    creator = {
        'id': 1,
        'user': {'first_name': 'John', 'last_name': 'Doe'},
        'categories': ['fashion', 'lifestyle'],
        'languages': ['en', 'es'],
        'total_campaigns': 25,
        'overall_rating': 4.8,
        'location': 'New York, USA',
        'estimated_followers': 250000
    }
    
    campaign = {
        'id': 1,
        'title': 'Summer Fashion 2024',
        'category': 'fashion',
        'platform': 'instagram',
        'budget': 8000,
        'duration_days': 30,
        'requirements': {
            'min_followers': 100000,
            'min_engagement_rate': 0.03
        },
        'target_audience': {
            'locations': ['New York, USA', 'Los Angeles, USA']
        }
    }
    
    # Get analysis
    analysis = service.get_comprehensive_analysis(creator, campaign)
    
    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)
    print(f"Match Score: {analysis['match_score']:.1f}/100")
    print(f"Estimated ROI: {analysis['ml_predictions']['estimated_roi']:.1f}%")
    print(f"Success Probability: {analysis['dl_predictions']['success_probability']*100:.1f}%")
    print(f"\nStrengths:")
    for s in analysis['strengths']:
        print(f"  ✅ {s}")
    print(f"\nConcerns:")
    for c in analysis['concerns']:
        print(f"  ⚠️  {c}")
    print("="*80)
