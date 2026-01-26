"""
LLM Integration for Enhanced Recommendations

This module integrates Large Language Models for:
1. Semantic understanding of creator content
2. Match explanation generation
3. Quality assessment
4. Risk identification
5. Recommendation justification

Using LLMs responsibly:
- LLM for explanation/enhancement, not core scoring
- Caching to reduce API calls
- Fallback when LLM unavailable
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib

from .entities import Creator, Campaign, MatchResult

logger = logging.getLogger(__name__)


class LLMExplainer:
    """
    LLM-powered explanation generator for match recommendations.
    
    This enhances the user experience by providing natural language
    explanations of why a creator is recommended for a campaign.
    """
    
    def __init__(self, model: str = "gemini-1.5-flash"):
        self.model = model
        self._cache = {}
        self._gemini_model = None
        
    def _init_gemini(self):
        """Initialize Gemini model"""
        if self._gemini_model is not None:
            return True
        
        try:
            import google.generativeai as genai
            
            api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
            if not api_key:
                logger.warning("Gemini API key not found")
                return False
            
            genai.configure(api_key=api_key)
            self._gemini_model = genai.GenerativeModel(self.model)
            return True
        except ImportError:
            logger.warning("google-generativeai not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            return False
    
    def generate_match_explanation(
        self,
        creator: Creator,
        campaign: Campaign,
        match_result: MatchResult,
        detailed: bool = False
    ) -> str:
        """
        Generate natural language explanation for a match.
        
        Args:
            creator: The matched creator
            campaign: The campaign
            match_result: The match result with scores
            detailed: Whether to generate detailed explanation
            
        Returns:
            Natural language explanation string
        """
        # Check cache
        cache_key = self._cache_key(creator.id, campaign.id, detailed)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Try LLM first
        if self._init_gemini():
            explanation = self._generate_llm_explanation(
                creator, campaign, match_result, detailed
            )
            if explanation:
                self._cache[cache_key] = explanation
                return explanation
        
        # Fallback to template-based explanation
        explanation = self._template_explanation(creator, campaign, match_result)
        self._cache[cache_key] = explanation
        return explanation
    
    def _generate_llm_explanation(
        self,
        creator: Creator,
        campaign: Campaign,
        match_result: MatchResult,
        detailed: bool
    ) -> Optional[str]:
        """Generate explanation using LLM"""
        try:
            prompt = self._build_explanation_prompt(
                creator, campaign, match_result, detailed
            )
            
            response = self._gemini_model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.7,
                    'max_output_tokens': 500 if detailed else 200,
                }
            )
            
            return response.text.strip()
        except Exception as e:
            logger.error(f"LLM explanation failed: {e}")
            return None
    
    def _build_explanation_prompt(
        self,
        creator: Creator,
        campaign: Campaign,
        match_result: MatchResult,
        detailed: bool
    ) -> str:
        """Build prompt for LLM explanation"""
        score_breakdown = match_result.score_breakdown
        
        prompt = f"""You are an expert marketing consultant explaining why an influencer matches a brand campaign.

CREATOR PROFILE:
- Name: {creator.name}
- Platform: {creator.platform}
- Followers: {creator.followers:,}
- Engagement Rate: {creator.engagement_rate:.2%}
- Categories: {', '.join(creator.categories)}
- Location: {creator.location}
- Experience: {creator.total_campaigns} campaigns, {creator.avg_campaign_rating:.1f}/5 avg rating

CAMPAIGN REQUIREMENTS:
- Brand: {campaign.brand_name}
- Campaign: {campaign.title}
- Platform: {campaign.platform}
- Categories: {', '.join(campaign.categories)}
- Budget: ${campaign.budget:,.0f}
- Target Followers: {campaign.min_followers:,} - {campaign.max_followers:,}
- Target Locations: {', '.join(campaign.target_locations)}

MATCH ANALYSIS:
- Overall Score: {match_result.final_score:.1f}/100
- Category Match: {score_breakdown.get('category_similarity', 0):.1%}
- Platform Match: {score_breakdown.get('platform_match', 0):.1%}
- Budget Fit: {score_breakdown.get('budget_fit', 0):.1%}
- Predicted Success: {match_result.predicted_success_probability:.1%}

{'Provide a detailed analysis including:' if detailed else 'Provide a brief analysis including:'}
1. Why this creator is a good match (2-3 key reasons)
2. Any potential concerns or risks (be honest)
3. Specific recommendations for the collaboration

{'Be thorough but concise. Focus on actionable insights.' if detailed else 'Keep it to 2-3 sentences per point.'}"""

        return prompt
    
    def _template_explanation(
        self,
        creator: Creator,
        campaign: Campaign,
        match_result: MatchResult
    ) -> str:
        """Generate template-based explanation as fallback"""
        parts = []
        
        # Opening
        score_desc = "excellent" if match_result.final_score >= 80 else \
                     "strong" if match_result.final_score >= 60 else \
                     "good" if match_result.final_score >= 40 else "moderate"
        
        parts.append(f"{creator.name} is a {score_desc} match for this campaign.")
        
        # Match reasons
        reasons = []
        
        # Category match
        if match_result.score_breakdown.get('category_similarity', 0) >= 0.7:
            common_cats = set(creator.categories) & set(campaign.categories)
            if common_cats:
                reasons.append(f"specializes in {', '.join(list(common_cats)[:2])}")
        
        # Engagement
        if creator.engagement_rate >= campaign.target_engagement_rate:
            reasons.append(f"has {creator.engagement_rate:.2%} engagement rate")
        
        # Experience
        if creator.total_campaigns >= 10 and creator.avg_campaign_rating >= 4:
            reasons.append(f"experienced creator with {creator.avg_campaign_rating:.1f}/5 rating")
        
        if reasons:
            parts.append("Key strengths: " + ", ".join(reasons) + ".")
        
        # Risks
        risks = []
        if creator.followers < campaign.min_followers:
            risks.append("follower count below target")
        if creator.avg_cost > campaign.budget * 0.8:
            risks.append("may consume most of budget")
        
        if risks:
            parts.append("Consider: " + ", ".join(risks) + ".")
        
        return " ".join(parts)
    
    def _cache_key(self, creator_id: str, campaign_id: str, detailed: bool) -> str:
        """Generate cache key for explanation"""
        key_str = f"{creator_id}:{campaign_id}:{detailed}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def clear_cache(self) -> None:
        """Clear explanation cache"""
        self._cache.clear()


class LLMQualityAssessor:
    """
    LLM-powered quality assessment for creator content.
    
    This analyzes creator content to assess:
    - Content quality and production value
    - Brand safety
    - Audience alignment
    """
    
    def __init__(self, model: str = "gemini-1.5-flash"):
        self.model = model
        self._gemini_model = None
        self._cache = {}
    
    def _init_gemini(self):
        """Initialize Gemini model"""
        if self._gemini_model is not None:
            return True
        
        try:
            import google.generativeai as genai
            
            api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
            if not api_key:
                return False
            
            genai.configure(api_key=api_key)
            self._gemini_model = genai.GenerativeModel(self.model)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            return False
    
    def assess_creator_quality(
        self,
        creator: Creator,
        content_samples: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Assess creator content quality.
        
        Args:
            creator: The creator to assess
            content_samples: Optional list of content captions/descriptions
            
        Returns:
            Dictionary with quality scores and insights
        """
        cache_key = f"quality_{creator.id}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if self._init_gemini() and content_samples:
            result = self._llm_quality_assessment(creator, content_samples)
            if result:
                self._cache[cache_key] = result
                return result
        
        # Fallback to heuristic assessment
        result = self._heuristic_quality_assessment(creator)
        self._cache[cache_key] = result
        return result
    
    def _llm_quality_assessment(
        self,
        creator: Creator,
        content_samples: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Use LLM for quality assessment"""
        try:
            samples_text = "\n".join([f"- {s[:200]}" for s in content_samples[:5]])
            
            prompt = f"""Analyze this influencer's content quality for brand partnerships.

CREATOR: {creator.name}
PLATFORM: {creator.platform}
CATEGORIES: {', '.join(creator.categories)}

CONTENT SAMPLES:
{samples_text}

Provide a JSON response with:
{{
    "content_quality": 0.0-1.0 (production value, creativity),
    "brand_safety": 0.0-1.0 (appropriateness for brands),
    "authenticity": 0.0-1.0 (genuine vs promotional feel),
    "engagement_quality": "high/medium/low" (comment quality, not just count),
    "content_style": ["list", "of", "style", "descriptors"],
    "brand_fit_categories": ["list", "of", "brand", "types"],
    "red_flags": ["any", "concerns"],
    "summary": "One sentence summary"
}}"""

            response = self._gemini_model.generate_content(
                prompt,
                generation_config={'temperature': 0.3}
            )
            
            # Parse JSON from response
            text = response.text
            # Extract JSON from markdown code block if present
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0]
            elif "```" in text:
                text = text.split("```")[1].split("```")[0]
            
            return json.loads(text.strip())
        except Exception as e:
            logger.error(f"LLM quality assessment failed: {e}")
            return None
    
    def _heuristic_quality_assessment(self, creator: Creator) -> Dict[str, Any]:
        """Heuristic-based quality assessment fallback"""
        return {
            'content_quality': creator.content_quality_score,
            'brand_safety': 0.8,  # Default assumption
            'authenticity': creator.audience_authenticity,
            'engagement_quality': 'medium' if creator.engagement_rate > 0.03 else 'low',
            'content_style': creator.categories[:3],
            'brand_fit_categories': creator.categories,
            'red_flags': [],
            'summary': f"{creator.name} is a {creator.tier} tier creator in {creator.categories[0] if creator.categories else 'general'} category."
        }


class LLMMatchEnhancer:
    """
    Use LLM to enhance match predictions and identify nuances.
    
    This goes beyond numerical scoring to understand:
    - Contextual fit
    - Cultural alignment
    - Creative synergy
    """
    
    def __init__(self, model: str = "gemini-1.5-flash"):
        self.model = model
        self._gemini_model = None
    
    def _init_gemini(self):
        """Initialize Gemini model"""
        if self._gemini_model is not None:
            return True
        
        try:
            import google.generativeai as genai
            
            api_key = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
            if not api_key:
                return False
            
            genai.configure(api_key=api_key)
            self._gemini_model = genai.GenerativeModel(self.model)
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            return False
    
    def enhance_match_result(
        self,
        creator: Creator,
        campaign: Campaign,
        match_result: MatchResult
    ) -> MatchResult:
        """
        Enhance match result with LLM insights.
        
        Adds:
        - Detailed match reasons
        - Risk identification
        - Collaboration recommendations
        - Natural language explanation
        """
        if not self._init_gemini():
            # Use template-based enhancement
            return self._template_enhancement(creator, campaign, match_result)
        
        try:
            enhanced = self._llm_enhancement(creator, campaign, match_result)
            return enhanced
        except Exception as e:
            logger.error(f"LLM enhancement failed: {e}")
            return self._template_enhancement(creator, campaign, match_result)
    
    def _llm_enhancement(
        self,
        creator: Creator,
        campaign: Campaign,
        match_result: MatchResult
    ) -> MatchResult:
        """Enhance match result using LLM"""
        prompt = f"""Analyze this influencer-campaign match and provide insights.

CREATOR:
- Name: {creator.name}
- Platform: {creator.platform}, {creator.followers:,} followers
- Categories: {', '.join(creator.categories)}
- Engagement: {creator.engagement_rate:.2%}
- Location: {creator.location}
- Past campaigns: {creator.total_campaigns}, Rating: {creator.avg_campaign_rating:.1f}/5

CAMPAIGN:
- Brand: {campaign.brand_name}
- Title: {campaign.title}
- Description: {campaign.description[:200] if campaign.description else 'N/A'}
- Categories: {', '.join(campaign.categories)}
- Budget: ${campaign.budget:,.0f}
- Platform: {campaign.platform}

INITIAL SCORE: {match_result.ranking_score:.1f}/100

Provide a JSON response:
{{
    "match_reasons": ["reason1", "reason2", "reason3"],
    "risks": ["risk1", "risk2"],
    "recommendations": ["rec1", "rec2"],
    "score_adjustment": -5 to +10 (adjustment to initial score based on qualitative factors),
    "explanation": "2-3 sentence natural language explanation"
}}"""

        response = self._gemini_model.generate_content(
            prompt,
            generation_config={'temperature': 0.5}
        )
        
        # Parse response
        text = response.text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        insights = json.loads(text.strip())
        
        # Update match result
        match_result.match_reasons = insights.get('match_reasons', [])
        match_result.risks = insights.get('risks', [])
        match_result.recommendations = insights.get('recommendations', [])
        match_result.llm_explanation = insights.get('explanation', '')
        
        # Apply score adjustment (bounded)
        adjustment = max(-5, min(10, insights.get('score_adjustment', 0)))
        match_result.final_score = min(100, max(0, match_result.ranking_score + adjustment))
        
        return match_result
    
    def _template_enhancement(
        self,
        creator: Creator,
        campaign: Campaign,
        match_result: MatchResult
    ) -> MatchResult:
        """Template-based match enhancement fallback"""
        # Generate match reasons
        reasons = []
        
        if set(creator.categories) & set(campaign.categories):
            reasons.append(f"Specializes in {', '.join(set(creator.categories) & set(campaign.categories))}")
        
        if creator.platform.lower() == campaign.platform.lower():
            reasons.append(f"Active on {campaign.platform}")
        
        if creator.engagement_rate >= campaign.target_engagement_rate:
            reasons.append(f"Strong {creator.engagement_rate:.1%} engagement rate")
        
        if creator.avg_campaign_rating >= 4:
            reasons.append(f"Highly rated ({creator.avg_campaign_rating:.1f}/5) by previous partners")
        
        match_result.match_reasons = reasons[:3]
        
        # Generate risks
        risks = []
        
        if creator.followers < campaign.min_followers:
            risks.append(f"Follower count ({creator.followers:,}) below target ({campaign.min_followers:,})")
        
        if creator.avg_cost > campaign.budget * 0.7:
            risks.append("May consume large portion of budget")
        
        if creator.completion_rate < 0.8:
            risks.append("Historical completion rate below 80%")
        
        match_result.risks = risks[:2]
        
        # Generate recommendations
        recs = []
        
        if creator.total_campaigns < 5:
            recs.append("Consider trial collaboration first")
        
        if creator.response_time_hours > 48:
            recs.append("Allow extra time for communication")
        
        recs.append("Review recent content for brand alignment")
        
        match_result.recommendations = recs[:2]
        
        # Set final score
        match_result.final_score = match_result.ranking_score
        
        return match_result
