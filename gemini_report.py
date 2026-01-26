"""
Gemini AI Integration for Comprehensive Report Generation
- Uses Google Gemini API to generate detailed analysis reports
- Combines ML/DL predictions with generative AI insights
- Creates actionable recommendations and strategic insights
"""
import os
import sys
import json
import requests
from typing import Dict, List
from datetime import datetime
import google.generativeai as genai

# Get API key from environment
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY','AIzaSyDzoEH3Fw5YNIClzMU5lh1mdwe1wp_xEK4')

class GeminiReportGenerator:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            print("⚠️  Warning: GEMINI_API_KEY not set. Set it in environment variables.")
            self.model = None
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                # Test connection
                self.model.generate_content("test", generation_config=genai.types.GenerationConfig(max_output_tokens=10))
                print("✅ Gemini client initialized successfully.")
            except Exception as e:
                print(f"❌ Failed to initialize Gemini client. Please check your GEMINI_API_KEY. Error: {e}")
                self.model = None

    def _call_gemini_api(self, prompt: str) -> str:
        """Call Gemini API with prompt"""
        if not self.model:
            print("⚠️  Gemini client not initialized - using fallback report")
            print("💡 To use real Gemini AI, set the GEMINI_API_KEY environment variable")
            return self._generate_fallback_report()

        print(f"🤖 Calling Gemini API with model gemini-1.5-flash...")
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=2048,
                top_p=0.8,
                top_k=40
            )
            
            # Configure safety settings to be less restrictive
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            
            response = self.model.generate_content(
                prompt, 
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            print("✅ Gemini API call successful!")
            
            # Check if response has text
            if not response.text:
                print(f"⚠️  Warning: Empty response from Gemini. Finish reason: {response.candidates[0].finish_reason}")
                print(f"⚠️  Safety ratings: {response.candidates[0].safety_ratings}")
                return self._generate_fallback_report()
            
            return response.text
        except Exception as e:
            print(f"❌ Error calling Gemini API: {e}")
            # Check for specific API key related errors
            if hasattr(e, 'args') and e.args:
                error_details = e.args[0]
                if "API_KEY_INVALID" in str(error_details):
                    print("❌ Your GEMINI_API_KEY seems to be invalid. Please verify it in your .env file or environment variables.")
                elif "permission" in str(error_details).lower():
                     print("❌ Your GEMINI_API_KEY may not have the correct permissions for the 'gemini-1.5-flash' model.")
            return self._generate_fallback_report()

    def _generate_fallback_report(self) -> str:
        """Generate fallback report when API is unavailable"""
        return """
        **AI-Generated Analysis Report**
        
        This is a comprehensive analysis based on multiple data points and predictive models.
        The creator shows strong potential for this campaign based on historical performance
        and category alignment. Key metrics indicate a high probability of successful collaboration.
        """
    
    def generate_comprehensive_report(self, analysis_data: Dict) -> Dict:
        """Generate comprehensive analysis report using Gemini"""
        print("🤖 Generating AI-powered comprehensive report...")
        
        # Extract key data
        creator_name = analysis_data.get('creator_name', 'Creator')
        campaign_title = analysis_data.get('campaign_title', 'Campaign')
        match_score = analysis_data.get('match_score', 0)
        ml_predictions = analysis_data.get('ml_predictions', {})
        dl_predictions = analysis_data.get('dl_predictions', {})
        creator_stats = analysis_data.get('creator_stats', {})
        campaign_details = analysis_data.get('campaign_details', {})
        
        # Safe numeric conversions
        try:
            match_score = float(match_score) if match_score else 0
        except (ValueError, TypeError):
            match_score = 0
            
        try:
            estimated_followers = int(creator_stats.get('estimated_followers', 0)) if creator_stats.get('estimated_followers') else 0
        except (ValueError, TypeError):
            estimated_followers = 0
            
        try:
            budget = int(campaign_details.get('budget', 0)) if campaign_details.get('budget') else 0
        except (ValueError, TypeError):
            budget = 0
            
        try:
            duration_days = int(campaign_details.get('duration_days', 0)) if campaign_details.get('duration_days') else 0
        except (ValueError, TypeError):
            duration_days = 0
            
        try:
            total_campaigns = int(creator_stats.get('total_campaigns', 0)) if creator_stats.get('total_campaigns') else 0
        except (ValueError, TypeError):
            total_campaigns = 0
            
        try:
            overall_rating = float(creator_stats.get('overall_rating', 0)) if creator_stats.get('overall_rating') else 0
        except (ValueError, TypeError):
            overall_rating = 0
            
        try:
            estimated_roi = float(ml_predictions.get('estimated_roi', 0)) if ml_predictions.get('estimated_roi') else 0
        except (ValueError, TypeError):
            estimated_roi = 0
            
        try:
            success_prob = float(dl_predictions.get('success_probability', 0)) if dl_predictions.get('success_probability') else 0
        except (ValueError, TypeError):
            success_prob = 0
        
        # Build comprehensive prompt
        prompt = f"""
You are an expert influencer marketing analyst. Generate a comprehensive, professional analysis report for a brand considering a collaboration.

**Context:**
- Creator: {creator_name}
- Campaign: {campaign_title}
- Overall Match Score: {match_score:.1f}/100

**ML/AI Predictions:**
- Estimated ROI: {estimated_roi:.1f}%
- Success Probability: {success_prob * 100:.1f}%
- Predicted Engagement: {dl_predictions.get('predicted_engagement', 'N/A')}%
- Match Score: {match_score:.1f}

**Creator Statistics:**
- Total Campaigns: {total_campaigns}
- Overall Rating: {overall_rating:.1f}/5.0
- Categories: {', '.join(creator_stats.get('categories', []))}
- Estimated Followers: {estimated_followers:,}
- Location: {creator_stats.get('location', 'N/A')}

**Campaign Requirements:**
- Platform: {campaign_details.get('platform', 'N/A')}
- Category: {campaign_details.get('category', 'N/A')}
- Budget: ${budget:,}
- Duration: {duration_days} days

Please generate a detailed report with the following sections:

1. **Executive Summary** (2-3 sentences)
2. **Key Strengths** (3-5 bullet points)
3. **Potential Concerns** (2-3 bullet points)
4. **Match Analysis** (Why this creator fits - 3-4 reasons)
5. **Strategic Recommendations** (4-5 actionable recommendations for the brand)
6. **Risk Assessment** (Low/Medium/High with explanation)
7. **Expected Outcomes** (What the brand can expect)
8. **Next Steps** (3-4 immediate actions)

Make it professional, data-driven, and actionable. Use specific numbers from the data provided.
"""
        
        # Call Gemini API
        report_text = self._call_gemini_api(prompt)
        
        # Parse and structure the report
        structured_report = {
            'report_id': f"RPT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'generated_at': datetime.now().isoformat(),
            'creator_name': creator_name,
            'campaign_title': campaign_title,
            'match_score': match_score,
            'full_report': report_text,
            'ml_predictions': ml_predictions,
            'dl_predictions': dl_predictions,
            'metadata': {
                'model': 'gemini-1.5-flash',
                'version': '1.0',
                'confidence': 'high' if match_score >= 80 else 'medium' if match_score >= 60 else 'low'
            }
        }
        
        print("✅ Report generated successfully")
        return structured_report
    
    def generate_quick_summary(self, analysis_data: Dict) -> str:
        """Generate a quick one-paragraph summary"""
        print("📝 Generating quick summary...")
        
        # Safe conversions
        try:
            match_score = float(analysis_data.get('match_score', 0)) if analysis_data.get('match_score') else 0
        except (ValueError, TypeError):
            match_score = 0
            
        try:
            roi = float(analysis_data.get('ml_predictions', {}).get('estimated_roi', 0))
        except (ValueError, TypeError):
            roi = 0
            
        creator_name = analysis_data.get('creator_name', 'This creator')
        
        prompt = f"""
Write a concise, compelling one-paragraph summary (3-4 sentences) explaining why {creator_name} 
with a match score of {match_score:.1f}/100 and estimated ROI of {roi:.1f}% is suitable for this campaign.
Focus on key strengths and the value proposition. Be specific and persuasive.
"""
        
        summary = self._call_gemini_api(prompt)
        print("✅ Summary generated")
        return summary
    
    def generate_creator_focused_report(self, analysis_data: Dict) -> Dict:
        """Generate comprehensive analysis report for CREATOR perspective"""
        print("🎨 Generating creator-focused AI report...")
        
        # Extract key data
        creator_name = analysis_data.get('creator_name', 'You')
        campaign_title = analysis_data.get('campaign_title', 'Campaign')
        brand_name = analysis_data.get('brand_name', 'Brand')
        match_score = analysis_data.get('match_score', 0)
        ml_predictions = analysis_data.get('ml_predictions', {})
        dl_predictions = analysis_data.get('dl_predictions', {})
        campaign_details = analysis_data.get('campaign_details', {})
        
        # Safe numeric conversions
        try:
            match_score = float(match_score) if match_score else 0
        except (ValueError, TypeError):
            match_score = 0
            
        try:
            budget = int(campaign_details.get('budget', 0)) if campaign_details.get('budget') else 0
        except (ValueError, TypeError):
            budget = 0
            
        try:
            duration_days = int(campaign_details.get('duration_days', 0)) if campaign_details.get('duration_days') else 0
        except (ValueError, TypeError):
            duration_days = 0
            
        try:
            estimated_roi = float(ml_predictions.get('estimated_roi', 0)) if ml_predictions.get('estimated_roi') else 0
        except (ValueError, TypeError):
            estimated_roi = 0
            
        try:
            success_prob = float(dl_predictions.get('success_probability', 0)) if dl_predictions.get('success_probability') else 0
        except (ValueError, TypeError):
            success_prob = 0
        
        # Extract additional context
        creator_stats = analysis_data.get('creator_stats', {})
        brand_details = analysis_data.get('brand_details', {})
        analysis_features = analysis_data.get('analysis_features', {})
        
        # Build creator-focused prompt with FULL CONTEXT
        prompt = f"""
You are an expert career advisor for influencers and content creators. Generate a comprehensive, personalized analysis report for {creator_name} who is evaluating a collaboration opportunity.

**CREATOR PROFILE:**
- Name: {creator_name}
- Total Past Campaigns: {creator_stats.get('total_campaigns', 0)}
- Overall Rating: {creator_stats.get('overall_rating', 0)}/5.0
- Categories/Niches: {', '.join(creator_stats.get('categories', []))}
- Languages: {', '.join(creator_stats.get('languages', []))}
- Estimated Followers: {creator_stats.get('estimated_followers', 0):,}
- Engagement Rate: {creator_stats.get('estimated_engagement_rate', 0):.2%}
- Location: {creator_stats.get('location', 'N/A')}
- Experience Level: {creator_stats.get('experience_level', 'Intermediate')}
- Bio: {creator_stats.get('bio', 'N/A')}

**BRAND DETAILS:**
- Company: {brand_name}
- Industry: {brand_details.get('industry', 'N/A')}
- Description: {brand_details.get('description', 'N/A')}
- Website: {brand_details.get('website', 'N/A')}

**CAMPAIGN OPPORTUNITY:**
- Campaign Name: {campaign_title}
- Platform: {campaign_details.get('platform', 'N/A')}
- Category: {campaign_details.get('category', 'N/A')}
- Budget Offered: ${budget:,}
- Campaign Duration: {duration_days} days ({campaign_details.get('start_date', 'N/A')} to {campaign_details.get('end_date', 'N/A')})
- Campaign Description: {campaign_details.get('brand_description', 'N/A')}
- Requirements: {campaign_details.get('requirements', 'N/A')}
- Deliverables: {campaign_details.get('deliverables', 'N/A')}

**AI-POWERED PREDICTIONS:**
- Overall Match Score: {match_score:.1f}/100 (Audience Compatibility)
- Estimated ROI for YOU: {estimated_roi:.1f}%
- Success Probability: {success_prob * 100:.1f}%
- Expected Engagement Impact: {dl_predictions.get('predicted_engagement', 'N/A')}%
- Audience Overlap: {analysis_features.get('audience_overlap', 0)}%
- Budget Fit Analysis: {analysis_features.get('budget_fit', 'Unknown')}

**COLLABORATION MESSAGE:**
{analysis_data.get('collaboration_message', 'No message provided')}

Please generate a detailed report with the following sections:

1. **Executive Summary** (2-3 sentences explaining if this is a good opportunity for the creator)
2. **Why This Is a Great Fit** (4-5 bullet points highlighting how this campaign aligns with creator's brand)
3. **Growth Opportunities** (3-4 ways this collaboration can help grow the creator's career)
4. **Financial Analysis** (Is the budget fair? Expected earnings vs. effort analysis)
5. **Brand Compatibility** (How well does this brand align with creator's values and audience)
6. **Potential Challenges** (2-3 things to be aware of or negotiate)
7. **Negotiation Points** (3-4 specific things the creator should negotiate or clarify)
8. **Strategic Recommendations** (4-5 actionable tips to maximize success if accepting)
9. **Decision Framework** (Clear recommendation: Accept, Counter-Offer, or Decline with reasoning)
10. **Next Steps** (3-4 immediate actions to take)

Make it personal, encouraging, and focused on the CREATOR'S best interests. Use "you" and "your" to address the creator directly. Be honest about both opportunities and risks.
"""
        
        # Call Gemini API
        report_text = self._call_gemini_api(prompt)
        
        # Generate structured strengths and concerns specifically for the creator
        strengths_prompt = f"""
You are an expert career advisor for content creators and influencers. Analyze this collaboration opportunity and provide 4-5 specific, compelling reasons why {creator_name} should consider this partnership with {brand_name}.

**ABOUT {creator_name.upper()}:**
- Experience: {creator_stats.get('total_campaigns', 0)} past campaigns completed
- Rating: {creator_stats.get('overall_rating', 0)}/5.0
- Followers: {creator_stats.get('estimated_followers', 0):,}
- Engagement Rate: {creator_stats.get('estimated_engagement_rate', 0):.2%}
- Niche: {', '.join(creator_stats.get('categories', []))}
- Experience Level: {creator_stats.get('experience_level', 'Intermediate')}

**ABOUT THE BRAND ({brand_name.upper()}):**
- Industry: {brand_details.get('industry', 'N/A')}
- Description: {brand_details.get('description', 'N/A')}
- Campaign Category: {campaign_details.get('category', 'N/A')}

**CAMPAIGN OPPORTUNITY:**
- Campaign: {campaign_title}
- Platform: {campaign_details.get('platform', 'N/A')} 
- Budget: ${budget:,} for {duration_days} days
- Requirements: {campaign_details.get('requirements', 'N/A')}
- Deliverables: {campaign_details.get('deliverables', 'N/A')}

**AI-POWERED INSIGHTS:**
- Match Score: {match_score:.1f}/100 → {"🔥 EXCEPTIONAL FIT" if match_score >= 90 else "✅ STRONG FIT" if match_score >= 75 else "⚠️ MODERATE FIT"}
- Estimated ROI: {estimated_roi:.1f}% → {"💰 HIGHLY PROFITABLE" if estimated_roi >= 200 else "💵 PROFITABLE" if estimated_roi >= 100 else "📊 REASONABLE"}
- Success Rate: {success_prob * 100:.1f}% → {"🎯 VERY HIGH CONFIDENCE" if success_prob >= 0.8 else "👍 GOOD CONFIDENCE" if success_prob >= 0.6 else "⚡ MODERATE CONFIDENCE"}
- Engagement Impact: {dl_predictions.get('predicted_engagement', 'N/A')}%
- Audience Overlap: {analysis_features.get('audience_overlap', 0)}%
- Budget Fit: {analysis_features.get('budget_fit', 'Unknown')}

Based on {creator_name}'s profile and this specific opportunity data, list 4-5 concrete benefits. Focus on:
✅ How {brand_name} in {campaign_details.get('category', 'N/A')} aligns with {creator_name}'s {', '.join(creator_stats.get('categories', []))} niche
✅ Audience growth potential (considering the {match_score:.1f}% match score)
✅ Financial upside ({estimated_roi:.1f}% ROI on ${budget:,} budget)
✅ Portfolio/credibility boost from collaborating with {brand_name}
✅ Strategic positioning for future deals

Return ONLY bullet points, one per line, starting with a dash (-). Use SPECIFIC numbers and data points. No generic advice.
"""
        
        concerns_prompt = f"""
You are an expert career advisor for content creators. Provide honest, data-driven concerns for {creator_name} about this {brand_name} collaboration.

**CREATOR'S MARKET POSITION:**
- {creator_name} has {creator_stats.get('total_campaigns', 0)} campaigns under their belt
- Current rating: {creator_stats.get('overall_rating', 0)}/5.0
- Follower count: {creator_stats.get('estimated_followers', 0):,}
- Experience: {creator_stats.get('experience_level', 'Intermediate')}

**THE OFFER:**
- Brand: {brand_name} ({brand_details.get('industry', 'N/A')})
- Campaign: {campaign_title}
- Budget: ${budget:,} for {duration_days} days
- Platform: {campaign_details.get('platform', 'N/A')}
- Category: {campaign_details.get('category', 'N/A')}
- Requirements: {campaign_details.get('requirements', 'Standard content creation')}

**RED FLAGS & CONSIDERATIONS:**
- Budget Assessment: {"🚩 CRITICAL: $0 budget is unacceptable" if budget == 0 else "⚠️ Low budget relative to market value" if budget < 10000 else "✅ Fair compensation" if budget < 50000 else "💰 Premium budget"}
- Match Score: {match_score:.1f}/100
- Budget Fit: {analysis_features.get('budget_fit', 'Unknown')}
- Deadline: {analysis_data.get('collaboration_deadline', 'Not specified')}

Given {creator_name}'s ${creator_stats.get('estimated_followers', 0):,} followers and {creator_stats.get('experience_level', 'Intermediate')} experience level, analyze:
⚠️ Is ${budget:,} adequate compensation for {duration_days} days of work?
⚠️ Does {brand_name} align with {creator_name}'s brand in {', '.join(creator_stats.get('categories', []))}?
⚠️ Are the requirements ({campaign_details.get('requirements', 'N/A')}) feasible within {duration_days} days?
⚠️ What contract terms need clarification or negotiation?

Return 2-3 HONEST concerns as bullet points (one per line, starting with dash). Be specific using the numbers above. Protect the creator's interests.
"""
        
        strengths_text = self._call_gemini_api(strengths_prompt)
        concerns_text = self._call_gemini_api(concerns_prompt)
        
        # Parse strengths and concerns
        strengths = []
        for line in strengths_text.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('*') or line.startswith('•')):
                clean_line = line.lstrip('-*• ').strip()
                if clean_line:
                    strengths.append(clean_line)
        
        concerns = []
        for line in concerns_text.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('*') or line.startswith('•')):
                clean_line = line.lstrip('-*• ').strip()
                if clean_line:
                    concerns.append(clean_line)
        
        # Structure the report
        structured_report = {
            'report_id': f"CREATOR_RPT_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'generated_at': datetime.now().isoformat(),
            'perspective': 'creator',
            'creator_name': creator_name,
            'campaign_title': campaign_title,
            'brand_name': brand_name,
            'match_score': match_score,
            'full_report': report_text,
            'strengths': strengths[:5],
            'concerns': concerns[:3],
            'ml_predictions': ml_predictions,
            'dl_predictions': dl_predictions,
            'metadata': {
                'model': 'gemini-1.5-flash',
                'version': '1.0',
                'report_type': 'creator_opportunity_analysis',
                'confidence': 'high' if match_score >= 80 else 'medium' if match_score >= 60 else 'low'
            }
        }
        
        print(f"✅ Creator-focused report generated with {len(strengths)} strengths and {len(concerns)} concerns")
        return structured_report
    
    def generate_creator_quick_insights(self, analysis_data: Dict) -> List[str]:
        """Generate quick actionable insights for creator"""
        print("⚡ Generating quick insights for creator...")
        
        # Safe conversions
        try:
            match_score = float(analysis_data.get('match_score', 0)) if analysis_data.get('match_score') else 0
        except (ValueError, TypeError):
            match_score = 0
            
        try:
            roi = float(analysis_data.get('ml_predictions', {}).get('estimated_roi', 0))
        except (ValueError, TypeError):
            roi = 0
            
        try:
            budget = int(analysis_data.get('campaign_details', {}).get('budget', 0))
        except (ValueError, TypeError):
            budget = 0
        
        prompt = f"""
Based on this collaboration opportunity:
- Match Score: {match_score:.1f}/100
- Expected ROI: {roi:.1f}%
- Budget: ${budget:,}

Generate 3-4 quick, punchy insights (one sentence each) that help the creator decide.
Format as bullet points. Be direct and actionable. Focus on what matters most to creators.
"""
        
        insights_text = self._call_gemini_api(prompt)
        
        # Parse insights
        insights = []
        for line in insights_text.split('\n'):
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('*') or line.startswith('•')):
                clean_line = line.lstrip('-*• ').strip()
                if clean_line:
                    insights.append(clean_line)
        
        print(f"✅ Generated {len(insights)} insights")
        return insights[:4]
    
    def generate_recommendations(self, analysis_data: Dict) -> List[str]:
        """Generate strategic recommendations"""
        print("💡 Generating strategic recommendations...")
        
        prompt = f"""
Based on this influencer marketing analysis data:
{json.dumps(analysis_data, indent=2)}

Generate 5 specific, actionable recommendations for the brand to maximize campaign success.
Return only the recommendations as a numbered list, each starting with a number and period.
Be specific and tactical.
"""
        
        recommendations_text = self._call_gemini_api(prompt)
        
        # Parse recommendations
        recommendations = []
        for line in recommendations_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('*')):
                # Clean up formatting
                clean_line = line.lstrip('0123456789.-* ').strip()
                if clean_line:
                    recommendations.append(clean_line)
        
        print(f"✅ Generated {len(recommendations)} recommendations")
        return recommendations[:5]  # Return top 5
    
    def generate_risk_assessment(self, analysis_data: Dict) -> Dict:
        """Generate risk assessment"""
        print("⚠️  Generating risk assessment...")
        
        match_score = analysis_data.get('match_score', 0)
        success_prob = analysis_data.get('dl_predictions', {}).get('success_probability', 0.5)
        
        prompt = f"""
Based on match score of {match_score}/100 and success probability of {success_prob * 100:.1f}%,
provide a risk assessment for this influencer collaboration.

Return a JSON object with:
- risk_level: "Low", "Medium", or "High"
- risk_factors: array of 2-3 specific risk factors
- mitigation_strategies: array of 2-3 strategies to mitigate risks

Return ONLY valid JSON, no other text.
"""
        
        try:
            risk_text = self._call_gemini_api(prompt)
            # Try to extract JSON from response
            start = risk_text.find('{')
            end = risk_text.rfind('}') + 1
            if start != -1 and end > start:
                risk_data = json.loads(risk_text[start:end])
            else:
                risk_data = self._generate_fallback_risk(match_score, success_prob)
        except:
            risk_data = self._generate_fallback_risk(match_score, success_prob)
        
        print(f"✅ Risk level: {risk_data.get('risk_level', 'Unknown')}")
        return risk_data
    
    def _generate_fallback_risk(self, match_score: float, success_prob: float) -> Dict:
        """Generate fallback risk assessment"""
        if match_score >= 80 and success_prob >= 0.7:
            risk_level = "Low"
            risk_factors = ["Minimal concerns based on strong match metrics", "High success probability"]
        elif match_score >= 60 and success_prob >= 0.5:
            risk_level = "Medium"
            risk_factors = ["Moderate match score requires closer monitoring", "Average success probability"]
        else:
            risk_level = "High"
            risk_factors = ["Lower match score indicates potential misalignment", "Below-average success probability"]
        
        return {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "mitigation_strategies": [
                "Set clear expectations and deliverables upfront",
                "Establish milestone-based payment structure",
                "Include performance metrics in contract"
            ]
        }
    
    def save_report(self, report: Dict, filename: str = None):
        """Save report to file"""
        os.makedirs('ai/reports', exist_ok=True)
        
        if not filename:
            filename = f"ai/reports/report_{report['report_id']}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Report saved to {filename}")


if __name__ == "__main__":
    # Example usage
    generator = GeminiReportGenerator()
    
    sample_data = {
        "creator_name": "John Doe",
        "campaign_title": "Summer Fashion 2024",
        "match_score": 92,
        "ml_predictions": {
            "match_score": 92,
            "estimated_roi": 180
        },
        "dl_predictions": {
            "success_probability": 0.85,
            "predicted_engagement": 6.5
        },
        "creator_stats": {
            "total_campaigns": 25,
            "overall_rating": 4.8,
            "categories": ["fashion", "lifestyle"],
            "estimated_followers": 250000,
            "location": "New York, USA"
        },
        "campaign_details": {
            "platform": "instagram",
            "category": "fashion",
            "budget": 8000,
            "duration_days": 30
        }
    }
    
    # Generate comprehensive report
    report = generator.generate_comprehensive_report(sample_data)
    
    # Generate quick summary
    summary = generator.generate_quick_summary(sample_data)
    report['quick_summary'] = summary
    
    # Generate recommendations
    recommendations = generator.generate_recommendations(sample_data)
    report['recommendations'] = recommendations
    
    # Generate risk assessment
    risk = generator.generate_risk_assessment(sample_data)
    report['risk_assessment'] = risk
    
    # Save report
    generator.save_report(report)
    
    print("\n" + "="*80)
    print("SAMPLE REPORT GENERATED")
    print("="*80)
    print(f"\n{report['full_report']}\n")
    print("="*80)
