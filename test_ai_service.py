"""
Test AI Service with actual creator and campaign data
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from ai_service import get_ai_service

def test_ai_service():
    """Test AI service with MrBeast creator"""
    print("🧪 Testing AI Service...\n")
    
    # Initialize service (will work without Gemini API key using fallbacks)
    service = get_ai_service(gemini_api_key=os.getenv('GEMINI_API_KEY', ''))
    
    # MrBeast Creator Data
    creator = {
        'id': '0718ef26-d10d-4168-9485-5de1157b20fd',
        'user': {
            'first_name': 'Jimmy',
            'last_name': 'Donaldson'
        },
        'categories': ['entertainment', 'challenges', 'philanthropy', 'gaming', 'food', 
                      'lifestyle', 'tech', 'fashion', 'travel', 'education'],
        'languages': ['en', 'es', 'pt', 'fr', 'de', 'ja', 'ko', 'hi'],
        'total_campaigns': 250,
        'overall_rating': 5.0,
        'location': 'North Carolina, USA',
        'bio': 'Philanthropist and content creator known for extreme challenges and giveaways',
        'estimated_followers': 423000000,  # Total across all platforms
        'estimated_engagement_rate': 0.08,  # 8% engagement
        'account_age_days': 3650  # ~10 years
    }
    
    # Sample Campaign Data
    campaign = {
        'id': '75c9de16-4f98-42d9-b0fa-23f8a8df0b54',
        'title': 'Tech Product Launch Campaign',
        'category': 'tech',
        'platform': 'youtube',
        'budget': 50000,
        'duration_days': 30,
        'requirements': {
            'min_followers': 1000000,
            'min_engagement_rate': 0.03
        },
        'target_audience': {
            'locations': ['United States', 'Canada', 'United Kingdom'],
            'age_range': '18-35',
            'gender': 'all'
        }
    }
    
    print("="*80)
    print("TEST 1: Comprehensive Analysis")
    print("="*80)
    
    # Get comprehensive analysis
    analysis = service.get_comprehensive_analysis(creator, campaign)
    
    print(f"\n✅ Match Score: {analysis['match_score']:.1f}/100")
    print(f"\n📊 ML Predictions:")
    print(f"   - Match Score: {analysis['ml_predictions']['match_score']:.1f}")
    print(f"   - Estimated ROI: {analysis['ml_predictions']['estimated_roi']:.1f}%")
    print(f"   - Estimated Engagement: {analysis['ml_predictions']['estimated_engagement']:.2f}%")
    
    print(f"\n🧠 DL Predictions:")
    print(f"   - Success Probability: {analysis['dl_predictions']['success_probability']*100:.1f}%")
    print(f"   - Match Score: {analysis['dl_predictions']['match_score']:.1f}")
    print(f"   - Predicted Engagement: {analysis['dl_predictions']['predicted_engagement']:.2f}%")
    
    print(f"\n💪 Strengths ({len(analysis['strengths'])}):")
    for i, strength in enumerate(analysis['strengths'], 1):
        print(f"   {i}. {strength}")
    
    print(f"\n⚠️  Concerns ({len(analysis['concerns'])}):")
    for i, concern in enumerate(analysis['concerns'], 1):
        print(f"   {i}. {concern}")
    
    print(f"\n🎯 Reasons ({len(analysis['reasons'])}):")
    for i, reason in enumerate(analysis['reasons'], 1):
        print(f"   {i}. {reason}")
    
    print(f"\n📈 Additional Metrics:")
    print(f"   - Audience Overlap: {analysis['audience_overlap']:.1f}%")
    print(f"   - Budget Fit: {analysis['budget_fit']}")
    print(f"   - Experience Level: {analysis['experience_level']}")
    
    print("\n" + "="*80)
    print("TEST 2: AI Report Generation")
    print("="*80)
    
    # Generate comprehensive report
    report = service.generate_ai_report(creator, campaign, analysis)
    
    print(f"\n📄 Report ID: {report['report_id']}")
    print(f"📅 Generated: {report['generated_at']}")
    
    print(f"\n📝 Quick Summary:")
    print(f"{report['quick_summary']}")
    
    print(f"\n💡 Recommendations ({len(report['recommendations'])}):")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    print(f"\n⚠️  Risk Assessment:")
    print(f"   Level: {report['risk_assessment']['risk_level']}")
    print(f"   Factors:")
    for factor in report['risk_assessment']['risk_factors']:
        print(f"      - {factor}")
    print(f"   Mitigation:")
    for strategy in report['risk_assessment']['mitigation_strategies']:
        print(f"      - {strategy}")
    
    print(f"\n📋 Full Report:")
    print("="*80)
    print(report['full_report'])
    print("="*80)
    
    print("\n✅ All tests completed successfully!")
    return analysis, report

if __name__ == "__main__":
    try:
        analysis, report = test_ai_service()
        print("\n🎉 AI Service is working correctly!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
