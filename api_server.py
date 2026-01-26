"""
AI/ML Microservice API Server
Flask REST API for AI predictions and analysis
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from ai_service import AIPredictionService
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for Node.js backend

# Initialize AI service (loads models once on startup)
logger.info("🚀 Initializing AI Service...")
ai_service = AIPredictionService()
logger.info("✅ AI Service initialized successfully")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'AI/ML Microservice',
        'models_loaded': ai_service.ml_model is not None
    })


@app.route('/api/analyze', methods=['POST'])
def analyze():
    """
    Get comprehensive AI analysis for creator-campaign match
    
    Request body:
    {
        "creator": { ... },
        "campaign": { ... }
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'creator' not in data or 'campaign' not in data:
            return jsonify({
                'error': 'Missing creator or campaign data'
            }), 400
        
        creator = data['creator']
        campaign = data['campaign']
        
        logger.info(f"🔍 Analyzing creator {creator.get('id')} for campaign {campaign.get('id')}")
        
        # Get comprehensive analysis
        analysis = ai_service.get_comprehensive_analysis(creator, campaign)
        
        logger.info(f"✅ Analysis complete - Match Score: {analysis.get('match_score', 0)}")
        
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"❌ Error in analysis: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    """
    Generate AI-powered comprehensive report
    
    Request body:
    {
        "creator": { ... },
        "campaign": { ... }
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'creator' not in data or 'campaign' not in data:
            return jsonify({
                'error': 'Missing creator or campaign data'
            }), 400
        
        creator = data['creator']
        campaign = data['campaign']
        
        logger.info(f"📄 Generating report for creator {creator.get('id')} and campaign {campaign.get('id')}")
        
        # Generate comprehensive report
        report = ai_service.generate_ai_report(creator, campaign)
        
        logger.info(f"✅ Report generated - ID: {report.get('report_id')}")
        
        return jsonify(report)
        
    except Exception as e:
        logger.error(f"❌ Error generating report: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/generate-creator-report', methods=['POST'])
def generate_creator_report():
    """
    Generate AI-powered report from CREATOR perspective
    
    Request body:
    {
        "creator": { ... },
        "campaign": { ... },
        "collaboration": { ... }
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'creator' not in data or 'campaign' not in data:
            return jsonify({
                'error': 'Missing creator or campaign data'
            }), 400
        
        creator = data['creator']
        campaign = data['campaign']
        collaboration = data.get('collaboration', {})
        
        logger.info(f"🎨 Generating creator-focused report for {creator.get('id')}")
        
        # Generate creator-focused report
        report = ai_service.generate_creator_report(creator, campaign, collaboration)
        
        logger.info(f"✅ Creator report generated - ID: {report.get('report_id')}")
        
        return jsonify(report)
        
    except Exception as e:
        logger.error(f"❌ Error generating creator report: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/api/match-score', methods=['POST'])
def match_score():
    """
    Calculate quick match score only
    
    Request body:
    {
        "creator": { ... },
        "campaign": { ... }
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'creator' not in data or 'campaign' not in data:
            return jsonify({
                'error': 'Missing creator or campaign data'
            }), 400
        
        creator = data['creator']
        campaign = data['campaign']
        
        # Calculate match score
        score = ai_service.calculate_match_score(creator, campaign)
        
        return jsonify({
            'match_score': score,
            'creator_id': creator.get('id'),
            'campaign_id': campaign.get('id')
        })
        
    except Exception as e:
        logger.error(f"❌ Error calculating match score: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/train', methods=['POST'])
def train_models():
    """
    Trigger model retraining (admin only in production)
    
    Request body:
    {
        "training_data_path": "path/to/data.csv"
    }
    """
    try:
        data = request.get_json()
        training_data_path = data.get('training_data_path', 'ai/data/training_data.csv')
        
        logger.info(f"🎓 Starting model training with data: {training_data_path}")
        
        # Train models
        if ai_service.ml_model:
            ai_service.ml_model.train_models(training_data_path)
        
        logger.info("✅ Model training complete")
        
        return jsonify({
            'status': 'success',
            'message': 'Models trained successfully',
            'models_saved': [
                'ai/models/match_score_model.pkl',
                'ai/models/roi_model.pkl',
                'ai/models/engagement_model.pkl',
                'ai/models/scaler.pkl'
            ]
        })
        
    except Exception as e:
        logger.error(f"❌ Error training models: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500


if __name__ == '__main__':
    import sys
    
    # Get port from command line args or use default
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5001
    
    logger.info(f"🚀 Starting AI/ML Microservice on port {port}")
    logger.info(f"📊 Models loaded: {ai_service.ml_model is not None}")
    logger.info(f"🌐 Health check: http://localhost:{port}/health")
    logger.info(f"🔍 Analysis endpoint: http://localhost:{port}/api/analyze")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,  # Set to True for development
        threaded=True  # Handle multiple requests concurrently
    )
