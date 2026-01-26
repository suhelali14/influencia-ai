"""
Production ML Inference API
FastAPI service with caching, monitoring, and batch prediction

Features:
- Redis caching for fast repeated predictions
- Prometheus metrics for monitoring
- Batch prediction endpoint
- Model explanation endpoint
- Health checks
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import time
import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.ensemble import LightweightEnsemble, EnsemblePredictor

# Prometheus metrics
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Redis for caching (optional - gracefully handle if not available)
try:
    import redis
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    REDIS_AVAILABLE = True
    print("✓ Redis connected")
except:
    REDIS_AVAILABLE = False
    print("⚠ Redis not available - caching disabled")

# Initialize FastAPI app
app = FastAPI(
    title="Influencia ML API",
    description="Production ML inference service for creator-campaign matching",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics - avoid duplicate registration
try:
    from prometheus_client import REGISTRY
    # Check if already registered
    if 'predictions_total' not in REGISTRY._names_to_collectors:
        prediction_counter = Counter('predictions_total', 'Total predictions made')
        prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
        cache_hit_counter = Counter('cache_hits_total', 'Total cache hits')
        cache_miss_counter = Counter('cache_misses_total', 'Total cache misses')
    else:
        # Get existing metrics
        prediction_counter = REGISTRY._names_to_collectors['predictions_total']
        prediction_latency = REGISTRY._names_to_collectors['prediction_latency_seconds']
        cache_hit_counter = REGISTRY._names_to_collectors['cache_hits_total']
        cache_miss_counter = REGISTRY._names_to_collectors['cache_misses_total']
except Exception as e:
    print(f"⚠ Prometheus metrics setup failed: {e}")
    # Create dummy metrics
    class DummyMetric:
        def inc(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
    prediction_counter = DummyMetric()
    prediction_latency = DummyMetric()
    cache_hit_counter = DummyMetric()
    cache_miss_counter = DummyMetric()

# Load model
print("Loading ML models...")
ensemble = LightweightEnsemble()
print("✓ Models loaded successfully")

# ========== Request/Response Models ==========

class CreatorProfile(BaseModel):
    creator_id: int
    bio: Optional[str] = ""
    categories: List[str] = Field(default_factory=list)
    platforms: List[str] = Field(default_factory=list)
    followers: int
    engagement_rate: float
    tier: str = "micro"
    total_campaigns: int = 0
    successful_campaigns: int = 0
    success_rate: float = 0.0
    overall_rating: float = 0.0
    total_earnings: float = 0.0
    audience_age_18_24: float = 0.0
    audience_age_25_34: float = 0.0
    audience_female_pct: float = 50.0


class CampaignDetails(BaseModel):
    campaign_id: int
    title: str
    description: Optional[str] = ""
    category: str
    platform: str
    industry: Optional[str] = ""
    budget: float
    duration_days: int
    deliverables: List[str] = Field(default_factory=list)
    min_followers: int
    min_engagement: float
    target_age_group: Optional[str] = ""
    target_gender: str = "All"


class BrandInfo(BaseModel):
    company_name: str
    description: Optional[str] = ""
    industry: Optional[str] = ""
    website: Optional[str] = ""


class PredictionRequest(BaseModel):
    creator: CreatorProfile
    campaign: CampaignDetails
    brand: Optional[BrandInfo] = None
    include_explanation: bool = False


class BatchPredictionRequest(BaseModel):
    campaign: CampaignDetails
    creators: List[CreatorProfile]
    top_k: int = 20
    brand: Optional[BrandInfo] = None


class PredictionResponse(BaseModel):
    match_score: float
    confidence: float
    processing_time_ms: float
    cached: bool = False
    explanation: Optional[Dict] = None


class BatchPredictionResponse(BaseModel):
    campaign_id: int
    total_creators: int
    top_matches: List[Dict]
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool
    cache_available: bool


# ========== API Endpoints ==========

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint"""
    return {
        "service": "Influencia ML API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=True,
        cache_available=REDIS_AVAILABLE
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_match(request: PredictionRequest):
    """
    Predict creator-campaign match score
    
    Returns match score (0-1), confidence, and optional explanation
    """
    start_time = time.time()
    
    # Create cache key
    cache_key = f"prediction:{request.creator.creator_id}:{request.campaign.campaign_id}"
    
    # Check cache
    cached_result = None
    if REDIS_AVAILABLE:
        try:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                cache_hit_counter.inc()
                result = json.loads(cached_result)
                result['cached'] = True
                result['processing_time_ms'] = (time.time() - start_time) * 1000
                return PredictionResponse(**result)
        except Exception as e:
            print(f"Cache error: {e}")
    
    cache_miss_counter.inc()
    
    try:
        # Convert Pydantic models to dicts
        creator_dict = request.creator.model_dump()
        creator_dict['categories'] = json.dumps(request.creator.categories)
        creator_dict['platforms'] = json.dumps(request.creator.platforms)
        
        campaign_dict = request.campaign.model_dump()
        campaign_dict['deliverables'] = json.dumps(request.campaign.deliverables)
        
        brand_dict = request.brand.model_dump() if request.brand else None
        
        # Run prediction
        prediction = ensemble.predict(
            creator=creator_dict,
            campaign=campaign_dict,
            brand=brand_dict
        )
        
        # Build response
        processing_time = (time.time() - start_time) * 1000
        
        response = PredictionResponse(
            match_score=prediction['match_score'],
            confidence=prediction['confidence'],
            processing_time_ms=processing_time,
            cached=False,
            explanation=prediction.get('components') if request.include_explanation else None
        )
        
        # Cache result (TTL: 1 hour)
        if REDIS_AVAILABLE:
            try:
                cache_data = response.model_dump()
                redis_client.setex(cache_key, 3600, json.dumps(cache_data))
            except Exception as e:
                print(f"Cache set error: {e}")
        
        # Update metrics
        prediction_counter.inc()
        prediction_latency.observe(processing_time / 1000)
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Predictions"])
async def batch_predict_matches(request: BatchPredictionRequest):
    """
    Rank multiple creators for a campaign
    
    Returns top K matches sorted by score
    """
    start_time = time.time()
    
    try:
        # Convert to dicts
        campaign_dict = request.campaign.model_dump()
        campaign_dict['deliverables'] = json.dumps(request.campaign.deliverables)
        
        brand_dict = request.brand.model_dump() if request.brand else None
        
        # Process each creator
        scores = []
        for creator in request.creators:
            creator_dict = creator.model_dump()
            creator_dict['categories'] = json.dumps(creator.categories)
            creator_dict['platforms'] = json.dumps(creator.platforms)
            
            prediction = ensemble.predict(
                creator=creator_dict,
                campaign=campaign_dict,
                brand=brand_dict
            )
            
            scores.append({
                'creator_id': creator.creator_id,
                'match_score': prediction['match_score'],
                'confidence': prediction['confidence']
            })
        
        # Sort by score
        scores.sort(key=lambda x: x['match_score'], reverse=True)
        top_matches = scores[:request.top_k]
        
        processing_time = (time.time() - start_time) * 1000
        
        # Update metrics
        prediction_counter.inc(len(request.creators))
        
        return BatchPredictionResponse(
            campaign_id=request.campaign.campaign_id,
            total_creators=len(request.creators),
            top_matches=top_matches,
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.post("/explain", tags=["Predictions"])
async def explain_prediction(request: PredictionRequest):
    """
    Get detailed explanation for a prediction
    
    Returns breakdown of scoring factors
    """
    try:
        # Convert to dicts
        creator_dict = request.creator.model_dump()
        creator_dict['categories'] = json.dumps(request.creator.categories)
        creator_dict['platforms'] = json.dumps(request.creator.platforms)
        
        campaign_dict = request.campaign.model_dump()
        campaign_dict['deliverables'] = json.dumps(request.campaign.deliverables)
        
        brand_dict = request.brand.model_dump() if request.brand else None
        
        # Get prediction with components
        prediction = ensemble.predict(
            creator=creator_dict,
            campaign=campaign_dict,
            brand=brand_dict
        )
        
        # Build detailed explanation
        explanation = {
            'overall_score': prediction['match_score'],
            'confidence': prediction['confidence'],
            'score_breakdown': prediction.get('components', {}),
            'factors': {
                'category_match': creator_dict.get('categories', '[]') in campaign_dict.get('category', ''),
                'platform_match': creator_dict.get('platforms', '[]') in campaign_dict.get('platform', ''),
                'meets_follower_requirement': request.creator.followers >= request.campaign.min_followers,
                'meets_engagement_requirement': request.creator.engagement_rate >= request.campaign.min_engagement,
                'experience_level': 'High' if request.creator.total_campaigns > 20 else 'Medium' if request.creator.total_campaigns > 5 else 'Low',
                'success_rate': f"{request.creator.success_rate * 100:.1f}%",
                'rating': f"{request.creator.overall_rating:.1f}/5.0"
            }
        }
        
        return explanation
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explanation error: {str(e)}")


@app.delete("/cache/clear", tags=["Cache"])
async def clear_cache():
    """Clear prediction cache"""
    if not REDIS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Redis not available")
    
    try:
        redis_client.flushdb()
        return {"status": "success", "message": "Cache cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cache clear error: {str(e)}")


# ========== Startup Message (no events to avoid deprecation warnings) ==========

# Print startup message once when module is imported
print("=" * 60)
print("Influencia ML API - Initializing")
print("=" * 60)
print(f"✓ FastAPI app created")
print(f"✓ ML models: Will load on first request")
print(f"✓ Redis: {'Available' if REDIS_AVAILABLE else 'Not available (caching disabled)'}")
print(f"✓ Prometheus metrics: Configured")
print("=" * 60)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=5001,
        reload=False,  # Disable reload to avoid Prometheus metric duplication
        log_level="info"
    )
