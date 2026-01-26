# Influencia AI Service

Production-grade AI/ML recommendation engine for the Influencia platform. Powers intelligent creator-brand matching using modern ML techniques.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 🧠 Features

### Recommendation System
- **Two-Tower Architecture** - Separate embeddings for creators and campaigns
- **Semantic Matching** - NLP-based category and content matching
- **Candidate Generation** - Fast approximate nearest neighbor search
- **Learning-to-Rank** - XGBoost/LightGBM ranking models
- **Re-ranking** - Diversity, exploration, and fairness optimization

### AI Capabilities
- **Creator Analysis** - Deep profile and performance analysis
- **Match Scoring** - Multi-factor compatibility scoring
- **Engagement Prediction** - Predict campaign performance
- **LLM Integration** - Gemini-powered explanations and insights

### Production Features
- **REST API** - Flask-based production API
- **Caching** - Response caching for performance
- **Rate Limiting** - Protection against abuse
- **Health Checks** - Kubernetes-ready health endpoints
- **Metrics** - Prometheus metrics for monitoring

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Recommendation Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Candidate  │───▶│   Ranking    │───▶│  Re-ranking  │       │
│  │  Generation  │    │    Model     │    │   (Diversity)│       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                 │
│         ▼                   ▼                   ▼                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │  Embeddings  │    │   Features   │    │  Thompson    │       │
│  │  + FAISS     │    │  Engineering │    │  Sampling    │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

- **Python 3.11+** - Core language
- **PyTorch** - Deep learning framework
- **Sentence Transformers** - Text embeddings
- **XGBoost** - Gradient boosting for ranking
- **scikit-learn** - ML utilities
- **Flask** - REST API framework
- **Google Generative AI** - LLM integration (Gemini)

## 📋 Prerequisites

- Python 3.11+
- pip or conda
- 4GB+ RAM (for embedding models)
- GPU optional (CPU works fine)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/suhelali14/influencia-ai.git
   cd influencia-ai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Add your GEMINI_API_KEY for LLM features
   ```

5. **Start the server**
   ```bash
   python api_server_v2.py
   ```

## ⚙️ Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | ❌ |
| `USE_LLM` | Enable LLM features | ❌ |
| `PORT` | Server port | ❌ (default: 5001) |
| `DEBUG` | Debug mode | ❌ |

## 📁 Project Structure

```
ai/
├── core/                    # Core recommendation engine
│   ├── entities.py          # Data classes
│   ├── feature_engineering.py   # Feature computation
│   ├── embeddings.py        # Two-tower embeddings
│   ├── ranking.py           # Learning-to-rank model
│   ├── reranking.py         # Diversity & exploration
│   ├── llm_integration.py   # Gemini integration
│   └── recommendation_engine.py  # Main orchestrator
├── training/                # Model training scripts
├── data/                    # Data storage
├── models/                  # Saved models
├── api_server.py           # Legacy API server
├── api_server_v2.py        # Production API server
├── ml_matching.py          # ML matching utilities
└── requirements.txt        # Python dependencies
```

## 🔌 API Endpoints

### V2 API (Recommended)

#### Recommendations
```http
POST /api/v2/recommend/creators
Content-Type: application/json

{
  "campaign": {
    "id": "camp_123",
    "name": "Summer Campaign",
    "category": "Fashion",
    "budget": 50000,
    "target_audience": "18-35, Urban India"
  },
  "top_k": 10,
  "diversity_weight": 0.3
}
```

#### Match Scoring
```http
POST /api/v2/match/score
Content-Type: application/json

{
  "creator": { ... },
  "campaign": { ... }
}
```

#### Creator Analysis
```http
POST /api/v2/analyze/creator
Content-Type: application/json

{
  "creator": { ... }
}
```

### Health & Monitoring
- `GET /health` - Basic health check
- `GET /api/v2/health` - Detailed health status
- `GET /metrics` - Prometheus metrics

### Legacy API (v1)
- `POST /api/ml/match` - Legacy matching endpoint
- `POST /api/ml/rank` - Legacy ranking endpoint

## 🧪 Testing

```bash
# Run tests
pytest

# With coverage
pytest --cov=core

# Specific test file
pytest test_comprehensive.py
```

## 🐳 Docker

```bash
# Build image
docker build -t influencia-ai .

# Run container
docker run -p 5001:5001 --env-file .env influencia-ai
```

## 📊 Model Training

```bash
# Generate training data
python training/data_generator.py

# Train ranking model
python training/train_models.py

# Train India-specific models
python training/train_india_models.py
```

## 🔧 Configuration

### Recommendation Config
```python
config = RecommendationConfig(
    top_k=20,                    # Number of candidates
    diversity_weight=0.3,        # Diversity vs relevance
    exploration_rate=0.1,        # Exploration probability
    min_score_threshold=0.3,     # Minimum match score
    use_llm_explanations=True    # Enable LLM explanations
)
```

## 📈 Performance

| Metric | Value |
|--------|-------|
| Latency (p50) | ~100ms |
| Latency (p99) | ~500ms |
| Throughput | 100 req/s |
| Model Size | ~500MB |

## 🤝 Related Repositories

- [influencia-backend](https://github.com/suhelali14/influencia-backend) - NestJS Backend API
- [influencia-frontend](https://github.com/suhelali14/influencia-frontend) - React Frontend

## 📄 License

MIT License
