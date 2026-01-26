"""
ML API Server - Production Startup Script
Starts the Influencia ML inference API on port 5001
"""
import uvicorn
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 70)
print("  INFLUENCIA ML API SERVER")
print("  India-Focused AI Matching Engine")
print("=" * 70)
print("\nStarting on: http://0.0.0.0:5001")
print("Documentation: http://localhost:5001/docs")
print("Health Check: http://localhost:5001/health")
print("\nPress Ctrl+C to stop")
print("=" * 70)

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=5001,
        reload=False,
        log_level="info"
    )
