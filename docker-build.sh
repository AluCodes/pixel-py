#!/bin/bash
# Docker build script for FastAPI Microservice Backend

set -e

echo "🐳 Building Docker image for FastAPI Microservice Backend"
echo "=========================================================="

# Build the image
docker build -t fastapi-microservice-backend:latest .

echo ""
echo "✅ Docker image built successfully!"
echo ""
echo "Image: fastapi-microservice-backend:latest"
echo ""
echo "Next steps:"
echo "  - Run locally: docker-compose up"
echo "  - Run in production: docker-compose -f docker-compose.prod.yml up -d"
echo "  - Test the image: docker run -p 8000:8000 fastapi-microservice-backend:latest"
