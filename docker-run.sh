#!/bin/bash
# Docker run script for FastAPI Microservice Backend

set -e

MODE=${1:-dev}

if [ "$MODE" = "prod" ]; then
    echo "🚀 Starting FastAPI Microservice Backend (Production Mode)"
    echo "==========================================================="
    docker-compose -f docker-compose.prod.yml up -d
    echo ""
    echo "✅ Production containers started!"
    echo ""
    echo "View logs: docker-compose -f docker-compose.prod.yml logs -f"
    echo "Stop: docker-compose -f docker-compose.prod.yml down"
else
    echo "🚀 Starting FastAPI Microservice Backend (Development Mode)"
    echo "============================================================"
    docker-compose up
fi
