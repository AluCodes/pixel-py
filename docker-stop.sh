#!/bin/bash
# Docker stop script for FastAPI Microservice Backend

set -e

MODE=${1:-dev}

if [ "$MODE" = "prod" ]; then
    echo "🛑 Stopping FastAPI Microservice Backend (Production)"
    docker-compose -f docker-compose.prod.yml down
else
    echo "🛑 Stopping FastAPI Microservice Backend (Development)"
    docker-compose down
fi

echo "✅ Containers stopped!"
