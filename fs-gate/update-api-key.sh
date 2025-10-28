#!/bin/bash

# Script to update DATAGOVIN_API_KEY in Docker deployment
# Usage: ./update-api-key.sh <new-api-key>

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <new-datagovin-api-key>"
    echo "Example: $0 57abc123def456"
    exit 1
fi

NEW_API_KEY="$1"
ENV_FILE=".env"

echo "🔧 Updating DATAGOVIN_API_KEY in Docker deployment..."

# Create .env file if it doesn't exist
if [ ! -f "$ENV_FILE" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example "$ENV_FILE"
fi

# Update the API key in .env file
if grep -q "DATAGOVIN_API_KEY=" "$ENV_FILE"; then
    # Replace existing key
    sed -i.bak "s/DATAGOVIN_API_KEY=.*/DATAGOVIN_API_KEY=$NEW_API_KEY/" "$ENV_FILE"
    echo "✅ Updated existing DATAGOVIN_API_KEY in $ENV_FILE"
else
    # Add new key
    echo "DATAGOVIN_API_KEY=$NEW_API_KEY" >> "$ENV_FILE"
    echo "✅ Added DATAGOVIN_API_KEY to $ENV_FILE"
fi

echo "🐳 Restarting Docker services to apply changes..."

# Restart the agricultural-ai-server to pick up new API key
docker-compose restart agricultural-ai-server

echo "🎉 API key updated successfully!"
echo "📊 Checking service health..."

# Wait a moment for service to start
sleep 5

# Check if service is healthy
if curl -f http://localhost:10001/health > /dev/null 2>&1; then
    echo "✅ Service is healthy and running with new API key"
else
    echo "⚠️  Service may still be starting. Check logs with: docker-compose logs agricultural-ai-server"
fi

echo ""
echo "🔍 To verify the API key is working:"
echo "curl -X POST http://localhost:10001/tools/crop-price \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"state\": \"Punjab\", \"commodity\": \"wheat\"}'"