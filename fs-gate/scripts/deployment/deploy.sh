#!/bin/bash

# Production deployment script for crop-price MCP server
set -e

echo "🌾 Building crop-price MCP server for production..."

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Build TypeScript
echo "🔨 Building TypeScript..."
npm run build

# Build Docker image
echo "🐳 Building Docker image..."
docker build -t crop-price-mcp:latest .

# Add to Docker MCP catalog
echo "📋 Adding to MCP catalog..."
docker mcp catalog add production crop-price ./crop-price.yaml

# Show catalog
echo "📖 Current catalog:"
docker mcp catalog show production

echo "✅ Deployment ready!"
echo ""
echo "Next steps:"
echo "1. Set your API key: docker mcp server configure crop-price DATAGOVIN_API_KEY=your_key"
echo "2. Enable server: docker mcp server enable crop-price"
echo "3. Start gateway: docker mcp gateway run --servers=crop-price"
echo ""
echo "🚀 Your production chatbot can now use the crop-price tool!"