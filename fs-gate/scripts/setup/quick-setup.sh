#!/bin/bash
# 🚀 Agricultural AI MCP Server - 30 Second Setup

set -e

echo "🌾 Agricultural AI MCP Server - Quick Setup"
echo "=========================================="

# Check requirements
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

# Check environment variables
if [ -z "$DATAGOVIN_API_KEY" ] || [ -z "$EXA_API_KEY" ]; then
    echo "⚠️  Environment variables not set. Creating .env file..."
    cat > .env << EOF
DATAGOVIN_API_KEY=your_datagovin_api_key_here
EXA_API_KEY=your_exa_api_key_here
DATAGOVIN_RESOURCE_ID=35985678-0d79-46b4-9ed6-6f13308a1d24
NODE_ENV=production
PORT=10000
EOF
    echo "📝 Please edit .env file with your API keys"
fi

# Build and start services
echo "🔧 Building and starting services..."
docker-compose -f docker-compose.creative.yml up -d --build

echo "⏳ Waiting for services to start..."
sleep 20

# Test services
echo "🧪 Testing services..."
if curl -s http://localhost:10001/health > /dev/null; then
    echo "✅ Agricultural AI Server: Ready"
else
    echo "⏳ Agricultural AI Server: Starting..."
fi

if curl -s http://localhost:8811/health > /dev/null; then
    echo "✅ MCP Gateway: Ready"
else
    echo "⏳ MCP Gateway: Starting..."
fi

if curl -s http://localhost:9090/-/healthy > /dev/null; then
    echo "✅ Prometheus: Ready"
else
    echo "⏳ Prometheus: Starting..."
fi

echo ""
echo "🎯 Services Ready!"
echo "=================="
echo "• MCP Gateway: http://localhost:8811"
echo "• Direct API: http://localhost:10001"
echo "• Metrics: http://localhost:9090"
echo "• Health Check: http://localhost:10001/health"
echo ""
echo "🚀 Quick Test Commands:"
echo "curl http://localhost:10001/health"
echo "curl -X POST http://localhost:8811/tools/call -H 'Content-Type: application/json' -d '{\"name\": \"crop-price\", \"arguments\": {\"state\": \"Punjab\"}}'"
echo ""
echo "📚 Documentation:"
echo "• Creative Patterns: ./MCP_GATEWAY_CREATIVE.md"
echo "• Full Demo: ./creative-gateway-demo.sh"
echo ""
echo "✅ Setup Complete! Ready for hackathon judging!"