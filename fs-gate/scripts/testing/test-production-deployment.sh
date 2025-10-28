#!/bin/bash

# Get droplet IP
DROPLET_IP=$(curl -s http://169.254.169.254/metadata/v1/interfaces/public/0/ipv4/address 2>/dev/null)
if [ -z "$DROPLET_IP" ]; then
    echo "Enter your droplet IP address:"
    read DROPLET_IP
fi

BASE_URL="http://$DROPLET_IP"
GATEWAY_URL="http://$DROPLET_IP:8811"

echo "🧪 Testing Agricultural AI MCP Server Production Deployment"
echo "=========================================================="
echo "Droplet IP: $DROPLET_IP"
echo "Base URL: $BASE_URL"
echo "Gateway URL: $GATEWAY_URL"
echo ""

# Test 1: Health Check
echo "1. Testing health endpoint..."
if curl -f -s "$BASE_URL/health" | jq '.' > /dev/null 2>&1; then
    echo "✅ Health check passed"
    curl -s "$BASE_URL/health" | jq '.protocols, .tools'
else
    echo "❌ Health check failed"
fi
echo ""

# Test 2: Crop Price Tool
echo "2. Testing crop price tool..."
CROP_RESPONSE=$(curl -s -X POST "$BASE_URL/tools/crop-price" \
  -H "Content-Type: application/json" \
  -d '{"state": "Punjab", "commodity": "Wheat", "limit": 3}')

if echo "$CROP_RESPONSE" | jq -e '.success' > /dev/null 2>&1; then
    echo "✅ Crop price tool working"
    echo "$CROP_RESPONSE" | jq '.data.total, .data.records[0].State, .data.records[0].Commodity'
else
    echo "❌ Crop price tool failed"
    echo "$CROP_RESPONSE"
fi
echo ""

# Test 3: Search Tool
echo "3. Testing search tool..."
SEARCH_RESPONSE=$(curl -s -X POST "$BASE_URL/tools/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "Indian agriculture news", "num_results": 2}')

if echo "$SEARCH_RESPONSE" | jq -e '.success' > /dev/null 2>&1; then
    echo "✅ Search tool working"
    echo "$SEARCH_RESPONSE" | jq '.data.total_results'
else
    echo "❌ Search tool failed"
    echo "$SEARCH_RESPONSE"
fi
echo ""

# Test 4: MCP Gateway
echo "4. Testing MCP Gateway..."
if curl -f -s "$GATEWAY_URL/" > /dev/null 2>&1; then
    echo "✅ MCP Gateway accessible"
    
    # Test MCP tools listing
    echo "5. Testing MCP Gateway tools..."
    if docker mcp tools list > /dev/null 2>&1; then
        echo "✅ MCP Gateway tools available:"
        docker mcp tools list
    else
        echo "⚠️  MCP Gateway tools not fully configured"
    fi
else
    echo "❌ MCP Gateway not accessible"
fi
echo ""

# Test 6: Docker Services Status
echo "6. Checking Docker services..."
docker-compose -f docker-compose.production.yml ps
echo ""

# Test 7: Resource Usage
echo "7. System resource usage..."
echo "Memory usage:"
free -h
echo ""
echo "Disk usage:"
df -h /
echo ""
echo "Docker stats:"
docker stats --no-stream
echo ""

echo "🎉 Production deployment testing complete!"
echo ""
echo "🌐 Your Agricultural AI MCP Server is available at:"
echo "   • Main API: $BASE_URL"
echo "   • Health Check: $BASE_URL/health"
echo "   • Crop Prices: $BASE_URL/tools/crop-price"
echo "   • Search: $BASE_URL/tools/search"
echo "   • MCP Gateway: $GATEWAY_URL"
echo ""
echo "🎯 For hackathon judges, share these URLs:"
echo "   curl $BASE_URL/health"
echo "   curl -X POST $BASE_URL/tools/crop-price -H 'Content-Type: application/json' -d '{\"state\":\"Punjab\",\"commodity\":\"Wheat\"}'"
echo "   curl $GATEWAY_URL/"