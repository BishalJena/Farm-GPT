#!/bin/bash
# 🌊 Check DigitalOcean Deployment Status

echo "🌊 DigitalOcean Agricultural AI MCP Server - Status Check"
echo "====================================================="

# Get your droplet IP (you'll need to provide this)
echo "Please provide your DigitalOcean droplet IP address:"
read -r DROPLET_IP

if [ -z "$DROPLET_IP" ]; then
    echo "❌ No IP provided. Exiting."
    exit 1
fi

echo "🔍 Checking services on $DROPLET_IP..."

# Check main API (try both ports)
echo "1. 🌾 Agricultural AI API:"
API_PORT=""
if curl -s -f "http://$DROPLET_IP:10000/health" > /dev/null; then
    echo "   ✅ API is running at http://$DROPLET_IP:10000"
    API_PORT="10000"
    curl -s "http://$DROPLET_IP:10000/health" | jq '.' 2>/dev/null || echo "   📊 Health check successful"
elif curl -s -f "http://$DROPLET_IP/health" > /dev/null; then
    echo "   ✅ API is running at http://$DROPLET_IP (port 80)"
    API_PORT="80"
    curl -s "http://$DROPLET_IP/health" | jq '.' 2>/dev/null || echo "   📊 Health check successful"
else
    echo "   ❌ API not responding on either port 10000 or 80"
fi

# Check MCP Gateway (if running)
echo "2. 🚪 MCP Gateway:"
if timeout 5 curl -s -f "http://$DROPLET_IP:8811/health" > /dev/null 2>&1; then
    echo "   ✅ Gateway is running at http://$DROPLET_IP:8811"
else
    echo "   ⚠️  Gateway not running at http://$DROPLET_IP:8811 (this is optional)"
fi

# Check port 80 (nginx)
echo "3. 🌐 Web Server:"
if timeout 5 curl -s -f "http://$DROPLET_IP/health" > /dev/null 2>&1; then
    echo "   ✅ Web server is running at http://$DROPLET_IP"
else
    echo "   ⚠️  Web server not responding at http://$DROPLET_IP (using direct port access)"
fi

# Test API endpoints
echo ""
echo "🧪 Testing API Endpoints:"

# Test crop-price (use detected port)
echo "4. 🌾 Testing crop-price tool:"
if [ "$API_PORT" = "80" ]; then
    API_BASE="http://$DROPLET_IP"
else
    API_BASE="http://$DROPLET_IP:$API_PORT"
fi

CROP_RESPONSE=$(curl -s -X POST "$API_BASE/tools/crop-price" \
  -H "Content-Type: application/json" \
  -d '{"state": "Punjab", "commodity": "Wheat", "limit": 3}' 2>/dev/null)

if echo "$CROP_RESPONSE" | jq -e '.success' > /dev/null 2>&1; then
    echo "   ✅ Crop price tool working"
    echo "$CROP_RESPONSE" | jq '.data.total' 2>/dev/null | sed 's/^/   📊 Records available: /'
else
    echo "   ❌ Crop price tool not working"
    echo "   Response: $CROP_RESPONSE"
fi

# Test search
echo "5. 🔍 Testing search tool:"
SEARCH_RESPONSE=$(curl -s -X POST "$API_BASE/tools/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "Indian agriculture", "num_results": 2}' 2>/dev/null)

if echo "$SEARCH_RESPONSE" | jq -e '.success' > /dev/null 2>&1; then
    echo "   ✅ Search tool working"
else
    echo "   ❌ Search tool not working"
    echo "   Response: $SEARCH_RESPONSE"
fi

echo ""
echo "🎯 Your Chatbot Endpoints:"
echo "=========================="
if [ "$API_PORT" = "80" ]; then
    echo "• Direct API: http://$DROPLET_IP"
    echo "• Health Check: http://$DROPLET_IP/health"
    echo "• Crop Prices: http://$DROPLET_IP/tools/crop-price"
    echo "• Search: http://$DROPLET_IP/tools/search"
else
    echo "• Direct API: http://$DROPLET_IP:$API_PORT"
    echo "• Health Check: http://$DROPLET_IP:$API_PORT/health"
    echo "• Crop Prices: http://$DROPLET_IP:$API_PORT/tools/crop-price"
    echo "• Search: http://$DROPLET_IP:$API_PORT/tools/search"
fi

if curl -s -f "http://$DROPLET_IP:8811/health" > /dev/null; then
    echo "• MCP Gateway: http://$DROPLET_IP:8811"
    echo "• Gateway Tools: http://$DROPLET_IP:8811/tools/call"
fi

echo ""
echo "📝 Integration Example for Your Chatbot:"
echo "========================================"
cat << EOF
const API_BASE = '$API_BASE';

// Get crop prices
async function getCropPrices(state, commodity) {
  const response = await fetch(\`\${API_BASE}/tools/crop-price\`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ state, commodity, limit: 10 })
  });
  return response.json();
}

// Search agriculture info
async function searchAgriculture(query) {
  const response = await fetch(\`\${API_BASE}/tools/search\`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, num_results: 5 })
  });
  return response.json();
}

// Usage
const prices = await getCropPrices('Punjab', 'Wheat');
const research = await searchAgriculture('sustainable farming India');
EOF

echo ""
echo "✅ Status check complete!"