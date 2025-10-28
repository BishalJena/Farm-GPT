#!/bin/bash
# 🚀 Update DigitalOcean Deployment with New Files

echo "🚀 Updating DigitalOcean Agricultural AI MCP Server"
echo "================================================="

# Get droplet IP
echo "Please provide your DigitalOcean droplet IP address:"
read -r DROPLET_IP

if [ -z "$DROPLET_IP" ]; then
    echo "❌ No IP provided. Exiting."
    exit 1
fi

echo "Please provide your SSH key path (or press Enter for default ~/.ssh/id_rsa):"
read -r SSH_KEY
SSH_KEY=${SSH_KEY:-~/.ssh/id_rsa}

echo "🔄 Updating deployment on $DROPLET_IP..."

# Create temporary directory for deployment files
TEMP_DIR=$(mktemp -d)
echo "📁 Created temp directory: $TEMP_DIR"

# Copy essential files to temp directory
cp docker-compose.creative.yml "$TEMP_DIR/"
cp agricultural-ai-catalog.yaml "$TEMP_DIR/"
cp prometheus-simple.yml "$TEMP_DIR/"
cp quick-setup.sh "$TEMP_DIR/"
cp creative-gateway-demo.sh "$TEMP_DIR/"
cp MCP_GATEWAY_CREATIVE.md "$TEMP_DIR/"
cp .env "$TEMP_DIR/" 2>/dev/null || echo "⚠️  No .env file found locally"

# Create deployment script
cat > "$TEMP_DIR/deploy.sh" << 'EOF'
#!/bin/bash
echo "🔄 Updating Agricultural AI MCP Server..."

# Stop existing services
docker-compose down 2>/dev/null || echo "No existing services to stop"

# Backup old files
mkdir -p backup/$(date +%Y%m%d_%H%M%S)
cp *.yml *.yaml backup/$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true

# Start new services
docker-compose -f docker-compose.creative.yml up -d --build

echo "⏳ Waiting for services to start..."
sleep 30

# Test services
echo "🧪 Testing services..."
if curl -s http://localhost:10000/health > /dev/null; then
    echo "✅ Agricultural AI API: Ready"
else
    echo "❌ Agricultural AI API: Not ready"
fi

if curl -s http://localhost:8811/health > /dev/null; then
    echo "✅ MCP Gateway: Ready"
else
    echo "⚠️  MCP Gateway: Not running (this is optional)"
fi

echo "✅ Update complete!"
echo "🎯 Services available at:"
echo "• API: http://$(curl -s ifconfig.me):10000"
echo "• Gateway: http://$(curl -s ifconfig.me):8811"
echo "• Metrics: http://$(curl -s ifconfig.me):9090"
EOF

chmod +x "$TEMP_DIR/deploy.sh"

# Upload files to droplet
echo "📤 Uploading files to droplet..."
scp -i "$SSH_KEY" -r "$TEMP_DIR"/* root@"$DROPLET_IP":/root/agricultural-ai-mcp/ 2>/dev/null || {
    echo "📁 Creating directory and uploading..."
    ssh -i "$SSH_KEY" root@"$DROPLET_IP" "mkdir -p /root/agricultural-ai-mcp"
    scp -i "$SSH_KEY" -r "$TEMP_DIR"/* root@"$DROPLET_IP":/root/agricultural-ai-mcp/
}

# Run deployment script on droplet
echo "🚀 Running deployment on droplet..."
ssh -i "$SSH_KEY" root@"$DROPLET_IP" "cd /root/agricultural-ai-mcp && ./deploy.sh"

# Clean up temp directory
rm -rf "$TEMP_DIR"

echo ""
echo "🎉 Deployment Update Complete!"
echo "=============================="
echo "🎯 Your Updated Endpoints:"
echo "• Direct API: http://$DROPLET_IP:10000"
echo "• Health Check: http://$DROPLET_IP:10000/health"
echo "• Crop Prices: http://$DROPLET_IP:10000/tools/crop-price"
echo "• Search: http://$DROPLET_IP:10000/tools/search"
echo "• MCP Gateway: http://$DROPLET_IP:8811 (if enabled)"
echo "• Metrics: http://$DROPLET_IP:9090"
echo ""
echo "🤖 For Your Chatbot Integration:"
echo "const API_BASE = 'http://$DROPLET_IP:10000';"
echo ""
echo "🧪 Test with:"
echo "./check-digitalocean.sh"