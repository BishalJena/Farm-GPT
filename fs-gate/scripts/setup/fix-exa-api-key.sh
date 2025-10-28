#!/bin/bash
# 🔧 Fix EXA API Key on DigitalOcean Deployment

echo "🔧 Fixing EXA API Key on DigitalOcean"
echo "===================================="

# Get droplet IP
echo "Please provide your DigitalOcean droplet IP address:"
read -r DROPLET_IP
DROPLET_IP=${DROPLET_IP:-165.232.190.215}

echo "Please provide your SSH key path (or press Enter for default ~/.ssh/id_rsa):"
read -r SSH_KEY
SSH_KEY=${SSH_KEY:-~/.ssh/id_rsa}

echo "Please provide your EXA API key:"
read -r EXA_API_KEY

if [ -z "$EXA_API_KEY" ]; then
    echo "❌ No EXA API key provided. Exiting."
    exit 1
fi

echo "🔄 Updating EXA API key on $DROPLET_IP..."

# Create update script
cat > /tmp/update-env.sh << EOF
#!/bin/bash
echo "🔄 Updating environment variables..."

# Find the application directory
APP_DIR="/root/agricultural-ai-mcp"
if [ ! -d "\$APP_DIR" ]; then
    APP_DIR="/opt/agricultural-ai-mcp"
fi
if [ ! -d "\$APP_DIR" ]; then
    APP_DIR="\$(find /root -name "agricultural-ai-mcp" -type d | head -1)"
fi

if [ -z "\$APP_DIR" ] || [ ! -d "\$APP_DIR" ]; then
    echo "❌ Cannot find application directory"
    exit 1
fi

echo "📁 Found app directory: \$APP_DIR"
cd "\$APP_DIR"

# Update .env file
if [ -f .env ]; then
    # Update existing EXA_API_KEY
    sed -i 's/^EXA_API_KEY=.*/EXA_API_KEY=$EXA_API_KEY/' .env
    echo "✅ Updated EXA_API_KEY in .env file"
else
    # Create .env file
    cat > .env << ENVEOF
DATAGOVIN_API_KEY=\${DATAGOVIN_API_KEY:-your_datagovin_api_key}
EXA_API_KEY=$EXA_API_KEY
DATAGOVIN_RESOURCE_ID=35985678-0d79-46b4-9ed6-6f13308a1d24
NODE_ENV=production
PORT=10000
ENVEOF
    echo "✅ Created .env file with EXA_API_KEY"
fi

# Restart the service
echo "🔄 Restarting services..."

# Try different restart methods
if command -v docker-compose &> /dev/null; then
    docker-compose down
    docker-compose up -d
    echo "✅ Restarted with docker-compose"
elif command -v docker &> /dev/null; then
    # Find and restart the container
    CONTAINER_ID=\$(docker ps -q --filter "name=agricultural")
    if [ -n "\$CONTAINER_ID" ]; then
        docker restart \$CONTAINER_ID
        echo "✅ Restarted Docker container"
    else
        echo "⚠️  No running container found, trying to start..."
        docker run -d --name agricultural-ai-mcp \\
            -p 80:10000 \\
            -e EXA_API_KEY=$EXA_API_KEY \\
            -e DATAGOVIN_API_KEY=\${DATAGOVIN_API_KEY} \\
            agricultural-ai-mcp:latest
    fi
elif systemctl is-active --quiet agricultural-ai; then
    systemctl restart agricultural-ai
    echo "✅ Restarted systemd service"
elif pm2 list | grep -q agricultural; then
    pm2 restart agricultural-ai
    echo "✅ Restarted PM2 process"
else
    echo "⚠️  Please manually restart your application"
fi

echo "⏳ Waiting for service to restart..."
sleep 10

# Test the service
echo "🧪 Testing search tool..."
RESPONSE=\$(curl -s -X POST "http://localhost/tools/search" \\
  -H "Content-Type: application/json" \\
  -d '{"query": "test", "num_results": 1}')

if echo "\$RESPONSE" | grep -q '"success":true'; then
    echo "✅ Search tool is now working!"
else
    echo "❌ Search tool still not working:"
    echo "\$RESPONSE"
fi

echo "✅ Update complete!"
EOF

chmod +x /tmp/update-env.sh

# Upload and run the script
echo "📤 Uploading update script..."
scp -i "$SSH_KEY" /tmp/update-env.sh root@"$DROPLET_IP":/tmp/

echo "🚀 Running update on droplet..."
ssh -i "$SSH_KEY" root@"$DROPLET_IP" "/tmp/update-env.sh"

# Clean up
rm /tmp/update-env.sh

echo ""
echo "🧪 Testing from your local machine..."
sleep 5

# Test crop prices
echo "1. Testing crop-price tool:"
CROP_RESPONSE=$(curl -s -X POST "http://$DROPLET_IP/tools/crop-price" \
  -H "Content-Type: application/json" \
  -d '{"state": "Punjab", "commodity": "Wheat", "limit": 2}')

if echo "$CROP_RESPONSE" | grep -q '"success":true'; then
    echo "   ✅ Crop price tool working"
else
    echo "   ❌ Crop price tool not working: $CROP_RESPONSE"
fi

# Test search
echo "2. Testing search tool:"
SEARCH_RESPONSE=$(curl -s -X POST "http://$DROPLET_IP/tools/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "sustainable farming", "num_results": 1}')

if echo "$SEARCH_RESPONSE" | grep -q '"success":true'; then
    echo "   ✅ Search tool working"
else
    echo "   ❌ Search tool not working: $SEARCH_RESPONSE"
fi

echo ""
echo "🎯 Your Working Endpoints:"
echo "• API: http://$DROPLET_IP"
echo "• Crop Prices: http://$DROPLET_IP/tools/crop-price"
echo "• Search: http://$DROPLET_IP/tools/search"
echo ""
echo "✅ Fix complete!"