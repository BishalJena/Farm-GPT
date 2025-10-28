# 🌊 DigitalOcean Deployment Guide
## Agricultural AI MCP Server with Docker MCP Gateway

This guide provides comprehensive instructions for deploying your Agricultural AI MCP Server with Docker MCP Gateway integration to DigitalOcean for production use in hackathons and deployed chatbots.

## 🎯 Deployment Options

### Option 1: DigitalOcean App Platform (Recommended for Hackathons)
**Best for**: Quick deployment, automatic scaling, zero-downtime updates
**Cost**: ~$12-25/month
**Setup Time**: 10 minutes

### Option 2: DigitalOcean Droplet with Docker Compose
**Best for**: Full control, custom configurations, Docker MCP Gateway
**Cost**: ~$12-48/month  
**Setup Time**: 20 minutes

### Option 3: Hybrid Deployment
**Best for**: Maximum reliability and creative showcase
**Cost**: ~$24-60/month
**Setup Time**: 30 minutes

---

## 🚀 Option 1: App Platform Deployment

### Step 1: Prepare Your Repository

```bash
# Ensure your code is pushed to GitHub
git add .
git commit -m "Prepare for DigitalOcean App Platform deployment"
git push origin main
```

### Step 2: Create App Platform Deployment

1. **Via GitHub Actions (Automated)**:

Create `.github/workflows/deploy-app-platform.yml`:

```yaml
name: Deploy to DigitalOcean App Platform

on:
  push:
    branches: [main]

permissions:
  contents: read
  packages: write

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        
      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
          
      - name: Build and push Docker image
        id: push
        uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ghcr.io/${{ github.repository }}:latest
          
      - name: Deploy to App Platform
        uses: digitalocean/app_action/deploy@v2
        with:
          token: ${{ secrets.DIGITALOCEAN_ACCESS_TOKEN }}
        env:
          DATAGOVIN_API_KEY: ${{ secrets.DATAGOVIN_API_KEY }}
          EXA_API_KEY: ${{ secrets.EXA_API_KEY }}
```

2. **Via DigitalOcean Control Panel**:

```bash
# 1. Go to https://cloud.digitalocean.com/apps
# 2. Click "Create App"
# 3. Select "GitHub" as source
# 4. Choose your repository and "main" branch
# 5. Enable "Autodeploy"
# 6. Configure environment variables:
#    - DATAGOVIN_API_KEY: your_datagovin_key
#    - EXA_API_KEY: your_exa_key
#    - PORT: 10000
# 7. Select region (NYC3, SFO3, or AMS3 recommended)
# 8. Choose plan: Basic ($12/month) or Pro ($25/month)
# 9. Click "Launch App"
```

### Step 3: Configure Environment Variables

In App Platform settings, add:
```bash
DATAGOVIN_API_KEY=your_datagovin_api_key_here
EXA_API_KEY=your_exa_api_key_here
DATAGOVIN_RESOURCE_ID=35985678-0d79-46b4-9ed6-6f13308a1d24
NODE_ENV=production
PORT=10000
```

### Step 4: Test Deployment

```bash
# Your app will be available at:
# https://your-app-name-xxxxx.ondigitalocean.app

# Test endpoints:
curl https://your-app-name-xxxxx.ondigitalocean.app/health
curl -X POST https://your-app-name-xxxxx.ondigitalocean.app/tools/crop-price \
  -H "Content-Type: application/json" \
  -d '{"state": "Punjab", "commodity": "Wheat"}'
```

---

## 🐳 Option 2: Droplet with Docker MCP Gateway

### Step 1: Create DigitalOcean Droplet

```bash
# Via DigitalOcean API
curl -X POST -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer $DIGITALOCEAN_TOKEN' \
  -d '{
    "name": "agricultural-ai-mcp-server",
    "region": "nyc3",
    "size": "s-2vcpu-4gb",
    "image": "docker-20-04",
    "ssh_keys": ["your-ssh-key-id"],
    "user_data": "#!/bin/bash\napt-get update\napt-get install -y git curl\ncurl -L \"https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)\" -o /usr/local/bin/docker-compose\nchmod +x /usr/local/bin/docker-compose"
  }' \
  "https://api.digitalocean.com/v2/droplets"
```

Or via Control Panel:
1. Go to https://cloud.digitalocean.com/droplets
2. Click "Create Droplet"
3. Choose "Docker on Ubuntu 20.04"
4. Select size: "Basic - $24/month (2 vCPU, 4GB RAM)"
5. Choose region closest to your users
6. Add your SSH key
7. Click "Create Droplet"

### Step 2: Connect and Setup

```bash
# Connect to your droplet
ssh root@your_droplet_ip

# Clone your repository
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Set environment variables
cat > .env << EOF
DATAGOVIN_API_KEY=your_datagovin_api_key_here
EXA_API_KEY=your_exa_api_key_here
DATAGOVIN_RESOURCE_ID=35985678-0d79-46b4-9ed6-6f13308a1d24
NODE_ENV=production
EOF

# Install Docker MCP Gateway
git clone https://github.com/docker/mcp-gateway.git /tmp/mcp-gateway
cd /tmp/mcp-gateway
mkdir -p "$HOME/.docker/cli-plugins/"
make docker-mcp
cd -

# Setup and run
./setup-mcp-gateway.sh
```

### Step 3: Configure Production Docker Compose

Create `docker-compose.production.yml`:

```yaml
version: '3.8'

services:
  # Docker MCP Gateway - Production
  mcp-gateway:
    image: docker/mcp-gateway:latest
    container_name: agricultural-mcp-gateway-prod
    ports:
      - "80:8811"    # HTTP access
      - "8811:8811"  # Direct MCP access
    environment:
      - DOCKER_MCP_METRICS_INTERVAL=60
      - NODE_ENV=production
    volumes:
      - ./agricultural-ai-catalog.yaml:/catalogs/agricultural-ai.yaml:ro
      - /var/run/docker.sock:/var/run/docker.sock
    command: 
      - --catalog=/catalogs/agricultural-ai.yaml
      - --servers=agricultural-ai-unified
      - --transport=sse
      - --port=8811
    restart: always
    depends_on:
      - agricultural-ai-server
    networks:
      - mcp-network

  # Agricultural AI Server - Production
  agricultural-ai-server:
    build: 
      context: .
      dockerfile: Dockerfile
    container_name: agricultural-ai-mcp-server-prod
    ports:
      - "10000:10000"
    env_file:
      - .env
    environment:
      - NODE_ENV=production
      - PORT=10000
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:10000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - mcp-network

  # Nginx Reverse Proxy
  nginx:
    image: nginx:alpine
    container_name: agricultural-mcp-nginx
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - mcp-gateway
      - agricultural-ai-server
    restart: always
    networks:
      - mcp-network

networks:
  mcp-network:
    driver: bridge
    name: agricultural-mcp-network-prod
```

### Step 4: Configure Nginx

Create `nginx.conf`:

```nginx
events {
    worker_connections 1024;
}

http {
    upstream mcp_gateway {
        server mcp-gateway:8811;
    }
    
    upstream ai_server {
        server agricultural-ai-server:10000;
    }
    
    server {
        listen 443 default_server;
        server_name _;
        
        # MCP Gateway routes
        location /mcp/ {
            proxy_pass http://mcp_gateway/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # SSE support
            proxy_buffering off;
            proxy_cache off;
            proxy_set_header Connection '';
            proxy_http_version 1.1;
            chunked_transfer_encoding off;
        }
        
        # Direct API routes
        location /api/ {
            proxy_pass http://ai_server/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }
        
        # Health check
        location /health {
            proxy_pass http://ai_server/health;
        }
        
        # Root - API documentation
        location / {
            proxy_pass http://ai_server/;
        }
    }
}
```

### Step 5: Deploy Production Services

```bash
# Start production services
docker-compose -f docker-compose.production.yml up -d

# Check status
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f

# Test services
curl http://your_droplet_ip/health
curl http://your_droplet_ip:8811/
```

---

## 🌐 Option 3: Hybrid Deployment

Deploy both App Platform (for reliability) and Droplet (for Docker MCP Gateway showcase):

### Step 1: Deploy to App Platform (Primary)
Follow Option 1 steps above.

### Step 2: Deploy to Droplet (Demo/Gateway)
Follow Option 2 steps above.

### Step 3: Configure Load Balancing

Create a DigitalOcean Load Balancer:
```bash
# Via API
curl -X POST -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer $DIGITALOCEAN_TOKEN' \
  -d '{
    "name": "agricultural-ai-lb",
    "algorithm": "round_robin",
    "status": "active",
    "forwarding_rules": [
      {
        "entry_protocol": "https",
        "entry_port": 443,
        "target_protocol": "http",
        "target_port": 80
      }
    ],
    "health_check": {
      "protocol": "http",
      "port": 80,
      "path": "/health",
      "check_interval_seconds": 10,
      "response_timeout_seconds": 5,
      "unhealthy_threshold": 3,
      "healthy_threshold": 2
    },
    "droplet_ids": [your_droplet_id],
    "region": "nyc3"
  }' \
  "https://api.digitalocean.com/v2/load_balancers"
```

---

## 🧪 Testing Your Deployment

### Automated Testing Script

Create `test-deployment.sh`:

```bash
#!/bin/bash

# Configuration
BASE_URL=${1:-"https://your-app.ondigitalocean.app"}
GATEWAY_URL=${2:-"http://your-droplet-ip:8811"}

echo "🧪 Testing Agricultural AI MCP Server Deployment"
echo "================================================"

# Test 1: Health Check
echo "1. Testing health endpoint..."
curl -f "$BASE_URL/health" | jq '.' || echo "❌ Health check failed"

# Test 2: Crop Price Tool
echo "2. Testing crop price tool..."
curl -X POST "$BASE_URL/tools/crop-price" \
  -H "Content-Type: application/json" \
  -d '{"state": "Punjab", "commodity": "Wheat", "limit": 5}' \
  | jq '.success' || echo "❌ Crop price tool failed"

# Test 3: Search Tool
echo "3. Testing search tool..."
curl -X POST "$BASE_URL/tools/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "Indian agriculture news", "num_results": 3}' \
  | jq '.success' || echo "❌ Search tool failed"

# Test 4: MCP Gateway (if available)
if [ "$GATEWAY_URL" != "http://your-droplet-ip:8811" ]; then
  echo "4. Testing MCP Gateway..."
  curl -f "$GATEWAY_URL/" || echo "❌ MCP Gateway not accessible"
fi

echo "✅ Deployment testing complete!"
```

### Load Testing

```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Test concurrent requests
ab -n 100 -c 10 https://your-app.ondigitalocean.app/health

# Test tool endpoints
ab -n 50 -c 5 -p crop-price-payload.json -T application/json \
  https://your-app.ondigitalocean.app/tools/crop-price
```

---

## 📊 Monitoring and Maintenance

### DigitalOcean Monitoring

1. **Enable Monitoring**:
```bash
# Via API
curl -X POST -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer $DIGITALOCEAN_TOKEN' \
  -d '{"type": "agent"}' \
  "https://api.digitalocean.com/v2/monitoring/alerts"
```

2. **Set Up Alerts**:
- CPU usage > 80%
- Memory usage > 90%
- Disk usage > 85%
- HTTP response time > 2s

### Application Monitoring

Create `monitoring/docker-compose.yml`:

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
```

---

## 🔒 Security Best Practices

### 1. Environment Variables Security

```bash
# Use DigitalOcean Secrets (App Platform)
# Or encrypted environment files (Droplets)

# For Droplets, encrypt sensitive files:
gpg --symmetric --cipher-algo AES256 .env
```

### 2. Network Security

```bash
# Configure UFW firewall (Droplets)
ufw allow ssh
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 8811/tcp  # MCP Gateway
ufw --force enable
```

### 3. SSL/TLS Configuration

```bash
# Install Certbot for Let's Encrypt
sudo apt-get install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

---

## 💰 Cost Optimization

### App Platform Costs
- **Basic Plan**: $12/month (512MB RAM, 1 vCPU)
- **Pro Plan**: $25/month (1GB RAM, 1 vCPU)
- **Bandwidth**: Free up to 100GB/month

### Droplet Costs
- **Basic**: $12/month (1GB RAM, 1 vCPU, 25GB SSD)
- **Standard**: $24/month (2GB RAM, 1 vCPU, 50GB SSD)
- **General Purpose**: $48/month (4GB RAM, 2 vCPU, 80GB SSD)

### Cost-Saving Tips
1. Use App Platform for production (auto-scaling)
2. Use smaller Droplet for development/demo
3. Enable monitoring to optimize resource usage
4. Use DigitalOcean Spaces for static assets

---

## 🎯 Hackathon Demo URLs

After deployment, you'll have:

### App Platform Deployment
```
Primary API: https://agricultural-ai-xxxxx.ondigitalocean.app
Health Check: https://agricultural-ai-xxxxx.ondigitalocean.app/health
Crop Prices: https://agricultural-ai-xxxxx.ondigitalocean.app/tools/crop-price
Search: https://agricultural-ai-xxxxx.ondigitalocean.app/tools/search
```

### Droplet with MCP Gateway
```
MCP Gateway: http://your-droplet-ip:8811
Direct API: http://your-droplet-ip:10000
Health Check: http://your-droplet-ip/health
```

### Integration Examples for Judges

```javascript
// For your chatbot integration
const API_BASE = 'https://agricultural-ai-xxxxx.ondigitalocean.app';
const GATEWAY_BASE = 'http://your-droplet-ip:8811';

// Direct API call
const cropPrices = await fetch(`${API_BASE}/tools/crop-price`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ state: 'Punjab', commodity: 'Wheat' })
});

// MCP Gateway call
const gatewayResponse = await fetch(`${GATEWAY_BASE}/tools/call`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    name: 'crop-price',
    arguments: { state: 'Punjab', commodity: 'Wheat' }
  })
});
```

---

## 🚨 Troubleshooting

### Common Issues

1. **Build Failures**:
```bash
# Check build logs
docker-compose logs agricultural-ai-server

# Rebuild with no cache
docker-compose build --no-cache
```

2. **Port Conflicts**:
```bash
# Check what's using ports
sudo netstat -tulpn | grep :8811
sudo netstat -tulpn | grep :10000

# Kill conflicting processes
sudo pkill -f "process-name"
```

3. **Memory Issues**:
```bash
# Check memory usage
free -h
docker stats

# Increase swap (Droplets)
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

4. **SSL Certificate Issues**:
```bash
# Check certificate status
sudo certbot certificates

# Renew certificates
sudo certbot renew --dry-run
```

### Getting Help

- **DigitalOcean Support**: Available 24/7 for paid accounts
- **Community**: https://www.digitalocean.com/community
- **Documentation**: https://docs.digitalocean.com
- **Status Page**: https://status.digitalocean.com

---

## 🎉 Success Checklist

- [ ] ✅ App deployed and accessible via HTTPS
- [ ] ✅ Health check endpoint responding
- [ ] ✅ Crop price tool returning real data
- [ ] ✅ Search tool returning agricultural news
- [ ] ✅ MCP Gateway routing tools (if using Droplet)
- [ ] ✅ Environment variables configured securely
- [ ] ✅ Monitoring and alerts set up
- [ ] ✅ SSL certificate installed and auto-renewing
- [ ] ✅ Demo URLs ready for hackathon judges
- [ ] ✅ Load testing completed successfully

**Your Agricultural AI MCP Server is now production-ready on DigitalOcean! 🌾🚀**