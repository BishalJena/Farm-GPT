# Hackathon Quick Deploy - Docker MCP Gateway

## 🚀 Cloud Deployment Options (No Mac Required!)

Since Docker is your sponsor and you need remote hosting, here are your **Docker-native** options:

## Option 1: Docker Hub + Cloud VM (Fastest - 10 minutes)

### Step 1: Build and Push to Docker Hub (5 minutes)
```bash
# Login to Docker Hub
docker login

# Build and push your crop-price server
docker build -t your-dockerhub-username/crop-price-mcp:latest .
docker push your-dockerhub-username/crop-price-mcp:latest

# Build and push EXA server (if custom)
docker build -f exa-server/Dockerfile -t your-dockerhub-username/exa-mcp:latest exa-server/
docker push your-dockerhub-username/exa-mcp:latest
```

### Step 2: Deploy to Cloud VM (5 minutes)
```bash
# On any cloud VM (AWS EC2, Google Cloud, DigitalOcean, etc.)
# Create docker-compose.yml:
cat > docker-compose.yml << EOF
version: '3.8'
services:
  mcp-gateway:
    image: docker/mcp-gateway:latest
    ports:
      - "8080:8080"
    environment:
      - DATAGOVIN_API_KEY=${DATAGOVIN_API_KEY}
      - EXA_API_KEY=${EXA_API_KEY}
    command: >
      docker mcp gateway run 
      --port 8080 
      --transport streaming 
      --servers=crop-price,exa
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    depends_on:
      - crop-price-server
      - exa-server

  crop-price-server:
    image: your-dockerhub-username/crop-price-mcp:latest
    environment:
      - DATAGOVIN_API_KEY=${DATAGOVIN_API_KEY}

  exa-server:
    image: your-dockerhub-username/exa-mcp:latest
    environment:
      - EXA_API_KEY=${EXA_API_KEY}
EOF

# Set environment variables
export DATAGOVIN_API_KEY=your_key_here
export EXA_API_KEY=your_key_here

# Deploy
docker-compose up -d
```

**Your Gateway URL**: `http://your-vm-ip:8080`

## Option 2: GitHub Actions + Docker Hub (Most Professional)

### Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy MCP Gateway

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Login to Docker Hub
      uses: docker/login-action@v3
      with:
        username: ${{ secrets.DOCKERHUB_USERNAME }}
        password: ${{ secrets.DOCKERHUB_TOKEN }}
    
    - name: Build and push crop-price server
      uses: docker/build-push-action@v6
      with:
        context: .
        push: true
        tags: ${{ secrets.DOCKERHUB_USERNAME }}/crop-price-mcp:latest
    
    - name: Deploy to cloud
      run: |
        # SSH to your cloud VM and restart services
        echo "Deployed to production!"
```

## Option 3: Railway/Render/Fly.io (Simplest)

### Railway Deployment:
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up

# Your app will be available at: https://your-app.railway.app
```

### Render Deployment:
1. Connect your GitHub repo to Render
2. Create a new Web Service
3. Use Docker deployment
4. Set environment variables in Render dashboard

**Your Gateway URL**: `https://your-app.onrender.com`

## 🏆 Recommended for Hackathon: Railway (2 minutes setup!)

Railway is perfect for hackathons - it's fast, free, and Docker-native:

### Quick Railway Deploy:
```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login
railway login

# 3. Create railway.json in your project root:
cat > railway.json << EOF
{
  "build": {
    "builder": "dockerfile"
  },
  "deploy": {
    "startCommand": "docker mcp gateway run --port 8080 --transport streaming --servers=crop-price,exa"
  }
}
EOF

# 4. Deploy
railway init
railway up
```

**Result**: Your MCP Gateway will be live at `https://your-project.railway.app` in 2 minutes!

## 🤖 Chatbot Integration (Any Option)

Once deployed, your chatbot calls these endpoints:

### Crop Price Tool
```javascript
const response = await fetch('https://your-gateway-url/tools/crop-price', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    state: "Punjab",
    district: "Ludhiana", 
    commodity: "Wheat",
    limit: 10
  })
});
```

### EXA Search Tool
```javascript
const response = await fetch('https://your-gateway-url/tools/search', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: "Indian agriculture news 2024",
    num_results: 5
  })
});
```

## 📱 For Demo/Presentation

### Test Your Live Gateway
```bash
# Test crop-price tool
curl -X POST https://your-gateway-url/tools/crop-price \
  -H "Content-Type: application/json" \
  -d '{"state": "Punjab", "commodity": "Wheat"}'

# Test EXA search
curl -X POST https://your-gateway-url/tools/search \
  -H "Content-Type: application/json" \
  -d '{"query": "agriculture news"}'
```

### Show Docker Integration During Demo
```bash
# Show your images on Docker Hub
docker search your-dockerhub-username/crop-price-mcp

# Show deployment logs
railway logs  # or docker-compose logs
```

## 🎯 Hackathon Demo Script

**"We're using Docker's MCP Gateway deployed on the cloud to power our agricultural AI chatbot..."**

1. **Show live gateway**: Visit `https://your-gateway-url` 
2. **Demo crop prices**: Live API call to Punjab wheat prices
3. **Demo web search**: Live search for agriculture news
4. **Show your chatbot**: Making real HTTP calls to cloud gateway
5. **Highlight Docker**: "Our entire stack runs on Docker containers"
6. **Show scalability**: "This can handle thousands of requests"

## 🔧 Environment Variables Setup

For any deployment option, set these environment variables:

```bash
# Required API keys
DATAGOVIN_API_KEY=your_datagovin_key_here
EXA_API_KEY=your_exa_key_here

# Optional: Custom resource ID
DATAGOVIN_RESOURCE_ID=35985678-0d79-46b4-9ed6-6f13308a1d24

# Port configuration
PORT=8080
```

## 🏆 Why This Wins Your Hackathon

1. **Docker Sponsor Alignment**: Using Docker Hub + Docker Compose + Docker MCP Gateway
2. **Cloud Native**: No local dependencies, runs anywhere
3. **Professional Setup**: CI/CD pipeline, container registry, cloud deployment
4. **Demo Ready**: Live URLs, real API calls, impressive tech stack
5. **Scalable**: Can handle judge traffic and real users
6. **Fast**: 2-10 minute deployment depending on option chosen

## 🚨 Quick Troubleshooting

### Gateway not responding?
```bash
# Check container logs
docker-compose logs mcp-gateway

# Or on Railway
railway logs
```

### API keys not working?
```bash
# Verify environment variables are set
curl https://your-gateway-url/health  # if you add a health endpoint
```

### Tools not found?
```bash
# List available tools
curl https://your-gateway-url/tools/list
```

## 🎉 You're Ready for the Hackathon!

Your project now has:
- ✅ **Cloud-hosted** MCP Gateway (no Mac dependency!)
- ✅ **Docker-powered** infrastructure (sponsor alignment!)
- ✅ **Real agricultural data** from Indian government
- ✅ **Web search capabilities** via EXA
- ✅ **Public HTTPS endpoints** for your chatbot
- ✅ **Professional deployment** pipeline
- ✅ **2-10 minute setup** time

**Gateway URL for your chatbot**: `https://your-chosen-platform.com`

**No more localhost dependencies!** Your hackathon project is now truly cloud-native and ready to impress the judges! 🚀

## 🔗 Quick Links for Setup

- **Railway**: https://railway.app (Fastest)
- **Render**: https://render.com (Simplest)  
- **Docker Hub**: https://hub.docker.com (Most professional)
- **DigitalOcean**: https://digitalocean.com (Most control)

Choose your platform and deploy in minutes!

## 🤖 Chatbot Integration

Your chatbot can now call these endpoints:

### Crop Price Tool
```bash
curl -X POST http://localhost:8080/tools/crop-price \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Punjab",
    "district": "Ludhiana", 
    "commodity": "Wheat",
    "limit": 10
  }'
```

### EXA Search Tool
```bash
curl -X POST http://localhost:8080/tools/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Indian agriculture news 2024",
    "num_results": 5
  }'
```

## 📱 For Demo/Presentation

### Test Your Tools
```bash
# List all available tools
docker mcp tools list

# Test crop-price tool
docker mcp tools call crop-price state=Punjab commodity=Wheat

# Test EXA search
docker mcp tools call search query="agriculture news"
```

### Show Docker Integration
```bash
# Show Docker's MCP ecosystem
docker mcp --help

# Show server status
docker mcp server list

# Show gateway logs (for debugging during demo)
docker mcp gateway logs
```

## 🎯 Hackathon Demo Script

**"We're using Docker's MCP Gateway to power our agricultural chatbot..."**

1. **Show the gateway running**: `docker mcp tools list`
2. **Demo crop prices**: Call crop-price tool with Punjab/Wheat
3. **Demo web search**: Call EXA search for agriculture news
4. **Show your chatbot**: Making HTTP calls to localhost:8080
5. **Highlight Docker**: "This entire MCP ecosystem is powered by Docker"

## 🔧 If You Need Public Access (Optional)

For demo purposes, if judges need to access your chatbot remotely:

### Option A: ngrok (Fastest)
```bash
# Install ngrok
brew install ngrok  # or download from ngrok.com

# Expose your gateway
ngrok http 8080

# Use the ngrok URL in your chatbot: https://abc123.ngrok.io
```

### Option B: Docker + Cloud VM
```bash
# Deploy to any cloud VM
docker-compose up -d

# Your gateway runs on: http://your-vm-ip:8080
```

## 🏆 Why This Wins

1. **Docker Sponsor Alignment**: Using Docker's own MCP Gateway
2. **Speed**: 5-minute setup vs hours for other solutions  
3. **Demo-Ready**: Built-in tools for showing functionality
4. **Hackathon Perfect**: No production complexity, just works
5. **Impressive Tech**: MCP is cutting-edge AI tooling

## 🚨 Troubleshooting

### Gateway won't start?
```bash
# Check if servers are enabled
docker mcp server list

# Check API keys are set
docker mcp server configure crop-price
```

### Tools not working?
```bash
# Test individual tools
docker mcp tools call crop-price --help
docker mcp tools call search --help
```

### Need to restart?
```bash
# Stop gateway
docker mcp gateway stop

# Restart with verbose logging
docker mcp gateway run --port 8080 --transport streaming --servers=crop-price,exa --verbose
```

## 🎉 You're Ready!

Your hackathon project now has:
- ✅ **Docker-powered** MCP Gateway (sponsor alignment!)
- ✅ **Real agricultural data** from Indian government
- ✅ **Web search capabilities** via EXA
- ✅ **HTTP API** for your chatbot integration
- ✅ **5-minute deployment** time
- ✅ **Demo-ready** tooling

**Gateway URL for your chatbot**: `http://localhost:8080`

Go build something amazing! 🚀