# 🎯 Render Deployment Guide - Agricultural AI MCP Server

## 🚀 Why Render is Perfect for Your Hackathon

- ✅ **Free tier** with 750 hours/month
- ✅ **Automatic deployments** from GitHub
- ✅ **Built-in HTTPS** and custom domains
- ✅ **Environment variables** management
- ✅ **Docker support** (perfect for your setup!)
- ✅ **No credit card** required for free tier

## 📁 What We've Built

Your project now has:
- ✅ **Unified HTTP server** with both crop-price and search tools
- ✅ **Clean project structure** (removed unnecessary files)
- ✅ **Production-ready Dockerfile**
- ✅ **Render configuration** (`render.yaml`)
- ✅ **Health check endpoints**
- ✅ **CORS enabled** for web access

## 🛠️ Step-by-Step Render Deployment

### Step 1: Push to GitHub (2 minutes)

```bash
# Initialize git if not already done
git init
git add .
git commit -m "Agricultural AI MCP Server ready for Render deployment"

# Create a new repo on GitHub, then:
git remote add origin https://github.com/your-username/agricultural-ai-mcp.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Render (3 clicks!)

1. **Go to**: https://render.com
2. **Sign up/Login** with your GitHub account
3. **Click**: "New +" → "Web Service"
4. **Connect** your GitHub repository
5. **Configure**:
   - **Name**: `agricultural-ai-mcp`
   - **Runtime**: `Docker`
   - **Build Command**: (leave empty - Docker handles it)
   - **Start Command**: (leave empty - Docker handles it)

### Step 3: Set Environment Variables

In the Render dashboard, add these environment variables:

```
DATAGOVIN_API_KEY = 579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b
EXA_API_KEY = 579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b
DATAGOVIN_RESOURCE_ID = 35985678-0d79-46b4-9ed6-6f13308a1d24
PORT = 10000
```

### Step 4: Deploy!

Click **"Create Web Service"** and Render will:
- ✅ Build your Docker image
- ✅ Deploy to their cloud
- ✅ Give you a live URL like: `https://agricultural-ai-mcp.onrender.com`

## 🎯 Your Live API Endpoints

Once deployed, your chatbot can call:

### Crop Price Tool
```bash
curl -X POST https://your-app.onrender.com/tools/crop-price \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Punjab",
    "commodity": "Wheat",
    "limit": 10
  }'
```

### Search Tool
```bash
curl -X POST https://your-app.onrender.com/tools/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Indian agriculture news 2024",
    "num_results": 5
  }'
```

### Health Check
```bash
curl https://your-app.onrender.com/health
```

### API Documentation
```bash
curl https://your-app.onrender.com/
```

## 🤖 Chatbot Integration

```javascript
// Your chatbot integration
const API_BASE = 'https://your-app.onrender.com';

// Get crop prices
async function getCropPrices(state, commodity, district = null) {
  const response = await fetch(`${API_BASE}/tools/crop-price`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      state,
      commodity,
      district,
      limit: 10
    })
  });
  return response.json();
}

// Search agriculture info
async function searchAgriculture(query) {
  const response = await fetch(`${API_BASE}/tools/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      num_results: 5
    })
  });
  return response.json();
}

// Example usage
const wheatPrices = await getCropPrices('Punjab', 'Wheat');
const agriNews = await searchAgriculture('wheat farming techniques India');
```

## 🎯 Hackathon Demo Points

1. **"We deployed our AI server on Render for 24/7 availability"**
2. **"Our API serves real agricultural data from the Indian government"**
3. **"Here's our live endpoint that judges can test right now"**
4. **"We have both crop price data and web search capabilities"**
5. **"The entire stack is containerized and production-ready"**

## 🔧 Render Features You Get

- **Automatic deployments** on git push
- **Environment variables** management
- **Logs and monitoring** built-in
- **Custom domains** if needed
- **Auto-scaling** based on traffic
- **Zero downtime** deployments
- **Built-in SSL/HTTPS**

## 🚨 Troubleshooting

### Check deployment logs:
1. Go to your Render dashboard
2. Click on your service
3. Check the "Logs" tab for any errors

### Common fixes:
- Make sure environment variables are set correctly
- Verify your GitHub repo is connected
- Check that `Dockerfile` is in the root directory
- Ensure `PORT` environment variable is set to `10000`

### Test locally first:
```bash
# Build and test locally
npm run build
npm start

# Test endpoints
curl http://localhost:10000/health
curl -X POST http://localhost:10000/tools/crop-price -H "Content-Type: application/json" -d '{"state":"Punjab","commodity":"Wheat"}'
```

## 🎉 Success!

Once deployed, you'll have:
- ✅ **Live API** at `https://your-app.onrender.com`
- ✅ **Automatic HTTPS** and SSL certificates
- ✅ **Global CDN** for fast access
- ✅ **Professional deployment** for your hackathon
- ✅ **No local dependencies** - runs 24/7 in the cloud!
- ✅ **Both tools unified** in one clean API

## 📊 Response Format

### Successful Response:
```json
{
  "success": true,
  "data": {
    "records": [...],
    "total": 150,
    "limit": 10,
    "offset": 0,
    "query": {
      "state": "Punjab",
      "commodity": "Wheat"
    }
  }
}
```

### Error Response:
```json
{
  "error": "Configuration error: API key not set"
}
```

Your hackathon project is now production-ready and deployed! 🚀

## 🔗 Quick Links

- **Render Dashboard**: https://dashboard.render.com
- **Render Docs**: https://render.com/docs
- **Your Live API**: `https://your-app.onrender.com`

**No more localhost dependencies!** Your Agricultural AI MCP Server is live and ready to impress the judges! 🌾🤖