# 🌾 Agricultural AI MCP Server

A production-ready HTTP server providing crop price data and web search capabilities for agricultural AI chatbots. Features **creative Docker MCP Gateway integration** for intelligent tool routing and management.

## 🚀 Quick Start

```bash
# 1. Quick setup (30 seconds)
./scripts/setup/quick-setup.sh

# 2. Test intelligent routing
curl -X POST http://localhost:8811/tools/call \
  -H "Content-Type: application/json" \
  -d '{"name": "crop-price", "arguments": {"state": "Punjab"}}'
```

## 📁 Project Structure

```
├── 📚 docs/                    # Documentation
│   ├── deployment/             # Deployment guides
│   ├── integration/            # Integration guides  
│   └── guides/                 # Feature guides
├── 🔧 scripts/                 # Automation scripts
│   ├── setup/                  # Setup scripts
│   ├── deployment/             # Deployment scripts
│   └── testing/                # Testing scripts
├── ⚙️ config/                  # Configuration files
│   ├── docker/                 # Docker configurations
│   ├── nginx/                  # Nginx configurations
│   └── catalogs/               # MCP catalogs
├── 🤖 examples/                # Integration examples
│   ├── chatbot/                # Chatbot integrations
│   └── integration/            # API examples
└── 💻 src/                     # Source code
```

## 🎯 Features

### Core Agricultural Intelligence
- **Crop Price Data**: Real-time agricultural commodity prices from data.gov.in
- **Web Search**: Agricultural news and information via EXA API
- **HTTP API**: RESTful endpoints for easy integration
- **Health Checks**: Built-in monitoring endpoints

### Docker MCP Gateway Integration
- **🎯 Intelligent Tool Routing**: Gateway routes queries to optimal agricultural tools
- **📋 Production Catalog Management**: Sophisticated server registry with metadata
- **🔄 Dynamic Server Discovery**: Runtime tool registration and management
- **🌐 Dual Protocol Support**: Both HTTP REST and MCP protocol endpoints
- **⚡ Real-time Tool Orchestration**: Gateway manages multiple agricultural intelligence servers

## 🛠️ Setup Options

### Option 1: Local Development
```bash
./scripts/setup/quick-setup.sh
```

### Option 2: DigitalOcean Deployment
```bash
./scripts/deployment/check-digitalocean.sh
```

### Option 3: Creative MCP Gateway
```bash
./scripts/testing/creative-gateway-demo.sh
```

## 📚 Documentation

- **[Setup Guide](docs/guides/quick-start.md)** - Get started in 30 seconds
- **[DigitalOcean Deployment](docs/deployment/DIGITALOCEAN_DEPLOYMENT.md)** - Production deployment
- **[Creative MCP Gateway](docs/guides/MCP_GATEWAY_CREATIVE.md)** - Advanced patterns
- **[Chatbot Integration](examples/chatbot/)** - Ready-to-use examples

## 🤖 Chatbot Integration

```javascript
const AgriculturalAI = require('./examples/chatbot/your-chatbot-config.js');
const agriAI = new AgriculturalAI();

// Ready to use!
const response = await agriAI.processQuery("What are wheat prices in Punjab?");
```

## 🎯 Live Endpoints

- **Direct API**: http://165.232.190.215
- **MCP Gateway**: http://165.232.190.215:8811
- **Health Check**: http://165.232.190.215/health

## 🏆 Perfect for Hackathons

- ✅ **Creative Docker MCP Gateway Usage** - Intelligent agricultural tool orchestration
- ✅ **Production-Ready Architecture** - Real catalog management and server routing
- ✅ **Real Agricultural Data** - Government crop prices and market intelligence
- ✅ **Easy Integration** - Works with any chatbot or AI framework
- ✅ **Live Demo Ready** - Judges can test both gateway and direct API access

## 📝 License

MIT License - Perfect for hackathon projects!
