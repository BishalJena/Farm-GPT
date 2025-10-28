# 🛠️ Adding New MCP Tools Guide

This guide explains how to add new tools to your Agricultural AI MCP Server.

## 📋 Quick Checklist

- [ ] Create tool handler function
- [ ] Add tool to toolHandlers Map
- [ ] Update API documentation in root endpoint
- [ ] Test locally
- [ ] Build and deploy

## 🔧 Step-by-Step Process

### Step 1: Create Tool Handler Function

Add your new tool handler in `src/server.ts` after the existing handlers:

```typescript
/**
 * Your New Tool Handler - Brief description
 */
const yourNewToolHandler = async (params: any) => {
    try {
        // 1. Validate required environment variables
        const API_KEY = process.env.YOUR_API_KEY;
        if (!API_KEY) {
            return {
                error: "Configuration error: YOUR_API_KEY not set in environment."
            };
        }

        // 2. Extract and validate parameters
        const { requiredParam, optionalParam = 'default' } = params;
        
        if (!requiredParam) {
            return {
                error: "Missing required parameter: requiredParam"
            };
        }

        // 3. Make API call or perform logic
        const response = await fetch('https://api.example.com/endpoint', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${API_KEY}`
            },
            body: JSON.stringify({
                query: requiredParam,
                options: optionalParam
            })
        });

        const text = await response.text();

        if (!response.ok) {
            return {
                error: `HTTP ${response.status} fetching API: ${text}`
            };
        }

        // 4. Parse and format response
        let data;
        try {
            data = JSON.parse(text);
        } catch (err) {
            return { error: `Invalid JSON response: ${text}` };
        }

        // 5. Return formatted success response
        return {
            success: true,
            data: {
                results: data.results || [],
                total: data.total || 0,
                query: requiredParam
            }
        };
    } catch (err) {
        return { error: `Server error: ${String(err)}` };
    }
};
```

### Step 2: Register Tool Handler

Add your tool to the toolHandlers Map (around line 155):

```typescript
// Store tool handlers
const toolHandlers = new Map();
toolHandlers.set('crop-price', cropPriceHandler);
toolHandlers.set('search', searchHandler);
toolHandlers.set('your-new-tool', yourNewToolHandler); // Add this line
```

### Step 3: Update API Documentation

Update the root endpoint documentation (around line 190) to include your new tool:

```typescript
tools: [
    {
        name: 'crop-price',
        description: 'Fetch crop price data from data.gov.in',
        endpoint: '/tools/crop-price',
        method: 'POST',
        parameters: { /* existing params */ }
    },
    {
        name: 'search',
        description: 'Search the web for agricultural information',
        endpoint: '/tools/search',
        method: 'POST',
        parameters: { /* existing params */ }
    },
    {
        name: 'your-new-tool',
        description: 'Description of what your tool does',
        endpoint: '/tools/your-new-tool',
        method: 'POST',
        parameters: {
            requiredParam: 'string (required) - Description of required parameter',
            optionalParam: 'string (optional) - Description of optional parameter'
        }
    }
],
```

Also update the examples section:

```typescript
examples: {
    'crop-price': { /* existing example */ },
    'search': { /* existing example */ },
    'your-new-tool': {
        url: '/tools/your-new-tool',
        method: 'POST',
        body: { requiredParam: 'example value', optionalParam: 'optional value' }
    }
}
```

### Step 4: Update Health Check

Update the health check to include your new tool (around line 170):

```typescript
res.end(JSON.stringify({ 
    status: 'healthy', 
    server: 'agricultural-ai-mcp',
    tools: ['crop-price', 'search', 'your-new-tool'], // Add your tool here
    timestamp: new Date().toISOString(),
    environment: {
        datagovin_key_set: !!process.env.DATAGOVIN_API_KEY,
        exa_key_set: !!process.env.EXA_API_KEY,
        your_api_key_set: !!process.env.YOUR_API_KEY, // Add this line
        port: PORT
    }
}));
```

### Step 5: Add Environment Variables

#### For Local Development:
Add to your `.env` file:
```bash
YOUR_API_KEY=your_api_key_here
```

#### For Render Deployment:
1. Go to Render Dashboard
2. Click on your service
3. Go to Environment tab
4. Add: `YOUR_API_KEY` = `your_api_key_here`

### Step 6: Test Your New Tool

#### Local Testing:
```bash
# Build the project
npm run build

# Start the server
npm start

# Test your new tool
curl -X POST http://localhost:10000/tools/your-new-tool \
  -H "Content-Type: application/json" \
  -d '{"requiredParam": "test value", "optionalParam": "optional"}'

# Test health check shows new tool
curl http://localhost:10000/health
```

#### Production Testing (after deployment):
```bash
curl -X POST https://fs-gate.onrender.com/tools/your-new-tool \
  -H "Content-Type: application/json" \
  -d '{"requiredParam": "test value"}'
```

### Step 7: Deploy

```bash
# Commit changes
git add .
git commit -m "Add new tool: your-new-tool"
git push origin main

# Render will auto-deploy
# Check logs for: "🔍 Your new tool: https://fs-gate.onrender.com/tools/your-new-tool"
```

## 🎯 Tool Design Best Practices

### 1. Error Handling
Always return consistent error format:
```typescript
return { error: "Descriptive error message" };
```

### 2. Success Response
Always return consistent success format:
```typescript
return {
    success: true,
    data: {
        // Your structured data here
        results: [],
        total: 0,
        query: params.query
    }
};
```

### 3. Parameter Validation
```typescript
// Required parameters
if (!requiredParam) {
    return { error: "Missing required parameter: requiredParam" };
}

// Optional parameters with defaults
const optionalParam = params.optionalParam ?? 'default_value';
```

### 4. API Key Management
```typescript
const API_KEY = process.env.YOUR_API_KEY;
if (!API_KEY) {
    return { error: "Configuration error: YOUR_API_KEY not set" };
}
```

### 5. HTTP Error Handling
```typescript
if (!response.ok) {
    return { error: `HTTP ${response.status} fetching API: ${text}` };
}
```

## 🔍 Example Tool Ideas

### Weather Tool
- **Purpose**: Get weather data for agricultural regions
- **API**: OpenWeatherMap API
- **Parameters**: `location`, `days_forecast`
- **Use Case**: "What's the weather forecast for Punjab farming regions?"

### Soil Data Tool
- **Purpose**: Get soil quality and composition data
- **API**: Government soil survey APIs
- **Parameters**: `state`, `district`, `soil_type`
- **Use Case**: "What's the soil composition in Maharashtra?"

### Market Trends Tool
- **Purpose**: Analyze crop price trends over time
- **API**: Historical price data APIs
- **Parameters**: `commodity`, `time_period`, `region`
- **Use Case**: "Show wheat price trends over the last 6 months"

### Fertilizer Recommendations Tool
- **Purpose**: Get fertilizer recommendations based on crop and soil
- **API**: Agricultural extension APIs
- **Parameters**: `crop`, `soil_type`, `season`
- **Use Case**: "What fertilizers are recommended for rice in monsoon?"

## 🚀 Quick Template

Copy this template for new tools:

```typescript
/**
 * [TOOL_NAME] Tool Handler - [DESCRIPTION]
 */
const [toolName]Handler = async (params: any) => {
    try {
        const API_KEY = process.env.[API_KEY_NAME];
        if (!API_KEY) {
            return { error: "Configuration error: [API_KEY_NAME] not set" };
        }

        const { [requiredParam] } = params;
        if (![requiredParam]) {
            return { error: "Missing required parameter: [requiredParam]" };
        }

        // Your API logic here

        return {
            success: true,
            data: {
                // Your response data
            }
        };
    } catch (err) {
        return { error: `Server error: ${String(err)}` };
    }
};

// Don't forget to:
// 1. Add to toolHandlers Map
// 2. Update API documentation
// 3. Update health check
// 4. Add environment variables
// 5. Test and deploy
```

## 📝 Notes

- **Tool names**: Use kebab-case (e.g., 'crop-price', 'weather-forecast')
- **Parameters**: Use camelCase (e.g., 'requiredParam', 'optionalValue')
- **Responses**: Always include success/error status
- **Testing**: Test locally before deploying
- **Documentation**: Update API docs for each new tool

## 🎯 Your Current Tools

1. **crop-price**: Fetches crop price data from data.gov.in
2. **search**: Web search for agricultural information via EXA API

Ready to add more tools to make your Agricultural AI MCP Server even more powerful! 🌾🤖