// 🤖 Agricultural AI Chatbot Integration
// For DigitalOcean Deployment

class AgriculturalAI {
  constructor(dropletIP) {
    // Replace with your actual DigitalOcean droplet IP
    this.apiBase = `http://${dropletIP}:10000`;
    this.gatewayBase = `http://${dropletIP}:8811`; // Optional MCP Gateway
    this.timeout = 10000; // 10 second timeout
  }

  // Get crop prices with intelligent filtering
  async getCropPrices(params = {}) {
    const {
      state = null,
      district = null,
      commodity = null,
      limit = 10,
      offset = 0
    } = params;

    try {
      const response = await fetch(`${this.apiBase}/tools/crop-price`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          state,
          district,
          commodity,
          limit,
          offset
        }),
        timeout: this.timeout
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      if (data.success) {
        return {
          success: true,
          data: data.data,
          message: `Found ${data.data.total} crop price records`
        };
      } else {
        return {
          success: false,
          error: data.error || 'Unknown error',
          message: 'Failed to fetch crop prices'
        };
      }
    } catch (error) {
      return {
        success: false,
        error: error.message,
        message: 'Network error while fetching crop prices'
      };
    }
  }

  // Search agricultural information
  async searchAgriculture(params = {}) {
    const {
      query,
      num_results = 5,
      include_domains = null,
      exclude_domains = null
    } = params;

    if (!query) {
      return {
        success: false,
        error: 'Query is required',
        message: 'Please provide a search query'
      };
    }

    try {
      const response = await fetch(`${this.apiBase}/tools/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          num_results,
          include_domains,
          exclude_domains
        }),
        timeout: this.timeout
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      if (data.success) {
        return {
          success: true,
          data: data.data,
          message: `Found ${data.data.results.length} search results`
        };
      } else {
        return {
          success: false,
          error: data.error || 'Unknown error',
          message: 'Failed to search agricultural information'
        };
      }
    } catch (error) {
      return {
        success: false,
        error: error.message,
        message: 'Network error while searching'
      };
    }
  }

  // Health check
  async checkHealth() {
    try {
      const response = await fetch(`${this.apiBase}/health`, {
        timeout: 5000
      });
      
      if (response.ok) {
        const data = await response.json();
        return {
          success: true,
          status: data.status,
          message: 'Agricultural AI service is healthy'
        };
      } else {
        return {
          success: false,
          message: 'Agricultural AI service is not responding properly'
        };
      }
    } catch (error) {
      return {
        success: false,
        error: error.message,
        message: 'Cannot connect to Agricultural AI service'
      };
    }
  }

  // Intelligent query processing (combines multiple tools)
  async processIntelligentQuery(userQuery) {
    const query = userQuery.toLowerCase();
    
    // Analyze query intent
    const hasPriceIntent = /price|cost|rate|market|trading/.test(query);
    const hasSearchIntent = /research|news|information|study|learn/.test(query);
    const hasLocationIntent = /punjab|maharashtra|gujarat|haryana|uttar pradesh/.test(query);
    const hasCropIntent = /wheat|rice|cotton|maize|sugarcane/.test(query);

    const results = {};

    // If query seems to be about prices and has location/crop info
    if (hasPriceIntent && (hasLocationIntent || hasCropIntent)) {
      // Extract entities (simple keyword matching)
      const states = ['punjab', 'maharashtra', 'gujarat', 'haryana', 'uttar pradesh'];
      const crops = ['wheat', 'rice', 'cotton', 'maize', 'sugarcane'];
      
      const detectedState = states.find(state => query.includes(state));
      const detectedCrop = crops.find(crop => query.includes(crop));

      results.prices = await this.getCropPrices({
        state: detectedState ? detectedState.charAt(0).toUpperCase() + detectedState.slice(1) : null,
        commodity: detectedCrop ? detectedCrop.charAt(0).toUpperCase() + detectedCrop.slice(1) : null,
        limit: 5
      });
    }

    // If query seems to be about research/information
    if (hasSearchIntent || (!hasPriceIntent && userQuery.length > 10)) {
      results.search = await this.searchAgriculture({
        query: userQuery + ' Indian agriculture',
        num_results: 3
      });
    }

    return {
      query: userQuery,
      intent: {
        price: hasPriceIntent,
        search: hasSearchIntent,
        location: hasLocationIntent,
        crop: hasCropIntent
      },
      results
    };
  }

  // Format response for chatbot display
  formatResponse(result) {
    let response = '';

    if (result.results.prices && result.results.prices.success) {
      const prices = result.results.prices.data.records.slice(0, 3);
      response += '🌾 **Current Crop Prices:**\n';
      prices.forEach(record => {
        response += `• ${record.commodity} in ${record.state}: ₹${record.modal_price}/quintal\n`;
      });
      response += '\n';
    }

    if (result.results.search && result.results.search.success) {
      const searchResults = result.results.search.data.results.slice(0, 2);
      response += '📚 **Agricultural Information:**\n';
      searchResults.forEach(item => {
        response += `• ${item.title}\n  ${item.text.substring(0, 100)}...\n\n`;
      });
    }

    if (!response) {
      response = 'I can help you with crop prices and agricultural information. Try asking about specific crops or states!';
    }

    return response.trim();
  }
}

// Usage Examples
async function exampleUsage() {
  // Replace 'YOUR_DROPLET_IP' with your actual DigitalOcean droplet IP
  const agriAI = new AgriculturalAI('YOUR_DROPLET_IP');

  // Check if service is healthy
  const health = await agriAI.checkHealth();
  console.log('Health:', health);

  // Get specific crop prices
  const prices = await agriAI.getCropPrices({
    state: 'Punjab',
    commodity: 'Wheat',
    limit: 5
  });
  console.log('Prices:', prices);

  // Search for information
  const search = await agriAI.searchAgriculture({
    query: 'sustainable farming practices India',
    num_results: 3
  });
  console.log('Search:', search);

  // Intelligent query processing
  const intelligent = await agriAI.processIntelligentQuery(
    'What are the current wheat prices in Punjab?'
  );
  console.log('Intelligent Response:', agriAI.formatResponse(intelligent));
}

// Export for use in your chatbot
if (typeof module !== 'undefined' && module.exports) {
  module.exports = AgriculturalAI;
}

// For browser usage
if (typeof window !== 'undefined') {
  window.AgriculturalAI = AgriculturalAI;
}

// Example for different chatbot frameworks:

// 1. Discord.js
/*
const { Client, GatewayIntentBits } = require('discord.js');
const AgriculturalAI = require('./chatbot-integration.js');

const client = new Client({ intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildMessages] });
const agriAI = new AgriculturalAI('YOUR_DROPLET_IP');

client.on('messageCreate', async (message) => {
  if (message.content.startsWith('!agri ')) {
    const query = message.content.slice(6);
    const result = await agriAI.processIntelligentQuery(query);
    const response = agriAI.formatResponse(result);
    message.reply(response);
  }
});
*/

// 2. Telegram Bot
/*
const TelegramBot = require('node-telegram-bot-api');
const AgriculturalAI = require('./chatbot-integration.js');

const bot = new TelegramBot('YOUR_BOT_TOKEN', { polling: true });
const agriAI = new AgriculturalAI('YOUR_DROPLET_IP');

bot.on('message', async (msg) => {
  const chatId = msg.chat.id;
  const query = msg.text;
  
  if (query.includes('crop') || query.includes('price') || query.includes('agriculture')) {
    const result = await agriAI.processIntelligentQuery(query);
    const response = agriAI.formatResponse(result);
    bot.sendMessage(chatId, response);
  }
});
*/

// 3. Express.js API
/*
const express = require('express');
const AgriculturalAI = require('./chatbot-integration.js');

const app = express();
const agriAI = new AgriculturalAI('YOUR_DROPLET_IP');

app.use(express.json());

app.post('/chat', async (req, res) => {
  const { message } = req.body;
  const result = await agriAI.processIntelligentQuery(message);
  const response = agriAI.formatResponse(result);
  res.json({ response });
});

app.listen(3000, () => {
  console.log('Chatbot API running on port 3000');
});
*/