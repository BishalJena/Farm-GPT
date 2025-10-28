// 🤖 Your Live DigitalOcean Agricultural AI Integration
// Ready to use in your chatbot application

class AgriculturalAI {
  constructor() {
    // Your confirmed working DigitalOcean endpoint (port 80)
    this.apiBase = 'http://165.232.190.215';
    this.timeout = 10000;
  }

  // ✅ TESTED: Get crop prices
  async getCropPrices(params = {}) {
    const {
      state = null,
      district = null, 
      commodity = null,
      limit = 10
    } = params;

    try {
      const response = await fetch(`${this.apiBase}/tools/crop-price`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ state, district, commodity, limit })
      });

      const data = await response.json();
      return data;
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  // ✅ TESTED: Search agricultural information  
  async searchAgriculture(params = {}) {
    const { query, num_results = 5 } = params;

    try {
      const response = await fetch(`${this.apiBase}/tools/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, num_results })
      });

      const data = await response.json();
      return data;
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  // ✅ TESTED: Health check
  async checkHealth() {
    try {
      const response = await fetch(`${this.apiBase}/health`);
      return await response.json();
    } catch (error) {
      return { success: false, error: error.message };
    }
  }

  // 🧠 Smart query processing for your chatbot
  async processQuery(userMessage) {
    const message = userMessage.toLowerCase();
    
    // Detect intent
    const isPriceQuery = /price|cost|rate|market/.test(message);
    const isSearchQuery = /research|news|info|learn|study/.test(message);
    
    // Extract entities
    const states = ['punjab', 'maharashtra', 'gujarat', 'haryana', 'uttar pradesh'];
    const crops = ['wheat', 'rice', 'cotton', 'maize', 'sugarcane'];
    
    const detectedState = states.find(s => message.includes(s));
    const detectedCrop = crops.find(c => message.includes(c));

    let response = '';

    // Handle price queries
    if (isPriceQuery && (detectedState || detectedCrop)) {
      const priceResult = await this.getCropPrices({
        state: detectedState ? this.capitalize(detectedState) : null,
        commodity: detectedCrop ? this.capitalize(detectedCrop) : null,
        limit: 5
      });

      if (priceResult.success && priceResult.data.records.length > 0) {
        response += '🌾 **Current Crop Prices:**\n';
        priceResult.data.records.slice(0, 3).forEach(record => {
          response += `• ${record.commodity} in ${record.state}: ₹${record.modal_price}/quintal\n`;
        });
        response += `\n📊 Total ${priceResult.data.total} records available\n\n`;
      }
    }

    // Handle search queries
    if (isSearchQuery || (!isPriceQuery && userMessage.length > 10)) {
      const searchResult = await this.searchAgriculture({
        query: userMessage + ' Indian agriculture',
        num_results: 3
      });

      if (searchResult.success && searchResult.data.results.length > 0) {
        response += '📚 **Agricultural Information:**\n';
        searchResult.data.results.slice(0, 2).forEach(item => {
          response += `• **${item.title}**\n`;
          response += `  ${item.text.substring(0, 150)}...\n\n`;
        });
      }
    }

    // Default response if no specific intent detected
    if (!response) {
      response = `🌾 I can help you with:
• Crop prices: "wheat prices in Punjab"
• Agricultural research: "sustainable farming practices"
• Market information: "cotton market trends"

Try asking about specific crops or states!`;
    }

    return response.trim();
  }

  capitalize(str) {
    return str.charAt(0).toUpperCase() + str.slice(1);
  }
}

// ✅ READY TO USE EXAMPLES:

// Example 1: Simple usage
async function example1() {
  const agriAI = new AgriculturalAI();
  
  // Test the connection
  const health = await agriAI.checkHealth();
  console.log('Health:', health);
  
  // Get wheat prices in Punjab
  const prices = await agriAI.getCropPrices({
    state: 'Punjab',
    commodity: 'Wheat'
  });
  console.log('Prices:', prices);
}

// Example 2: Chatbot integration
async function handleChatMessage(userMessage) {
  const agriAI = new AgriculturalAI();
  const response = await agriAI.processQuery(userMessage);
  return response;
}

// Example 3: Discord bot integration
/*
const { Client, GatewayIntentBits } = require('discord.js');
const client = new Client({ intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildMessages, GatewayIntentBits.MessageContent] });

const agriAI = new AgriculturalAI();

client.on('messageCreate', async (message) => {
  if (message.author.bot) return;
  
  if (message.content.startsWith('!agri ')) {
    const query = message.content.slice(6);
    const response = await agriAI.processQuery(query);
    message.reply(response);
  }
});

client.login('YOUR_BOT_TOKEN');
*/

// Example 4: Express.js API
/*
const express = require('express');
const app = express();
app.use(express.json());

const agriAI = new AgriculturalAI();

app.post('/chat', async (req, res) => {
  const { message } = req.body;
  const response = await agriAI.processQuery(message);
  res.json({ response });
});

app.listen(3000);
*/

// Export for Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = AgriculturalAI;
}

// For browser usage
if (typeof window !== 'undefined') {
  window.AgriculturalAI = AgriculturalAI;
}

console.log('🌾 Agricultural AI integration ready!');
console.log('📡 API Base:', 'http://165.232.190.215');
console.log('🧪 Test with: new AgriculturalAI().checkHealth()');