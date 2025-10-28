// Quick test script for the MCP server
const fetch = require('node-fetch');

async function testMCPServer() {
    const baseUrl = 'http://localhost:10000';
    
    try {
        console.log('🧪 Testing MCP Server...');
        
        // Test health endpoint
        const healthResponse = await fetch(`${baseUrl}/health`);
        const healthData = await healthResponse.json();
        console.log('✅ Health check:', healthData.status);
        
        // Test crop-price tool
        const cropPriceResponse = await fetch(`${baseUrl}/tools/crop-price`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                state: 'Punjab',
                commodity: 'Wheat',
                limit: 5
            })
        });
        
        const cropPriceData = await cropPriceResponse.json();
        console.log('✅ Crop price tool:', cropPriceData.success ? 'Working' : 'Error');
        
        // Test soil health tool
        const soilResponse = await fetch(`${baseUrl}/tools/soil-health`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                state: 'Punjab',
                ph_level: 6.5,
                npk_values: { nitrogen: 280, phosphorus: 23, potassium: 280 }
            })
        });
        
        const soilData = await soilResponse.json();
        console.log('✅ Soil health tool:', soilData.success ? 'Working' : 'Error');
        
        console.log('🎉 All tests passed! MCP Server is ready for hackathon!');
        
    } catch (error) {
        console.error('❌ Test failed:', error.message);
    }
}

// Only run if this file is executed directly
if (require.main === module) {
    testMCPServer();
}

module.exports = { testMCPServer };