// Quick test script to verify the MCP tool works
// Run with: node test-tool.js

import { spawn } from 'child_process';

const testMCPTool = () => {
    console.log('🧪 Testing crop-price MCP tool...');
    
    // Sample MCP request
    const mcpRequest = {
        jsonrpc: "2.0",
        id: 1,
        method: "tools/call",
        params: {
            name: "crop-price",
            arguments: {
                state: "Punjab",
                commodity: "Wheat", 
                limit: 5
            }
        }
    };

    const server = spawn('node', ['dist/server.js'], {
        stdio: ['pipe', 'pipe', 'pipe']
    });

    server.stdin.write(JSON.stringify(mcpRequest) + '\n');
    server.stdin.end();

    server.stdout.on('data', (data) => {
        console.log('📊 Server response:', data.toString());
    });

    server.stderr.on('data', (data) => {
        console.error('❌ Server error:', data.toString());
    });

    server.on('close', (code) => {
        console.log(`✅ Test completed with exit code ${code}`);
    });
};

// Check if built
import { existsSync } from 'fs';
if (!existsSync('./dist/server.js')) {
    console.log('⚠️  Please run "npm run build" first');
    process.exit(1);
}

testMCPTool();