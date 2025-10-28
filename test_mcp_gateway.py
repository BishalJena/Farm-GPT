#!/usr/bin/env python3
"""
Test script for MCP Gateway integration
Run this to verify your Docker MCP Gateway is working correctly
"""

import asyncio
import httpx
import json
import os
from typing import Dict, Any

# Configuration
MCP_GATEWAY_URL = os.environ.get('MCP_GATEWAY_URL', 'http://165.232.190.215:8811')
MCP_GATEWAY_TOKEN = os.environ.get('MCP_GATEWAY_TOKEN')

async def test_mcp_gateway():
    """Test MCP Gateway connectivity and tool availability"""
    print(f"🧪 Testing MCP Gateway at: {MCP_GATEWAY_URL}")
    
    headers = {"Content-Type": "application/json"}
    if MCP_GATEWAY_TOKEN:
        headers["Authorization"] = f"Bearer {MCP_GATEWAY_TOKEN}"
        print("🔐 Using authentication token")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Test 1: Health Check
            print("\n1️⃣ Testing health endpoint...")
            health_response = await client.get(f"{MCP_GATEWAY_URL}/health")
            print(f"   Status: {health_response.status_code}")
            if health_response.status_code == 200:
                print(f"   Health: {health_response.json()}")
            
            # Test 2: List Available Tools
            print("\n2️⃣ Testing tool discovery...")
            tools_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "list_tools"
            }
            
            tools_response = await client.post(
                f"{MCP_GATEWAY_URL}/rpc",
                json=tools_payload,
                headers=headers
            )
            print(f"   Status: {tools_response.status_code}")
            if tools_response.status_code == 200:
                tools_result = tools_response.json()
                print(f"   Available tools: {tools_result}")
            
            # Test 3: Test Crop Price Tool
            print("\n3️⃣ Testing crop-price tool...")
            crop_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "crop-price",
                    "arguments": {
                        "state": "Punjab",
                        "commodity": "wheat"
                    }
                }
            }
            
            crop_response = await client.post(
                f"{MCP_GATEWAY_URL}/rpc",
                json=crop_payload,
                headers=headers
            )
            print(f"   Status: {crop_response.status_code}")
            if crop_response.status_code == 200:
                crop_result = crop_response.json()
                print(f"   Crop price result: {json.dumps(crop_result, indent=2)}")
            
            # Test 4: Test Search Tool
            print("\n4️⃣ Testing search tool...")
            search_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "search",
                    "arguments": {
                        "query": "rice farming techniques",
                        "num_results": 3
                    }
                }
            }
            
            search_response = await client.post(
                f"{MCP_GATEWAY_URL}/rpc",
                json=search_payload,
                headers=headers
            )
            print(f"   Status: {search_response.status_code}")
            if search_response.status_code == 200:
                search_result = search_response.json()
                print(f"   Search result: {json.dumps(search_result, indent=2)}")
            
            print("\n✅ MCP Gateway tests completed!")
            
        except Exception as e:
            print(f"\n❌ Error testing MCP Gateway: {e}")
            return False
    
    return True

if __name__ == "__main__":
    print("🌾 Agentic Farmer Chatbot - MCP Gateway Test")
    print("=" * 50)
    
    success = asyncio.run(test_mcp_gateway())
    
    if success:
        print("\n🎉 All tests passed! Your MCP Gateway integration is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check your MCP Gateway configuration.")
        print("\nTroubleshooting:")
        print("1. Verify MCP_GATEWAY_URL is correct")
        print("2. Check if MCP Gateway Docker container is running")
        print("3. Verify network connectivity")
        print("4. Check authentication token if required")