#!/usr/bin/env python3
"""
Test script for the new scheme-tool and phone authentication
"""

import asyncio
import httpx
import json

# Test Configuration
BACKEND_URL = "http://localhost:8000"
MCP_URL = "http://localhost:10000"  # fs-gate MCP server

async def test_phone_auth():
    """Test phone number authentication"""
    print("🔐 Testing Phone Number Authentication...")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        # Test send OTP
        print("  📱 Testing send OTP...")
        response = await client.post(
            f"{BACKEND_URL}/api/auth/send-otp",
            json={"phone_number": "9876543210"}
        )
        print(f"     Status: {response.status_code}")
        if response.status_code == 200:
            print(f"     Response: {response.json()}")
        
        # Test verify OTP
        print("  🔑 Testing verify OTP...")
        response = await client.post(
            f"{BACKEND_URL}/api/auth/verify-otp",
            json={"phone_number": "9876543210", "otp": "7521"}
        )
        print(f"     Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"     Token received: {data.get('access_token')[:20]}...")
            print(f"     Phone: {data.get('phone_number')}")
            print(f"     Is new user: {data.get('is_new_user')}")
            return data.get('access_token')
        
        return None

async def test_scheme_tool():
    """Test scheme-tool functionality"""
    print("\n🛡️ Testing Scheme Tool...")
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        # Test scheme-tool directly
        print("  🌾 Testing crop damage assistance...")
        response = await client.post(
            f"{MCP_URL}/tools/scheme-tool",
            json={
                "damage_type": "flood",
                "crop_type": "rice",
                "state": "Punjab",
                "district": "Ludhiana",
                "damage_extent": "severe",
                "has_insurance": True,
                "insurance_type": "pmfby",
                "land_size_acres": 5.0
            }
        )
        print(f"     Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                result = data.get("data", {})
                print(f"     ✅ Estimated compensation: ₹{result.get('estimated_compensation', 0):,}")
                print(f"     📋 Schemes found: {result.get('applicable_schemes_count', 0)}")
                print(f"     🚨 Action plan steps: {len(result.get('action_plan', {}).get('immediate_actions', []))}")
                return True
            else:
                print(f"     ❌ Tool failed: {data}")
        else:
            print(f"     ❌ HTTP Error: {response.text}")
        
        return False

async def test_chat_with_scheme_tool(token):
    """Test chat integration with scheme-tool"""
    if not token:
        print("\n⚠️ Skipping chat test - no auth token")
        return
    
    print("\n💬 Testing Chat Integration with Scheme Tool...")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test scheme-tool query
        print("  🌾 Testing crop damage query...")
        response = await client.post(
            f"{BACKEND_URL}/api/chat",
            json={"message": "My rice crop is damaged due to flood in Punjab. I have PMFBY insurance. Please help me with schemes and compensation."},
            headers={"Authorization": f"Bearer {token}"}
        )
        print(f"     Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"     🛡️ Tools used: {data.get('tools_used', [])}")
            print(f"     📝 Response length: {len(data.get('message', ''))}")
            if 'scheme-tool' in data.get('tools_used', []):
                print("     ✅ Scheme-tool successfully integrated!")
                print(f"     💬 Response preview: {data.get('message', '')[:200]}...")
                return True
            else:
                print("     ⚠️ Scheme-tool not triggered")
        else:
            print(f"     ❌ Chat failed: {response.text}")
        
        return False

async def test_mcp_health():
    """Test MCP server health"""
    print("\n🏥 Testing MCP Server Health...")
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(f"{MCP_URL}/health")
        print(f"     Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            tools = data.get("tools", [])
            print(f"     🛠️ Available tools: {len(tools)}")
            if "scheme-tool" in tools:
                print("     ✅ Scheme-tool available!")
                return True
            else:
                print("     ❌ Scheme-tool not found in tools")
        
        return False

async def main():
    """Run all tests"""
    print("🧪 CropGPT Development Changes - Test Suite")
    print("=" * 50)
    
    results = {
        "mcp_health": False,
        "phone_auth": False,
        "scheme_tool": False,
        "chat_integration": False
    }
    
    # Test MCP server health
    results["mcp_health"] = await test_mcp_health()
    
    # Test phone authentication
    token = await test_phone_auth()
    results["phone_auth"] = token is not None
    
    # Test scheme-tool
    results["scheme_tool"] = await test_scheme_tool()
    
    # Test chat integration
    results["chat_integration"] = await test_chat_with_scheme_tool(token)
    
    # Summary
    print("\n📊 Test Results Summary:")
    print("=" * 30)
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {test_name.replace('_', ' ').title()}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\n🎯 Overall: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("🎉 All tests passed! Your changes are working correctly!")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return total_passed == total_tests

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)