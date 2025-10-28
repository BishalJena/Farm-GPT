#!/usr/bin/env python3
"""
Complete System Integration Test
Tests all components working together for hackathon demo
"""

import asyncio
import httpx
import json
import time
import base64
from typing import Dict, Any

# Test Configuration
BACKEND_URL = "http://localhost:8000"
MCP_URL = "http://localhost:10000"
FRONTEND_URL = "http://localhost:3000"

class SystemIntegrationTest:
    """Complete system integration test suite"""
    
    def __init__(self):
        self.test_results = []
        self.auth_token = None
        self.test_user_email = "integration-test@agricultural-ai.com"
        self.test_user_password = "hackathon2024"
    
    async def run_all_tests(self):
        """Run complete integration test suite"""
        print("🧪 Agricultural AI - Complete System Integration Test")
        print("=" * 60)
        
        # Test sequence
        tests = [
            ("🌐 Frontend Accessibility", self.test_frontend_access),
            ("🔧 Backend API Health", self.test_backend_health),
            ("🛠️ MCP Server Health", self.test_mcp_health),
            ("👤 User Registration", self.test_user_registration),
            ("🔐 User Authentication", self.test_user_login),
            ("🤖 Basic Chat Functionality", self.test_basic_chat),
            ("🌾 Crop Price Tool Integration", self.test_crop_price_tool),
            ("🔍 Search Tool Integration", self.test_search_tool),
            ("🧪 Soil Health Tool Integration", self.test_soil_health_tool),
            ("🌤️ Weather Tool Integration", self.test_weather_tool),
            ("🐛 Pest Identifier Tool Integration", self.test_pest_tool),
            ("💰 Mandi Price Tool Integration", self.test_mandi_price_tool),
            ("🎤 Voice Interface Capabilities", self.test_voice_capabilities),
            ("🚜 Workflow System", self.test_workflow_system),
            ("📊 Metrics Dashboard", self.test_metrics_dashboard),
            ("🌍 Multilingual Support", self.test_multilingual_support),
            ("⚡ Performance Benchmarks", self.test_performance),
        ]
        
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            try:
                print(f"\n{test_name}...")
                result = await test_func()
                if result:
                    print(f"   ✅ PASSED")
                    passed += 1
                else:
                    print(f"   ❌ FAILED")
                    failed += 1
                self.test_results.append({"test": test_name, "passed": result})
            except Exception as e:
                print(f"   ❌ ERROR: {e}")
                failed += 1
                self.test_results.append({"test": test_name, "passed": False, "error": str(e)})
        
        # Summary
        print(f"\n🎯 Test Summary:")
        print(f"   ✅ Passed: {passed}")
        print(f"   ❌ Failed: {failed}")
        print(f"   📊 Success Rate: {(passed/(passed+failed)*100):.1f}%")
        
        if failed == 0:
            print(f"\n🏆 ALL TESTS PASSED! System is ready for hackathon! 🚀")
        else:
            print(f"\n⚠️ Some tests failed. Check the issues above.")
        
        return failed == 0
    
    async def test_frontend_access(self) -> bool:
        """Test frontend accessibility"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(FRONTEND_URL)
                return response.status_code == 200
        except Exception:
            return False
    
    async def test_backend_health(self) -> bool:
        """Test backend API health"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{BACKEND_URL}/api/health")
                if response.status_code == 200:
                    health_data = response.json()
                    return health_data.get("status") == "healthy"
                return False
        except Exception:
            return False
    
    async def test_mcp_health(self) -> bool:
        """Test MCP server health"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{MCP_URL}/health")
                if response.status_code == 200:
                    health_data = response.json()
                    return health_data.get("status") == "healthy"
                return False
        except Exception:
            return False
    
    async def test_user_registration(self) -> bool:
        """Test user registration"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{BACKEND_URL}/api/auth/register",
                    json={
                        "email": self.test_user_email,
                        "password": self.test_user_password
                    }
                )
                return response.status_code in [200, 201, 400]  # 400 if user exists
        except Exception:
            return False
    
    async def test_user_login(self) -> bool:
        """Test user authentication"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{BACKEND_URL}/api/auth/login",
                    json={
                        "email": self.test_user_email,
                        "password": self.test_user_password
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    self.auth_token = data.get("access_token")
                    return self.auth_token is not None
                return False
        except Exception:
            return False
    
    async def test_basic_chat(self) -> bool:
        """Test basic chat functionality"""
        if not self.auth_token:
            return False
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{BACKEND_URL}/api/chat",
                    json={"message": "Hello, I need help with farming"},
                    headers={"Authorization": f"Bearer {self.auth_token}"}
                )
                if response.status_code == 200:
                    data = response.json()
                    return "message" in data and len(data["message"]) > 0
                return False
        except Exception:
            return False
    
    async def test_crop_price_tool(self) -> bool:
        """Test crop price tool integration"""
        if not self.auth_token:
            return False
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{BACKEND_URL}/api/chat",
                    json={"message": "What is the current price of wheat in Punjab?"},
                    headers={"Authorization": f"Bearer {self.auth_token}"}
                )
                if response.status_code == 200:
                    data = response.json()
                    tools_used = data.get("tools_used", [])
                    return "crop-price" in tools_used or len(data.get("message", "")) > 50
                return False
        except Exception:
            return False
    
    async def test_search_tool(self) -> bool:
        """Test search tool integration"""
        if not self.auth_token:
            return False
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{BACKEND_URL}/api/chat",
                    json={"message": "Latest research on organic farming techniques"},
                    headers={"Authorization": f"Bearer {self.auth_token}"}
                )
                if response.status_code == 200:
                    data = response.json()
                    return len(data.get("message", "")) > 50
                return False
        except Exception:
            return False
    
    async def test_soil_health_tool(self) -> bool:
        """Test soil health tool"""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"{MCP_URL}/tools/soil-health",
                    json={
                        "state": "Punjab",
                        "ph_level": 6.5,
                        "npk_values": {"nitrogen": 280, "phosphorus": 23, "potassium": 280}
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("success", False)
                return False
        except Exception:
            return False
    
    async def test_weather_tool(self) -> bool:
        """Test weather tool"""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"{MCP_URL}/tools/weather",
                    json={
                        "location": "Punjab, India",
                        "days": 7
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("success", False)
                return False
        except Exception:
            return False
    
    async def test_pest_tool(self) -> bool:
        """Test pest identifier tool"""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"{MCP_URL}/tools/pest-identifier",
                    json={
                        "crop": "rice",
                        "symptoms": "yellowing leaves, stunted growth"
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("success", False)
                return False
        except Exception:
            return False
    
    async def test_mandi_price_tool(self) -> bool:
        """Test mandi price tool"""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.post(
                    f"{MCP_URL}/tools/mandi-price",
                    json={
                        "commodity": "wheat",
                        "state": "Punjab"
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("success", False)
                return False
        except Exception:
            return False
    
    async def test_voice_capabilities(self) -> bool:
        """Test voice interface capabilities"""
        if not self.auth_token:
            return False
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{BACKEND_URL}/api/voice/capabilities",
                    headers={"Authorization": f"Bearer {self.auth_token}"}
                )
                if response.status_code == 200:
                    data = response.json()
                    return len(data.get("supported_languages", [])) > 0
                return False
        except Exception:
            return False
    
    async def test_workflow_system(self) -> bool:
        """Test workflow system"""
        if not self.auth_token:
            return False
        
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Test available workflows
                response = await client.get(
                    f"{BACKEND_URL}/api/workflows/available",
                    headers={"Authorization": f"Bearer {self.auth_token}"}
                )
                if response.status_code == 200:
                    data = response.json()
                    workflows = data.get("workflows", [])
                    return len(workflows) >= 4  # Should have 4 workflows
                return False
        except Exception:
            return False
    
    async def test_metrics_dashboard(self) -> bool:
        """Test metrics dashboard"""
        if not self.auth_token:
            return False
        
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    f"{BACKEND_URL}/api/metrics/dashboard",
                    headers={"Authorization": f"Bearer {self.auth_token}"}
                )
                if response.status_code == 200:
                    data = response.json()
                    return "performance_metrics" in data and "impact_metrics" in data
                return False
        except Exception:
            return False
    
    async def test_multilingual_support(self) -> bool:
        """Test multilingual support"""
        if not self.auth_token:
            return False
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Test Hindi query
                response = await client.post(
                    f"{BACKEND_URL}/api/chat",
                    json={"message": "गेहूं की खेती के बारे में बताएं"},
                    headers={"Authorization": f"Bearer {self.auth_token}"}
                )
                if response.status_code == 200:
                    data = response.json()
                    return len(data.get("message", "")) > 20
                return False
        except Exception:
            return False
    
    async def test_performance(self) -> bool:
        """Test system performance"""
        if not self.auth_token:
            return False
        
        try:
            # Test response time
            start_time = time.time()
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{BACKEND_URL}/api/chat",
                    json={"message": "Quick farming advice"},
                    headers={"Authorization": f"Bearer {self.auth_token}"}
                )
                
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    # Should respond within 10 seconds (generous for integration test)
                    return response_time < 10.0
                return False
        except Exception:
            return False

async def main():
    """Run the complete system integration test"""
    tester = SystemIntegrationTest()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 System Integration Test: PASSED")
        print("🚀 Your Agricultural AI system is ready for the hackathon!")
    else:
        print("\n⚠️ System Integration Test: FAILED")
        print("🔧 Please check the failed tests and fix issues before demo.")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())