import httpx
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MCPGatewayClient:
    def __init__(self, gateway_url: str, token: Optional[str] = None):
        self.gateway_url = gateway_url.rstrip('/')
        self.token = token
        self.headers = {'Content-Type': 'application/json'}
        if token:
            self.headers['Authorization'] = f'Bearer {token}'
    
    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool via the MCP gateway"""
        try:
            url = f"{self.gateway_url}/tools/{tool_name}"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=parameters, headers=self.headers)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"MCP tool call failed: {response.status_code} - {response.text}")
                    return {"error": f"HTTP {response.status_code}: {response.text}"}
                    
        except Exception as e:
            logger.error(f"Error calling MCP tool {tool_name}: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """Check MCP gateway health"""
        try:
            url = f"{self.gateway_url}/health"
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
                    
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}