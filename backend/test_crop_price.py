#!/usr/bin/env python3
"""
Test script for the integrated crop price tool
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agricultural_tools import AgriculturalTools

async def test_crop_price():
    """Test the crop price tool with real API call"""
    print("Testing integrated crop price tool...")
    
    tools = AgriculturalTools()
    
    # Test parameters
    params = {
        'state': 'Punjab',
        'district': 'Ludhiana', 
        'commodity': 'Wheat',
        'limit': 5
    }
    
    print(f"Calling crop-price tool with params: {params}")
    
    try:
        result = await tools.crop_price_tool(params)
        
        if result.get('success'):
            data = result.get('data', {})
            records = data.get('records', [])
            print(f"✅ Success! Retrieved {len(records)} records")
            print(f"Total records available: {data.get('total', 0)}")
            
            if records:
                print("\nSample records:")
                for i, record in enumerate(records[:3]):
                    print(f"  {i+1}. Market: {record.get('Market')}")
                    print(f"     Date: {record.get('Arrival_Date')}")
                    print(f"     Price: ₹{record.get('Modal_Price')} per quintal")
                    print(f"     Variety: {record.get('Variety')}")
                    print()
        else:
            print(f"❌ Error: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_crop_price())