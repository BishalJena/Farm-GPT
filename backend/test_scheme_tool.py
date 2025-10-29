#!/usr/bin/env python3
"""
Test script for the integrated scheme tool
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agricultural_tools import AgriculturalTools

async def test_scheme_tool():
    """Test the scheme tool"""
    print("Testing integrated scheme tool...")
    
    tools = AgriculturalTools()
    
    # Test parameters for drought damage
    params = {
        'damage_type': 'drought',
        'crop_type': 'wheat',
        'state': 'Punjab',
        'district': 'Ludhiana',
        'damage_extent': 'moderate',
        'has_insurance': False,
        'land_size_acres': 3.0,
        'farmer_category': 'small'
    }
    
    print(f"Calling scheme-tool with params: {params}")
    
    try:
        result = await tools.scheme_tool(params)
        
        if result.get('success'):
            data = result.get('data', {})
            print(f"✅ Success!")
            print(f"Estimated compensation: ₹{data.get('estimated_compensation', 0)}")
            print(f"Applicable schemes: {data.get('applicable_schemes_count', 0)}")
            
            # Show immediate actions
            immediate_actions = data.get('action_plan', {}).get('immediate_actions', [])
            if immediate_actions:
                print(f"\nImmediate actions ({len(immediate_actions)}):")
                for i, action in enumerate(immediate_actions[:2]):
                    print(f"  {i+1}. {action.get('action')}")
                    print(f"     Timeline: {action.get('timeline')}")
                    print(f"     Contact: {action.get('contact')}")
                    print()
            
            # Show top schemes
            recommendations = data.get('recommendations', [])
            if recommendations:
                print(f"Top applicable schemes:")
                for i, scheme in enumerate(recommendations[:2]):
                    print(f"  {i+1}. {scheme.get('name')}")
                    print(f"     Category: {scheme.get('category')}")
                    print(f"     Priority: {scheme.get('priority')}")
                    print()
                    
        else:
            print(f"❌ Error: {result.get('error')}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_scheme_tool())