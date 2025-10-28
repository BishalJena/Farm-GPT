#!/usr/bin/env python3
"""Test script to check all imports"""

try:
    from database import get_database
    print("✅ database import OK")
except ImportError as e:
    print(f"❌ database import failed: {e}")

try:
    from models import User, ChatMessage, Conversation
    print("✅ models import OK")
except ImportError as e:
    print(f"❌ models import failed: {e}")

try:
    from cultural_context import CulturalContextManager
    print("✅ cultural_context import OK")
except ImportError as e:
    print(f"❌ cultural_context import failed: {e}")

try:
    from conversational_memory import ConversationalMemory, FarmProfile
    print("✅ conversational_memory import OK")
except ImportError as e:
    print(f"❌ conversational_memory import failed: {e}")

try:
    from agricultural_rag import AgriculturalRAG
    print("✅ agricultural_rag import OK")
except ImportError as e:
    print(f"❌ agricultural_rag import failed: {e}")

try:
    from voice_interface import VoiceInterface
    print("✅ voice_interface import OK")
except ImportError as e:
    print(f"❌ voice_interface import failed: {e}")

try:
    from workflow_engine import WorkflowEngine
    print("✅ workflow_engine import OK")
except ImportError as e:
    print(f"❌ workflow_engine import failed: {e}")

try:
    from metrics_system import MetricsSystem
    print("✅ metrics_system import OK")
except ImportError as e:
    print(f"❌ metrics_system import failed: {e}")

print("Import test completed!")