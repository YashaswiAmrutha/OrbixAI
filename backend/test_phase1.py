"""
Phase 1 Test: Verify LangGraph workflow execution
"""

import asyncio
import sys
import json

# Add backend to path
sys.path.insert(0, '.')

from orchestration.workflow import run_workflow, get_workflow
from orchestration.graph_state import new_state


async def test_workflow():
    """Test the LangGraph workflow end-to-end."""
    print("\n" + "="*60)
    print("Phase 1 Test: LangGraph Workflow Execution")
    print("="*60)
    
    # Test 1: Load workflow
    print("\n[Test 1] Loading workflow...")
    try:
        workflow = get_workflow()
        print("✓ Workflow loaded successfully")
        print(f"  Workflow type: {type(workflow)}")
    except Exception as e:
        print(f"✗ Failed to load workflow: {e}")
        return False
    
    # Test 2: Route general chat query
    print("\n[Test 2] Testing chat routing...")
    try:
        state = await run_workflow("What is your name?", session_id="test-session-1")
        print("✓ Workflow completed")
        print(f"  Intent: {state.get('intent')}")
        print(f"  Module: {state.get('module_name')}")
        print(f"  Response: {state.get('module_output', {}).get('formatted', 'N/A')[:80]}...")
        assert state.get("module_name") == "chat", "Should route to chat"
    except Exception as e:
        print(f"✗ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Route travel query
    print("\n[Test 3] Testing travel routing...")
    try:
        state = await run_workflow("Plan a trip to Paris", session_id="test-session-2")
        print("✓ Workflow completed")
        print(f"  Intent: {state.get('intent')}")
        print(f"  Module: {state.get('module_name')}")
        print(f"  Response: {state.get('module_output', {}).get('formatted', 'N/A')[:80]}...")
        assert state.get("module_name") == "travel", "Should route to travel"
    except Exception as e:
        print(f"✗ Workflow failed: {e}")
        return False
    
    # Test 4: Route action query
    print("\n[Test 4] Testing action routing...")
    try:
        state = await run_workflow("Send an email to john@example.com", session_id="test-session-3")
        print("✓ Workflow completed")
        print(f"  Intent: {state.get('intent')}")
        print(f"  Module: {state.get('module_name')}")
        print(f"  Response: {state.get('module_output', {}).get('formatted', 'N/A')[:80]}...")
        assert state.get("module_name") == "action", "Should route to action"
    except Exception as e:
        print(f"✗ Workflow failed: {e}")
        return False
    
    # Test 5: Check extraction tasks queued
    print("\n[Test 5] Verifying task queuing...")
    try:
        state = await run_workflow("My birthday is March 15", session_id="test-session-4")
        extraction_count = len(state.get("extraction_tasks", []))
        print(f"✓ Tasks queued: {extraction_count} extraction task(s)")
        assert extraction_count > 0, "Should have queued extraction task"
    except Exception as e:
        print(f"✗ Task queueing failed: {e}")
        return False
    
    print("\n" + "="*60)
    print("✓ All Phase 1 tests passed!")
    print("="*60)
    return True


if __name__ == "__main__":
    success = asyncio.run(test_workflow())
    sys.exit(0 if success else 1)
