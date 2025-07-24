"""
Test script to verify all application components can be imported without syntax errors.
"""

import sys
import traceback

def test_import(module_name, description):
    """Test importing a module and report results."""
    try:
        __import__(module_name)
        print(f"✅ {description}: Import successful")
        return True
    except Exception as e:
        print(f"❌ {description}: Import failed - {e}")
        traceback.print_exc()
        return False

def main():
    """Run all import tests."""
    print("Testing application imports...")
    print("=" * 50)
    
    all_passed = True
    
    # Test core application components
    all_passed &= test_import("streamlit_app", "Streamlit Application")
    all_passed &= test_import("agent.agent", "LangGraph Agent")
    
    # Test specific components from agent module
    try:
        from agent.agent import graph, get_weather, create_dynamic_prompt
        print("✅ Agent Components: graph, get_weather, create_dynamic_prompt imported successfully")
    except Exception as e:
        print(f"❌ Agent Components: Import failed - {e}")
        all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("🎉 All imports successful! Application is ready to run.")
    else:
        print("⚠️  Some imports failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

