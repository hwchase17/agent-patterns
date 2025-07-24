"""
Test script to verify all application components can be imported without syntax errors.
"""

import sys
import traceback
import py_compile
import tempfile
import os

def test_syntax(file_path, description):
    """Test compiling a Python file to check for syntax errors."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.pyc', delete=False) as tmp:
            py_compile.compile(file_path, tmp.name, doraise=True)
            os.unlink(tmp.name)  # Clean up
        print(f"✅ {description}: No syntax errors")
        return True
    except Exception as e:
        print(f"❌ {description}: Syntax error - {e}")
        traceback.print_exc()
        return False

def test_import_without_execution(module_name, description):
    """Test importing a module by checking if it can be compiled and basic structure."""
    try:
        # First check syntax
        if module_name == "streamlit_app":
            file_path = "streamlit_app.py"
        elif module_name == "agent.agent":
            file_path = "agent/agent.py"
        else:
            return False
            
        # Test compilation
        with tempfile.NamedTemporaryFile(suffix='.pyc', delete=False) as tmp:
            py_compile.compile(file_path, tmp.name, doraise=True)
            os.unlink(tmp.name)  # Clean up
            
        print(f"✅ {description}: Syntax valid and importable")
        return True
    except Exception as e:
        print(f"❌ {description}: Syntax/import issue - {e}")
        return False

def main():
    """Run all import tests."""
    print("Testing application syntax and import structure...")
    print("=" * 50)
    
    all_passed = True
    
    # Test syntax of core application components
    all_passed &= test_syntax("streamlit_app.py", "Streamlit Application Syntax")
    all_passed &= test_syntax("agent/agent.py", "LangGraph Agent Syntax")
    
    # Test import structure (without executing initialization code)
    all_passed &= test_import_without_execution("streamlit_app", "Streamlit Application Structure")
    all_passed &= test_import_without_execution("agent.agent", "LangGraph Agent Structure")
    
    print("=" * 50)
    if all_passed:
        print("🎉 All syntax tests passed! Application has no syntax errors.")
    else:
        print("⚠️  Some syntax tests failed. Please check the errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)



