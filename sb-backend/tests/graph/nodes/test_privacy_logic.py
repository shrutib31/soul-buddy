import sys
import os
from dataclasses import dataclass

# --- SETUP: Add the parent directory to path so we can import your nodes ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

# 1. Mock the State (so we don't need your full DB connection)
@dataclass
class MockState:
    user_message: str

# 2. Import your actual node 
# (Adjust this import path if your folder structure is different)
try:
    from graph.nodes.function_nodes.privacy_masking import privacy_masking_node
except ImportError:
    print("❌ CRITICAL: Could not import 'privacy_masking_node'.")
    print("   Make sure you created the file 'graph/nodes/function_nodes/privacy_masking.py'")
    sys.exit(1)

def run_test_case(name, input_text, expected_tokens):
    print(f"🧪 TESTING: {name}")
    print(f"   Input:    '{input_text}'")
    
    # Create state and run node
    state = MockState(user_message=input_text)
    result = privacy_masking_node(state)
    output_text = result.get("user_message", "")
    
    print(f"   Output:   '{output_text}'")
    
    # Verification
    passed = True
    missing_tokens = []
    for token in expected_tokens:
        if token not in output_text:
            passed = False
            missing_tokens.append(token)
            
    if passed:
        print("   Status:   ✅ PASS")
    else:
        print(f"   Status:   ❌ FAIL (Missing: {missing_tokens})")
    print("-" * 50)

if __name__ == "__main__":
    print("\n🛡️  STARTING PRIVACY SHIELD UNIT TESTS 🛡️\n" + "="*50)

    # TEST 1: Basic Personal Info
    run_test_case(
        "Basic Identity",
        "My name is Sarah Connor and my email is sarah@sky.net",
        ["<NAME>", "<EMAIL>"]
    )

    # TEST 2: Medical Record Number (Custom Regex Check)
    run_test_case(
        "Medical Record ID",
        "Patient ID: 99281 needs immediate assistance.",
        ["<MRN_ID>"] # If this fails, your custom regex in privacy.py is off
    )

    # TEST 3: Phone Numbers
    run_test_case(
        "Phone Number Formats",
        "Call me at 555-0199 or (555) 123-4567.",
        ["<PHONE>"]
    )

    # TEST 4: Mixed / Complex
    run_test_case(
        "Complex Sentence",
        "Hi, I am John. MRN-44512. I live in Chicago.",
        ["<NAME>", "<MRN_ID>"] # Note: Chicago might be masked as <LOCATION> or not depending on the model model
    )

    # TEST 5: No PII (Control Group)
    run_test_case(
        "No Sensitive Data",
        "I am feeling sad today.",
        ["I am feeling sad today."] # Should remain exactly the same
    )