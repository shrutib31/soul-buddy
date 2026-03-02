import sys
import os
from dataclasses import dataclass

# --- SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

@dataclass
class MockState:
    user_message: str

try:
    # Use your actual masking node
    from graph.nodes.function_nodes.privacy_masking import new_masking_node as privacy_masking_node
except ImportError:
    print("❌ CRITICAL: Could not import masking node.")
    sys.exit(1)

def run_test_case(name, input_text, expected_tags, exact_match=False):
    state = MockState(user_message=input_text)
    result = privacy_masking_node(state)
    output_text = result.get("user_message", "")
    
    passed = True
    if exact_match:
        passed = (output_text == expected_tags[0])
    else:
        for tag in expected_tags:
            if tag not in output_text:
                passed = False

    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {name.ljust(25)} | In: {input_text[:30].ljust(30)}... -> Out: {output_text}")
    return passed

if __name__ == "__main__":
    print("\n🛡️  STARTING 50-POINT PRIVACY SHIELD STRESS TEST 🛡️\n" + "="*95)
    
    results = []

    # --- 1. INDIAN NAMES (10 Unique Cases) ---
    tests_names = [
        ("Bengali Name", "Anirban Chatterjee requested a refund.", ["[NAME]"]),
        ("South Indian Name", "Patient Venkatakrishnan is in Room 402.", ["[NAME]"]),
        ("Sikh Name", "Consultation with Harpreet Singh.", ["[NAME]"]),
        ("Marathi Name", "Met with Rahul Deshmukh at the bank.", ["[NAME]"]),
        ("Parsi Name", "Contact Cyrus Ponawalla for the docs.", ["[NAME]"]),
        ("Malayali Name", "Dr. Lakshmi Nair is the surgeon.", ["[NAME]"]),
        ("Assamese Name", "Arnab Baruah lives in Guwahati.", ["[NAME]"]),
        ("Common Name 1", "Meeting with Priya Sharma.", ["[NAME]"]),
        ("Common Name 2", "Vikram Singh signed the form.", ["[NAME]"]),
        ("Initials Name", "V.S. Naipaul wrote this note.", ["[NAME]"]),
    ]

    # --- 2. GOVERNMENT IDS ---
    tests_ids = [
        ("Aadhaar Standard", "My Aadhaar is 1234 5678 9012", ["[AADHAAR]"]),
        ("Aadhaar Hyphen", "UID: 1122-3344-5566", ["[AADHAAR]"]),
        ("Aadhaar Continuous", "Number 998877665544", ["[AADHAAR]"]),
        ("PAN Card", "My PAN is ABCDE1234F", ["[PAN_CARD]"]),
        ("PAN Card Lower", "pan: pqrst5566m", ["[PAN_CARD]"]),
    ]

    # --- 3. CONTACT INFO ---
    tests_contact = [
        ("Mobile +91", "Call +91 9876543210", ["[PHONE_NUMBER]"]),
        ("Mobile Simple", "My number is 8877665544", ["[PHONE_NUMBER]"]),
        ("Test Mobile", "1234567890", ["[PHONE_NUMBER]"]),
        ("Email Work", "reach out at sam@company.co.in", ["[EMAIL]"]),
        ("Email Personal", "my mail is test.user@gmail.com", ["[EMAIL]"]),
        ("UPI Handle 1", "Pay me at user@okaxis", ["[UPI_ID]"]),
        ("UPI Handle 2", "upi: rahul.sharma@ybl", ["[UPI_ID]"]),
        ("UPI Handle 3", "Transfer to 9999988888@paytm", ["[UPI_ID]"]),
    ]

    # --- 4. LOCATIONS & ORGS ---
    tests_ai = [
        ("Location City", "I am in Bangalore today", ["[LOCATION]"]),
        ("Location State", "The patient moved to Maharashtra", ["[LOCATION]"]),
        ("Organization 1", "I work at Tata Consultancy Services", ["[ORGANIZATION]"]),
        ("Organization 2", "Reliance Industries reported growth", ["[ORGANIZATION]"]),
        ("Specific Office", "Office at Prestige Tech Park", ["[LOCATION]"]),
    ]

    # --- 5. FALSE POSITIVE CHECKS (SHOULD NOT MASK) ---
    tests_false_positives = [
        ("Amount Check", "I owe you 5000 rupees", ["5000"], True),
        ("Date Check", "Meeting on 25-12-2024", ["25-12-2024"], True),
        ("Short Number", "My lucky number is 123", ["123"], True),
        ("PIN Code", "The PIN code is 560001", ["560001"], True),
        ("Transaction ID", "TXN987654321", ["TXN987654321"], True),
    ]

    # --- 6. PHI & MEDICAL (INDIAN CONTEXT) ---
    tests_phi = [
        ("ABHA ID", "My ABHA ID is 14-1234-5678-9012", ["14-1234-5678-9012"]), # Regex would catch as Aadhaar if not careful
        ("Hospital Name", "Admitted to Apollo Hospital", ["[ORGANIZATION]"]),
        ("Prescription", "Dr. Gupta prescribed Paracetamol", ["[NAME]"]),
        ("Medical History", "Patient Arjun has a history of asthma", ["[NAME]"]),
    ]

    # --- 7. STRESS & MIXED ---
    tests_stress = [
        ("Mixed IDs", "Aadhaar 111122223333, PAN ABCDE1234F", ["[AADHAAR]", "[PAN_CARD]"]),
        ("Deduplication Test", "Sam Sam Sam", ["[NAME]"]),
        ("Sentence Mixed", "Arnav from Mumbai used UPI arnav@okaxis", ["[NAME]", "[LOCATION]", "[UPI_ID]"]),
        ("Continuous Mobile", "Phone 919876543210 is active", ["[PHONE_NUMBER]"]),
        ("Lowercase Org", "it was reported by infosys limited", ["[ORGANIZATION]"]),
        ("Address string", "Residence at 123, MG Road, Mumbai", ["[LOCATION]"]),
        ("Large Number", "The population is 1400000000", ["1400000000"], True),
        ("Email with numbers", "user123.test@domain.com", ["[EMAIL]"]),
        ("Phone with +91", "Contact me at +919988776655", ["[PHONE_NUMBER]"]),
        ("Name in Address", "Deliver to Sunil, HSR Layout", ["[NAME]", "[LOCATION]"]),
    ]

    # Combine all into the 50-test suite
    all_tests = (tests_names + tests_ids + tests_contact + tests_ai + 
                 tests_false_positives + tests_phi + tests_stress)

    # Ensure we are at exactly 50
    all_tests = all_tests[:50]

    for name, text, tags, *exact in all_tests:
        res = run_test_case(name, text, tags, exact[0] if exact else False)
        results.append(res)

    print("="*95)
    print(f"TOTAL TESTS: {len(all_tests)} | PASSED: {sum(results)} | FAILED: {len(all_tests) - sum(results)}")