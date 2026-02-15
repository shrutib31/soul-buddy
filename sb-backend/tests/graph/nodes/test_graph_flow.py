import sys
import os
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

# --- SETUP PATH ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import your node
from graph.nodes.function_nodes.privacy_masking import privacy_masking_node

# 1. Define a Minimal State for testing
class TestState(BaseModel):
    user_message: str

def test_graph_flow():
    print("\n🔗 TESTING GRAPH FLOW INTEGRATION\n")

    # 2. Build a Tiny Graph (Start -> Privacy -> End)
    workflow = StateGraph(TestState)
    
    workflow.add_node("privacy_shield", privacy_masking_node)
    
    # Set Privacy as the ENTRY POINT (This is what we want to verify)
    workflow.set_entry_point("privacy_shield")
    
    # End immediately after masking
    workflow.add_edge("privacy_shield", END)
    
    app = workflow.compile()

    # 3. Run the Graph
    input_data = TestState(user_message="My name is Robert Paulson.")
    
    print(f"Input State:  {input_data.user_message}")
    
    # Execute
    result = app.invoke(input_data)
    
    # 4. Check the Final State
    final_msg = result["user_message"]
    print(f"Final State:  {final_msg}")

    if "<NAME>" in final_msg and "Robert" not in final_msg:
        print("\n✅ SUCCESS: The graph successfully routed through the Privacy Node.")
    else:
        print("\n❌ FAILURE: The graph skipped the Privacy Node or failed to mask.")

if __name__ == "__main__":
    test_graph_flow()