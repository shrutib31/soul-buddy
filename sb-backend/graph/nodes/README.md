# LangGraph Node Structure Guide

This guide explains the standard structure for creating nodes in the Soul Buddy conversation graph.

## Node Architecture

### Overview
Each node in the graph is an async function that:
- Takes `ConversationState` as input
- Returns `Dict[str, Any]` with state updates
- Handles errors gracefully
- Keeps business logic separate from node logic

### File Structure
```
graph/
├── state.py                    # ConversationState definition
├── graph_builder.py           # Graph assembly and compilation
└── nodes/
    ├── NODE_TEMPLATE.py       # Template and examples
    ├── function-nodes/        # Pure function nodes
    │   ├── conv_id_handler.py
    │   ├── risk.py
    │   ├── crisis.py
    │   └── ...
    └── agentic-nodes/         # LLM/AI-powered nodes
        ├── guardrail.py
        ├── response-generator.py
        └── ...
```

## Standard Node Pattern

```python
from typing import Dict, Any
from graph.state import ConversationState

async def my_node(state: ConversationState) -> Dict[str, Any]:
    """
    Brief description of what this node does.
    
    Args:
        state: Current conversation state
    
    Returns:
        Dict with updated state fields
    """
    try:
        # 1. Extract needed values from state
        user_message = state.user_message
        
        # 2. Perform your logic
        result = await process_logic(user_message)
        
        # 3. Return only fields that need updating
        return {
            "response_draft": result,
        }
        
    except Exception as e:
        return {
            "error": f"Error in my_node: {str(e)}"
        }
```

## Key Principles

### 1. Return Only What Changes
✅ **DO**: Return only fields that need updating
```python
return {
    "conversation_id": new_id,
}
```

❌ **DON'T**: Return unchanged fields
```python
return {
    "conversation_id": new_id,
    "mode": mode,  # Unnecessary if unchanged
    "user_message": state.user_message,  # Don't return unchanged
}
```

### 2. Handle Errors Gracefully
✅ **DO**: Catch exceptions and set error field
```python
try:
    result = await risky_operation()
    return {"result": result}
except Exception as e:
    return {"error": f"Operation failed: {str(e)}"}
```

❌ **DON'T**: Let exceptions propagate uncaught
```python
result = await risky_operation()  # Might crash the graph!
return {"result": result}
```

### 3. Keep Business Logic Separate
✅ **DO**: Separate node logic from business logic
```python
async def my_node(state: ConversationState) -> Dict[str, Any]:
    result = await complex_business_logic(state.user_message)
    return {"field": result}

async def complex_business_logic(message: str) -> str:
    # All the complex logic here
    pass
```

❌ **DON'T**: Mix everything in the node function
```python
async def my_node(state: ConversationState) -> Dict[str, Any]:
    # 100 lines of complex logic here...
    pass
```

### 4. Use Type Hints
✅ **DO**: Use proper type hints
```python
async def my_node(state: ConversationState) -> Dict[str, Any]:
    pass
```

❌ **DON'T**: Skip type hints
```python
async def my_node(state):
    pass
```

## State Fields Reference

Available fields in `ConversationState`:

```python
# Core fields
conversation_id: str
mode: str                    # "cognito" | "incognito"
domain: str                  # "student" | "employee" | "general"
user_message: str

# Safety
risk_level: str = "low"      # "low" | "medium" | "high"

# Situation
situation: Optional[str] = None
severity: Optional[str] = None
flow_id: Optional[str] = None

# Flow execution
step_index: int = 0
response_draft: str = ""

# Readiness & solutions
readiness_score: int = 0
tool: Optional[Dict[str, Any]] = None

# Context
page_context: Dict[str, Any] = {}
domain_config: Dict[str, Any] = {}
user_personality_profile: Dict[str, Any] = {}
user_preferences: Dict[str, Any] = {}

# Metadata
error: Optional[str] = None
```

## Common Node Types

### 1. Validation Node
Validates input and sets error if invalid.

```python
async def validation_node(state: ConversationState) -> Dict[str, Any]:
    if not state.conversation_id:
        return {"error": "Missing conversation_id"}
    return {}  # No updates needed
```

### 2. Database Query Node
Queries database and updates state.

```python
async def db_node(state: ConversationState) -> Dict[str, Any]:
    try:
        async with db.get_session() as session:
            result = await session.execute(query)
            data = result.scalar()
        return {"flow_id": data}
    except Exception as e:
        return {"error": str(e)}
```

### 3. Classification Node
Classifies input and sets appropriate state fields.

```python
async def classify_node(state: ConversationState) -> Dict[str, Any]:
    classification = await classify(state.user_message)
    return {
        "situation": classification.situation,
        "severity": classification.severity,
    }
```

### 4. Generation Node
Generates content using LLM.

```python
async def generate_node(state: ConversationState) -> Dict[str, Any]:
    try:
        response = await llm.generate(state.user_message)
        return {"response_draft": response}
    except Exception as e:
        return {"error": str(e)}
```

## Registering Nodes in Graph

In `graph_builder.py`:

```python
from langgraph.graph import StateGraph, END
from graph.state import ConversationState
from graph.nodes.function-nodes.conv_id_handler import conv_id_handler_node

def get_compiled_flow():
    graph = StateGraph(ConversationState)
    
    # Add nodes
    graph.add_node("conv_id_handler", conv_id_handler_node)
    
    # Set entry point
    graph.set_entry_point("conv_id_handler")
    
    # Add edges
    graph.add_edge("conv_id_handler", "next_node")
    
    # Or conditional edges
    graph.add_conditional_edges(
        "conv_id_handler",
        lambda s: "error_node" if s.error else "next_node"
    )
    
    return graph.compile()
```

## Testing Nodes

Example test structure:

```python
import pytest
from graph.state import ConversationState
from graph.nodes.function-nodes.my_node import my_node

@pytest.mark.asyncio
async def test_my_node():
    # Arrange
    state = ConversationState(
        conversation_id="test-id",
        mode="incognito",
        domain="student",
        user_message="test message"
    )
    
    # Act
    result = await my_node(state)
    
    # Assert
    assert "error" not in result
    assert result["expected_field"] == "expected_value"
```

## Best Practices

1. **Keep nodes focused**: Each node should do one thing well
2. **Document thoroughly**: Include docstrings explaining the node's purpose
3. **Handle edge cases**: Consider empty strings, None values, invalid data
4. **Log appropriately**: Use logging for debugging, not print statements
5. **Test independently**: Each node should be testable in isolation
6. **Use constants**: Define magic strings and numbers as constants
7. **Validate inputs**: Check state values before using them
8. **Return early**: Use early returns for error cases

## Example: Complete Node Implementation

See `conv_id_handler.py` for a complete, production-ready example that demonstrates all these principles.
