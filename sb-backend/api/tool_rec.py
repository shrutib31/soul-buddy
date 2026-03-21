import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from graph.nodes.function_nodes.tool_recommendation import run_tool_recommendation
import uvicorn

app = FastAPI()

class ToolRecRequest(BaseModel):
    message: str
    personality: dict

@app.post("/tool-recommendation")
async def tool_recommendation_endpoint(request: ToolRecRequest):
    try:
        result = await run_tool_recommendation(
            message=request.message,
            personality=request.personality,
        )
        return {"status": "success", "response": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)