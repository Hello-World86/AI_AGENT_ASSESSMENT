# Making of the main application that serves the API endpoints and connects to the agent.

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from agent import run_agent

app = FastAPI(
    title="Customer Support AI Agent",
    description=(
        "An AI agent that assists customer support teams by retrieving order details, "
        "customer profiles, and relevant policy information to provide accurate responses "
        "to customer inquiries."
    ),
    version="1.0"
)


# Make Pydantic models for request and response validation

class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[Dict[str, str]]] = None


class ChatResponse(BaseModel):
    answer: str
    steps: List[Any]
    iterations: int


# Define API endpoints

@app.get("/")
def root():
    return {"message": "Customer Support API is running."}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = run_agent(
        query=request.query,
        chat_history=request.chat_history
    )

    return ChatResponse(
        answer=result.get("answer", "Sorry, no answer generated."),
        steps=result.get("steps", []),
        iterations=result.get("iterations", 0)
    )
