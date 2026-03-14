from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List, Dict

from agent import run_agent

app = FastAPI(
    title = "Customer Support AI Agent",
    description = "An AI agent that assists customer support teams by retrieving order details, customer profiles, and relevant policy information to provide accurate responses to customer inquiries.",
    version = "1.0"
)

class ChatRequest(BaseModel):
    query: str
    chat_history: Optional[List[Dict[str,str]]] = None

class ChatResponse(BaseModel):
    answer: str
    steps: list[Dict[str, str]]
    iterations: int

@app.get("/")
def root():
    return{"messege": "Customer Support API is running."}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    result = run_agent(
        query = request.query,
        chat_history = request.chat_history
    )

    return ChatResponse(
        answer = result.get("answer", "Sorry, No answer generated."),
        steps = result.get("steps", []),
        iterations = result.get("iterations", 0)
    )