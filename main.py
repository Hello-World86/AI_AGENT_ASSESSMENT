from fastapi import FastAPI, HTTPException
import agent from agent
app = FastAPI()

@app.get("/")
def root():
    return{"messege": "API is running"}

@app.post("/chat")
def QueryRequest(query:str):
    response = agent(query)
from fastapi import FastAPI
from models import ChatRequest
from agent import run_agent

app = FastAPI()


@app.post("/chat")
def chat(request: ChatRequest):

    answer = run_agent(request.query)

    return {"answer": answer}
