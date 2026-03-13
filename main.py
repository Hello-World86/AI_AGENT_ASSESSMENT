from fastapi import FastAPI, HTTPException
import agent from agent
app = FastAPI()

@app.get("/")
def root():
    return{"messege": "API is running"}

@app.post("/chat")
def QueryRequest(query:str):
    response = agent(query)

