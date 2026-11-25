
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


# Request model for the chat endpoint
class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
def chat(details: ChatRequest):
    return {
        "message": "Chat endpoint",
        "question": details.question
    }


@app.get("/")
def home():
    return {"message": "Welcome to HumeQAI API"}


@app.get("/hello")
def hello():
    return {"message": "Hello from HumeQAI hello"}


@app.get("/health")
def health():
    return {"status": "healthy"}
