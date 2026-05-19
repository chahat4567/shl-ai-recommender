from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os

app = FastAPI()

# Gemini API Key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")


class QueryRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: QueryRequest):

    prompt = f"""
    Recommend SHL assessments for this requirement:

    {req.query}

    Return:
    - assessment name
    - type
    - short description
    """

    response = model.generate_content(prompt)

    return {
        "reply": response.text
    }
