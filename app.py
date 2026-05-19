from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os

# -----------------------------
# GEMINI API CONFIG
# -----------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("models/gemini-1.5-flash")

# -----------------------------
# FASTAPI APP
# -----------------------------
app = FastAPI()

# -----------------------------
# CORS FIX
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# REQUEST MODEL
# -----------------------------
class ChatRequest(BaseModel):
    query: str

# -----------------------------
# HOME ROUTE
# -----------------------------
@app.get("/")
def home():
    return {"status": "ok"}

# -----------------------------
# CHAT ROUTE
# -----------------------------
@app.post("/chat")
def chat(req: ChatRequest):

    prompt = f"""
You are an SHL assessment recommendation assistant.

User Query:
{req.query}

Give response in JSON format only:

{{
  "reply": "short explanation",
  "recommendations": [
    {{
      "name": "assessment name",
      "url": "assessment url",
      "test_type": "type"
    }}
  ],
  "end_of_conversation": false
}}

Recommend suitable SHL assessments based on the query.
"""

    response = model.generate_content(prompt)

    return {
        "reply": response.text
    }
