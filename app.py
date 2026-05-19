from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
import os
import json

app = FastAPI()

# Gemini API Key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("models/gemini-1.5-flash")

class QueryRequest(BaseModel):
    query: str


@app.get("/")
def home():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: QueryRequest):

    user_query = req.query

    prompt = f"""
You are an SHL assessment recommendation assistant.

Your task:
- Ask follow-up questions if information is incomplete.
- Recommend SHL assessments only when enough details are available.
- Return response ONLY in JSON format.

Required JSON structure:

{{
  "reply": "text response",
  "recommendations": [
    {{
      "name": "assessment name",
      "url": "assessment url",
      "test_type": "test category"
    }}
  ],
  "end_of_conversation": true
}}

Rules:
- If details are insufficient:
  - ask a follow-up question
  - recommendations should be []
  - end_of_conversation = false

- If enough details are available:
  - provide recommendations
  - end_of_conversation = true

User query:
{user_query}
"""

    response = model.generate_content(prompt)

    try:
        cleaned = response.text.strip()

        if cleaned.startswith("```json"):
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        result = json.loads(cleaned)

        return result

    except Exception as e:
        return {
            "reply": response.text,
            "recommendations": [],
            "end_of_conversation": False
        }
