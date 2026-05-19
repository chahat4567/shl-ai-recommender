from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
import json

# ----------------------------------------
# FASTAPI APP
# ----------------------------------------

app = FastAPI()

# ----------------------------------------
# ENABLE CORS
# ----------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------
# GEMINI CONFIG
# ----------------------------------------

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel(
    "models/gemini-1.5-flash"
)

# ----------------------------------------
# REQUEST MODEL
# ----------------------------------------

class ChatRequest(BaseModel):
    query: str

# ----------------------------------------
# HOME ROUTE
# ----------------------------------------

@app.get("/")
def home():

    return {
        "status": "ok"
    }

# ----------------------------------------
# HEALTH ROUTE
# ----------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok"
    }

# ----------------------------------------
# CHAT ROUTE
# ----------------------------------------

@app.post("/chat")
def chat(req: ChatRequest):

    prompt = f"""
You are an SHL assessment recommendation assistant.

User Query:
{req.query}

Recommend suitable SHL assessments.

Return ONLY valid JSON in this format:

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

Rules:
- Recommend relevant SHL assessments.
- Keep response professional.
- Mention technical, leadership, aptitude or personality tests if relevant.
- Return only JSON.
"""

    try:

        response = model.generate_content(prompt)

        text_response = response.text.strip()

        # ----------------------------------------
        # REMOVE MARKDOWN JSON BLOCKS IF PRESENT
        # ----------------------------------------

        text_response = text_response.replace(
            "```json",
            ""
        )

        text_response = text_response.replace(
            "```",
            ""
        )

        text_response = text_response.strip()

        # ----------------------------------------
        # CONVERT TO JSON
        # ----------------------------------------

        parsed_json = json.loads(text_response)

        return parsed_json

    except Exception as e:

        return {
            "reply": "Error generating recommendations",
            "error": str(e),
            "recommendations": [],
            "end_of_conversation": False
                }
