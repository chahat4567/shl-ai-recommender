from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer

import google.generativeai as genai

import faiss
import numpy as np
import json
import os

# ------------------------------------------------
# FASTAPI APP
# ------------------------------------------------

app = FastAPI()

# ------------------------------------------------
# ENABLE CORS
# ------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------
# GEMINI CONFIGURATION
# ------------------------------------------------

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

gemini_model = genai.GenerativeModel(
    "gemini-1.5-flash"
)

# ------------------------------------------------
# LOAD DATASET
# ------------------------------------------------

with open("shl_catalog.json", "r") as f:

    catalog = json.load(f)

# ------------------------------------------------
# LOAD EMBEDDING MODEL
# ------------------------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ------------------------------------------------
# CREATE TEXTS
# ------------------------------------------------

texts = [

    item["name"] + " " +

    item["description"] + " " +

    item["test_type"]

    for item in catalog
]

# ------------------------------------------------
# CREATE EMBEDDINGS
# ------------------------------------------------

embeddings = embedding_model.encode(texts)

# ------------------------------------------------
# CREATE FAISS INDEX
# ------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

# ------------------------------------------------
# REQUEST MODEL
# ------------------------------------------------

class ChatRequest(BaseModel):

    messages: list

# ------------------------------------------------
# OFF TOPIC FILTER
# ------------------------------------------------

def is_off_topic(user_message):

    blocked_topics = [

        "politics",
        "religion",
        "sports",
        "weather",
        "medical",
        "legal"
    ]

    user_message = user_message.lower()

    return any(
        word in user_message
        for word in blocked_topics
    )

# ------------------------------------------------
# RETRIEVAL FUNCTION
# ------------------------------------------------

def retrieve_assessments(query, top_k=10):

    query_embedding = embedding_model.encode(
        [query]
    )

    distances, indices = index.search(
        np.array(query_embedding),
        top_k
    )

    results = []

    for idx in indices[0]:

        results.append(catalog[idx])

    return results

# ------------------------------------------------
# HEALTH ENDPOINT
# ------------------------------------------------

@app.get("/health")

def health():

    return {
        "status": "ok"
    }

# ------------------------------------------------
# CHAT ENDPOINT
# ------------------------------------------------

@app.post("/chat")

def chat(req: ChatRequest):

    # --------------------------------------------
    # BUILD CONVERSATION CONTEXT
    # --------------------------------------------

    conversation_text = ""

    latest_user_message = ""

    for msg in req.messages:

        role = msg["role"]

        content = msg["content"]

        conversation_text += f"{role}: {content}\n"

        if role == "user":

            latest_user_message = content

    # --------------------------------------------
    # OFF TOPIC CHECK
    # --------------------------------------------

    if is_off_topic(latest_user_message):

        return {

            "reply":
            "I can only help with SHL assessment recommendations and hiring assessments.",

            "recommendations": [],

            "end_of_conversation": False
        }

    # --------------------------------------------
    # END CONVERSATION CHECK
    # --------------------------------------------

    closing_words = [

        "thanks",
        "thank you",
        "perfect",
        "great",
        "that works",
        "sounds good"
    ]

    if any(

        word in latest_user_message.lower()

        for word in closing_words
    ):

        return {

            "reply":
            "Glad I could help with your SHL assessment recommendations.",

            "recommendations": [],

            "end_of_conversation": True
        }

    # --------------------------------------------
    # SMART CLARIFICATION
    # --------------------------------------------

    short_query = len(latest_user_message.split()) < 4

    if short_query:

        return {

            "reply":
            "Could you provide more details about the role, required skills, experience level, or hiring objective?",

            "recommendations": [],

            "end_of_conversation": False
        }

    # --------------------------------------------
    # CONTEXT-AWARE QUERY
    # --------------------------------------------

    weighted_query = (
        latest_user_message + " " +
        latest_user_message + " " +
        conversation_text
    )

    # --------------------------------------------
    # RETRIEVE ASSESSMENTS
    # --------------------------------------------

    retrieved = retrieve_assessments(
        weighted_query
    )

    # --------------------------------------------
    # FORMAT RETRIEVED RESULTS
    # --------------------------------------------

    retrieved_text = ""

    for item in retrieved[:5]:

        retrieved_text += (

            f"Assessment: {item['name']}\n"

            f"Type: {item['test_type']}\n"

            f"Description: {item['description']}\n"

            f"URL: {item['url']}\n\n"
        )

    # --------------------------------------------
    # GEMINI PROMPT
    # --------------------------------------------

    prompt = f"""
You are an SHL assessment recommendation assistant.

Conversation:
{conversation_text}

Relevant SHL assessments:
{retrieved_text}

Instructions:
- Understand the hiring context carefully.
- Respond conversationally and professionally.
- Recommend the most relevant assessments.
- Explain WHY each assessment is useful.
- Ask follow-up questions if information is incomplete.
- Keep the response concise but informative.
"""

    # --------------------------------------------
    # GEMINI RESPONSE
    # --------------------------------------------

    gemini_response = gemini_model.generate_content(
        prompt
    )

    reply = gemini_response.text

    # --------------------------------------------
    # STRUCTURED RECOMMENDATIONS
    # --------------------------------------------

    recommendations = [

        {
            "name": item["name"],

            "url": item["url"],

            "test_type": item["test_type"]
        }

        for item in retrieved[:10]
    ]

    # --------------------------------------------
    # FINAL RESPONSE
    # --------------------------------------------

    return {

        "reply": reply,

        "recommendations": recommendations,

        "end_of_conversation": False
    }
    
