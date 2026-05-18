
from fastapi import FastAPI
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer

import faiss
import numpy as np
import json

# ------------------------------------------------
# FASTAPI APP
# ------------------------------------------------

app = FastAPI()

# ------------------------------------------------
# LOAD CLEAN DATASET
# ------------------------------------------------

with open("shl_catalog.json", "r") as f:
    catalog = json.load(f)

# ------------------------------------------------
# EMBEDDING MODEL
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
# EMBEDDINGS
# ------------------------------------------------

embeddings = embedding_model.encode(texts)

# ------------------------------------------------
# FAISS INDEX
# ------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

# ------------------------------------------------
# REQUEST SCHEMA
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
        "weather",
        "medical",
        "sports",
        "legal"
    ]
    user_message = user_message.lower()
    return any(
        word in user_message
        for word in blocked_topics
    )

# ------------------------------------------------
# CLARIFICATION LOGIC
# ------------------------------------------------

def needs_clarification(user_message):
    important_keywords = [
        "developer",
        "engineer",
        "manager",
        "leadership",
        "sales",
        "java",
        "python",
        "coding",
        "personality",
        "analytical"
    ]
    user_message = user_message.lower()
    has_details = any(
        word in user_message
        for word in important_keywords
    )
    return not has_details

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
# RESPONSE GENERATOR
# ------------------------------------------------

def generate_reply(retrieved_assessments):
    reply = (
        "Based on your requirements, "
        "these SHL assessments are recommended:\n\n"
    )
    for assessment in retrieved_assessments[:5]:
        reply += (
            f"- {assessment['name']}\n"
            f"  Type: {assessment['test_type']}\n"
            f"  {assessment['description']}\n"
            f"  URL: {assessment['url']}\n\n"
        )
    return reply

# ------------------------------------------------
# HEALTH ENDPOINT
# ------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

# ------------------------------------------------
# CHAT ENDPOINT
# ------------------------------------------------

@app.post("/chat")
def chat(req: ChatRequest):
    latest_user_message = ""
    for msg in reversed(req.messages):
        if msg["role"] == "user":
            latest_user_message = msg["content"]
            break

    # --------------------------------------------
    # OFF TOPIC
    # --------------------------------------------

    if is_off_topic(latest_user_message):
        return {
            "reply":
                "I can only help with SHL assessment recommendations.",
            "recommendations": [],
            "end_of_conversation": False
        }

    # --------------------------------------------
    # CLARIFICATION
    # --------------------------------------------

    if needs_clarification(latest_user_message):
        return {
            "reply":
                "Could you provide more details about the role, skills, seniority level, or assessment requirements?",
            "recommendations": [],
            "end_of_conversation": False
        }

    # --------------------------------------------
    # RETRIEVAL
    # --------------------------------------------

    retrieved = retrieve_assessments(
        latest_user_message
    )

    # --------------------------------------------
    # RESPONSE
    # --------------------------------------------

    reply = generate_reply(
        retrieved
    )

    recommendations = [
        {
            "name": item["name"],
            "url": item["url"],
            "test_type": item["test_type"]
        }
        for item in retrieved[:10]
    ]

    return {
        "reply": reply,
        "recommendations": recommendations,
        "end_of_conversation": False
    }
