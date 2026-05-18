from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
    # SMART CLARIFICATION LOGIC
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
    # RETRIEVE ASSESSMENTS
    # --------------------------------------------

    retrieved = retrieve_assessments(
        conversation_text
    )

    # --------------------------------------------
    # GENERATE REPLY
    # --------------------------------------------

    intro = (
        "Based on the conversation and hiring requirements, "
        "these SHL assessments are recommended:\n\n"
    )

    recommendation_text = ""

    for assessment in retrieved[:5]:

        recommendation_text += (

            f"- {assessment['name']}\n"

            f"  Type: {assessment['test_type']}\n"

            f"  {assessment['description']}\n"

            f"  URL: {assessment['url']}\n\n"
        )

    reply = intro + recommendation_text

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
