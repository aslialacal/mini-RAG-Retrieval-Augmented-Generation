import os
import time
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from openai import OpenAI
import pandas as pd

app = FastAPI()

# Load wine dataset and embed it once on startup
df = pd.read_csv(r"C:\Users\aslia\azure-rag\wine-ratings.csv")

# Load SentenceTransformer model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Qdrant
qdrant = QdrantClient(host="localhost", port=6333)

collection_name = "wines"

# Create collection if not exists
if collection_name not in [col.name for col in qdrant.get_collections().collections]:
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

# Embed documents if collection is empty
if qdrant.count(collection_name).count == 0:
    vectors = embedder.encode(df["description"].tolist()).tolist()
    payload = df.to_dict(orient="records")

    points = [
        PointStruct(id=i, vector=vectors[i], payload=payload[i])
        for i in range(len(vectors))
    ]

    qdrant.upsert(collection_name=collection_name, points=points)
    print("Embeddings uploaded to Qdrant.")
else:
    print("Collection already populated.")

# OpenAI client with Llamafile
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="sk-no-key-required"
)

class Body(BaseModel):
    query: str

@app.get('/')
def root():
    return RedirectResponse(url='/docs', status_code=301)

@app.post('/ask')
def ask(body: Body):
    query = body.query
    embedded_query = embedder.encode(query).tolist()

    search_result = qdrant.search(
        collection_name=collection_name,
        query_vector=embedded_query,
        limit=3,
        with_payload=True
    )

    # Get top result context
    context = search_result[0].payload["description"]

    messages = [
        {"role": "system", "content": "Assistant is a sommelier helping users choose wines."},
        {"role": "user", "content": query},
        {"role": "assistant", "content": context}
    ]

    response = client.chat.completions.create(
        model="llava",  # you can leave this even if it's Phi-2, llamafile doesn't care
        messages=messages
    )

    return {"response": response.choices[0].message.content}
