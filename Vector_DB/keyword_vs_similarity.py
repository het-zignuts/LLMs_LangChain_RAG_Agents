import os
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def keyword_search(query, documents):
    results = []
    for doc in documents:
        if query.lower() in doc.lower():
            results.append(doc)
    return results

with open("sample_text.txt", "r") as file:
    text=file.read()

chunks=[chunk.strip() for chunk in text.split("\n\n")]

query="FastAPI"

keyword_results = keyword_search(query, chunks)
print("Keyword search results:")
for result in keyword_results:
    print(result)

model=SentenceTransformer("all-MiniLM-L6-v2")
vector_embeds=model.encode(chunks)

query_embedding=model.encode([query])
scores=cosine_similarity(query_embedding, vector_embeds)[0]
results=sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)

for text, score in results:
    print(round(score, 3), "→", text)