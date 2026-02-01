import os
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

with open("sample_text.txt", "r") as file:
    text=file.read()

chunks=[chunk.strip() for chunk in text.split("\n\n")]

model=SentenceTransformer("all-MiniLM-L6-v2")
vector_embeds=model.encode(chunks)

similarity_matrix = cosine_similarity(vector_embeds)
print(similarity_matrix)

tsne = TSNE(n_components=2, perplexity=3, random_state=42)
embedding_2d = tsne.fit_transform(np.array(vector_embeds))

plt.scatter(embedding_2d[:,0], embedding_2d[:,1])
for i, text in enumerate(chunks):
    plt.annotate(i, (embedding_2d[i,0], embedding_2d[i,1]))
plt.title("Semantic Clustering of RAG-related Chunks")
plt.show()