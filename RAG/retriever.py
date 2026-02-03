import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

with open("sample_text.txt", "r") as file:
    text=file.read()

chunks=[chunk.strip() for chunk in text.split("\n\n")]

formatted_chunks=[]

for i, chunk in enumerate(chunks):
    formatted_chunks.append(Document(page_content=chunk, metadata={"id": i}))

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(formatted_chunks, embedding=embeddings, persist_directory="./chroma_db")
retriever=vector_db.as_retriever(k=2)
query="Explain about FastAPI"

results=retriever.invoke(query)

for doc in results:
    print("Chunk ID:", doc.metadata["id"])
    print(doc.page_content)