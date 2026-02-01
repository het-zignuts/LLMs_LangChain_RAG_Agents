import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

with open("sample_text.txt", "r") as file:
    text=file.read()

chunks=[chunk.strip() for chunk in text.split("\n\n")]

formatted_chunks=[]

for i, chunk in enumerate(chunks):
    formatted_chunks.append(Document(page_content=chunk, metadata={"id": i}))

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(formatted_chunks, embedding=embeddings, persist_directory="./chroma_db")
retriever=vector_db.as_retriever(k=2)

PROMPT = """
You are a question-answer assistant bot.
Use ONLY the context below to answer questions.

Context:
{context}

Question:
{question}

Answer:
"""

def format_context(docs):
    return "\n\n".join(
        f"[Chunk {d.metadata['id']}]\n{d.page_content}"
        for d in docs
    )

query="Explain about RAG"

results=retriever.invoke(query)
context = format_context(results)
prompt = PROMPT.format(context=context, question=query)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0 
)

response = llm.invoke(prompt)

print("===== ANSWER =====")
print(response.content)