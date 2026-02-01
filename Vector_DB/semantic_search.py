import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

load_dotenv()

with open("sample_text.txt", "r", encoding="utf-8") as file:
    text = file.read()

chunks=[chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]

documents=[Document(page_content=chunk, metadata={"chunk_id": idx}) for idx, chunk in enumerate(chunks)]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

retriever=vector_db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 3,
        "score_threshold": 0.5
    }
)


PROMPT="""
You are a semantic search assistant.

Rules:
- Answer ONLY using the provided context.
- If the answer is not found in the context, respond exactly:
  "No matching information identified. Please upload relevant documents."

Context:
{context}

Question:
{question}

Answer:
"""

def format_context(docs):
    return "\n\n".join(
        f"[Chunk {doc.metadata['chunk_id']}]\n{doc.page_content}"
        for doc in docs
    )

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)

def ask_llm(question: str):
    docs=retriever.invoke(question)
    if not docs:
        return {
            "answer": "No matching information identified. Please upload relevant documents.",
            "sources": [],
            "confidence": 0.0
        }

    context=format_context(docs)
    prompt=PROMPT.format(context=context, question=question)
    response=llm.invoke(prompt)
    confidence=round(1/len(docs), 2)  
    return {
        "answer": response.content,
        "sources": [doc.metadata["chunk_id"] for doc in docs],
        "confidence": confidence
    }

def main():
    query = "Explain what RAG is"
    result = ask_llm(query)
    print("\nAnswer: ")
    print(result["answer"])

    print("\nSources: ")
    print(result["sources"])

    print("\nConfidence")
    print(result["confidence"])

main()