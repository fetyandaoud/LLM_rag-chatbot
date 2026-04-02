#  RAG Chatbot for Research Papers (LLM-powered)

This project is a Retrieval-Augmented Generation (RAG) system that allows users to ask questions about research papers and receive grounded answers with source citations.

##  Features

-  Upload and index PDF research papers
-  Semantic search using embeddings (SentenceTransformers)
-  Fast vector retrieval with ChromaDB
-  Re-ranking with Cross-Encoder for improved accuracy
-  LLM-powered answers using Google Gemini
-  Source-grounded responses with page references
-  Conversational memory for follow-up questions
-  Interactive UI with Streamlit

---

##  Architecture

User Question  
→ Embedding (bi-encoder)  
→ Vector Search (ChromaDB)  
→ Top-k Results  
→ Re-ranking (cross-encoder)  
→ Context Construction  
→ LLM (Gemini)  
→ Final Answer + Sources  

---

##  Tech Stack

- Python
- ChromaDB (vector database)
- SentenceTransformers (embeddings + reranker)
- Google Gemini (LLM)
- PyPDF (PDF parsing)
- Streamlit (frontend)

---

##  Installation

```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot
pip install -r requirements.txt

##  LLM Integration

The system uses a Large Language Model (Google Gemini) to generate answers based strictly on retrieved context, ensuring:

- Reduced hallucination
- Context-grounded reasoning
- Accurate citation of sources
