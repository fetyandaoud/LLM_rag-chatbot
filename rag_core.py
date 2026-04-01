import os
from typing import List, Tuple, Dict, Optional

import chromadb
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY saknas i .env-filen.")

COLLECTION_NAME = "papers"
CHROMA_PATH = "chroma_db"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
client = genai.Client(api_key=GEMINI_API_KEY)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


def get_or_create_collection():
    try:
        return chroma_client.get_collection(name=COLLECTION_NAME)
    except Exception:
        return chroma_client.create_collection(name=COLLECTION_NAME)


def reset_collection():
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    return chroma_client.create_collection(name=COLLECTION_NAME)


def chunk_text_smart(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """
    Enkel förbättrad chunking:
    - försöker dela på stycken först
    - håller ihop text bättre än ren fast slicing
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)

            # Om stycket självt är för långt, dela det
            if len(para) > chunk_size:
                start = 0
                while start < len(para):
                    end = start + chunk_size
                    piece = para[start:end].strip()
                    if piece:
                        chunks.append(piece)
                    if end >= len(para):
                        break
                    start += chunk_size - overlap
                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    # liten overlap mellan chunks
    if overlap > 0 and len(chunks) > 1:
        merged = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                merged.append(chunk)
            else:
                prev_tail = chunks[i - 1][-overlap:]
                merged.append((prev_tail + "\n\n" + chunk).strip())
        chunks = merged

    return chunks


def extract_pdf_pages(file_path: str, source_name: str) -> List[Dict]:
    reader = PdfReader(file_path)
    pages = []

    for page_index, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append(
                {
                    "text": text.strip(),
                    "source": source_name,
                    "page": page_index + 1,
                }
            )

    return pages


def index_pdf_file(file_path: str, source_name: Optional[str] = None) -> int:
    """
    Indexera en enda PDF till befintlig collection.
    """
    collection = get_or_create_collection()

    if source_name is None:
        source_name = os.path.basename(file_path)

    pages = extract_pdf_pages(file_path, source_name)

    texts = []
    metadatas = []
    ids = []

    # hitta start-id
    existing_count = collection.count()

    counter = existing_count
    for page in pages:
        chunks = chunk_text_smart(page["text"])
        for chunk_index, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append(
                {
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_index": chunk_index,
                }
            )
            ids.append(str(counter))
            counter += 1

    if not texts:
        return 0

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).tolist()

    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return len(texts)


def index_folder(folder_path: str, reset: bool = True) -> int:
    """
    Indexera alla PDF:er i en mapp.
    """
    if reset:
        collection = reset_collection()
    else:
        collection = get_or_create_collection()

    total_chunks = 0
    counter = 0

    for file_name in os.listdir(folder_path):
        if not file_name.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(folder_path, file_name)
        pages = extract_pdf_pages(file_path, file_name)

        texts = []
        metadatas = []
        ids = []

        for page in pages:
            chunks = chunk_text_smart(page["text"])
            for chunk_index, chunk in enumerate(chunks):
                texts.append(chunk)
                metadatas.append(
                    {
                        "source": page["source"],
                        "page": page["page"],
                        "chunk_index": chunk_index,
                    }
                )
                ids.append(str(counter))
                counter += 1

        if texts:
            embeddings = embedding_model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).tolist()

            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            total_chunks += len(texts)

    return total_chunks


def search_papers(question: str, n_results: int = 10, where: Dict = None) -> Dict:
    collection = get_or_create_collection()
    query_embedding = embedding_model.encode(question).tolist()

    args = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
    }
    if where:
        args["where"] = where

    return collection.query(**args)


def rerank_results(question: str, results: Dict, top_k: int = 5) -> Tuple[List[str], List[Dict]]:
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    if not documents:
        return [], []

    pairs = [(question, doc) for doc in documents]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(documents, metadatas, scores),
        key=lambda x: float(x[2]),
        reverse=True,
    )

    ranked = ranked[:top_k]
    return [x[0] for x in ranked], [x[1] for x in ranked]


def deduplicate_chunks(documents: List[str], metadatas: List[Dict]) -> Tuple[List[str], List[Dict]]:
    seen = set()
    final_docs = []
    final_metas = []

    for doc, meta in zip(documents, metadatas):
        key = (meta.get("source"), meta.get("page"), meta.get("chunk_index"))
        if key not in seen:
            seen.add(key)
            final_docs.append(doc)
            final_metas.append(meta)

    return final_docs, final_metas


def build_context(documents: List[str], metadatas: List[Dict]) -> Tuple[str, List[str]]:
    """
    Bygger kontext och snygga citations.
    """
    context_parts = []
    citations = []
    seen = set()

    for doc, meta in zip(documents, metadatas):
        source = meta["source"]
        page = meta["page"]
        chunk_index = meta.get("chunk_index", "?")

        context_parts.append(
            f"[Source: {source}, Page: {page}, Chunk: {chunk_index}]\n{doc.strip()}"
        )

        label = f"{source}, page {page}"
        if label not in seen:
            seen.add(label)
            citations.append(label)

    return "\n\n".join(context_parts), citations


def format_history(messages: List[Dict], max_turns: int = 6) -> str:
    if not messages:
        return "No previous conversation."

    recent = messages[-max_turns:]
    lines = []
    for item in recent:
        role = item.get("role", "user")
        content = item.get("content", "")
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)


def generate_answer(question: str, context: str, history_text: str) -> str:
    prompt = f"""
You answer ONLY using the provided context.

Rules:
- Use only the context below.
- If the context supports only a partial answer, provide the partial answer.
- Do not invent information.
- If the answer is truly missing, say exactly:
I could not find a clear answer in the papers.
- Keep the answer clear and factual.
- Add source citations naturally in the answer like:
  (2603.28651v1.pdf, page 2)

Previous conversation:
{history_text}

Context:
{context}

Question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return response.text.strip()


def ask_rag(question: str, history: List[Dict], where_filter: Dict = None) -> Dict:
    results = search_papers(question, n_results=10, where=where_filter)
    docs, metas = rerank_results(question, results, top_k=5)

    if not docs:
        return {
            "answer": "I could not find a clear answer in the papers.",
            "sources": [],
            "context": "",
        }

    docs, metas = deduplicate_chunks(docs, metas)
    context, sources = build_context(docs, metas)
    history_text = format_history(history)
    answer = generate_answer(question, context, history_text)

    return {
        "answer": answer,
        "sources": sources,
        "context": context,
    }