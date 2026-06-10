import os
import json
import string
import numpy as np
from pathlib import Path
from typing import List, Dict
from functools import lru_cache
from dotenv import load_dotenv
import chromadb
from rank_bm25 import BM25Okapi
from openai import AzureOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
BASE_DIR = Path(__file__).resolve().parent

# ====================== LOAD ENVIRONMENT ======================
load_dotenv(BASE_DIR.parent / ".env")

API_Key = os.getenv("AZURE_OPENAI_API_KEY")

if not API_Key:
    raise RuntimeError("Missing Azure OpenAI credentials. Set AZURE_OPENAI_API_KEY in .env or environment.")

# ====================== HELPER FUNCTIONS ======================

def simple_tokenize(text: str) -> List[str]:
    """Simple tokenizer - splits on spaces and removes punctuation"""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()

@lru_cache(maxsize=1)
def build_bm25_index_from_documents(docs_dir: str):
    """
    Reads raw JSONs from documents/, chunks them EXACTLY like the ingestion script,
    and builds a BM25 index. Cached so it only runs once per application lifecycle.
    """
    docs_path = Path(docs_dir)
    all_chunks = []  # Will hold {"text": str, "metadata": dict}

    # 1. Process Web Documents (data.json)
    data_json_path = docs_path / "data.json"
    if data_json_path.exists():
        with open(data_json_path, 'r', encoding='utf-8') as f:
            web_docs = json.load(f)
        
        # MUST match ingestion script parameters exactly
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000, chunk_overlap=200, length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        for doc in web_docs:
            url = doc.get("url", "").strip()
            text = doc.get("text", "").strip()
            if not text: continue
            
            chunks = splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": chunk,
                    "metadata": {"content_type": "web", "url": url, "chunk_index": i, "total_chunks": len(chunks)}
                })

    # 2. Process Image Annotations (image_annotations_*.json)
    for img_file in docs_path.glob("image_annotations_*.json"):
        with open(img_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
            
        for image_name, entry in annotations.items():
            if not isinstance(entry, dict): continue
            image_path = str(entry.get("path", image_name))
            
            # Match ingestion logic: use existing 'chunks' if available
            if "chunks" in entry and entry["chunks"]:
                chunks = [c.strip() for c in entry["chunks"] if c and c.strip()]
            else:
                # Fallback if no chunks exist (mimicking _extract_image_text)
                annotation = entry.get("annotation", "")
                full_text = str(annotation[0]) if isinstance(annotation, list) and annotation else str(annotation)
                if not full_text.strip(): continue
                chunks = splitter.split_text(full_text)

            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": chunk,
                    "metadata": {"content_type": "image", "image_name": image_name, "image_path": image_path, "chunk_index": i, "total_chunks": len(chunks)}
                })

    # Build BM25 Index
    tokenized_corpus = [simple_tokenize(doc["text"]) for doc in all_chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    
    print(f"BM25 Index built successfully with {len(all_chunks)} chunks.")
    return bm25, all_chunks

def combine_scores(results: List[Dict]) -> List[Dict]:
    """Average scores for duplicate documents"""
    doc_scores = {}
    doc_metadata = {}
    
    for r in results:
        doc_text = r["text"]
        if doc_text not in doc_scores:
            doc_scores[doc_text] = []
            doc_metadata[doc_text] = r["metadata"]
        doc_scores[doc_text].append(r["score"])
    
    combined = []
    for doc_text, scores in doc_scores.items():
        combined.append({
            "text": doc_text,
            "score": sum(scores) / len(scores),
            "metadata": doc_metadata[doc_text]
        })
    
    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined

def reciprocal_rank_fusion(vector_results, bm25_results, top_k=5, k=60):
    """Combine rankings using RRF"""
    scores = {}
    
    for rank, item in enumerate(vector_results[:top_k], 1):
        scores[item["text"]] = scores.get(item["text"], 0) + 1 / (k + rank)
    
    for rank, item in enumerate(bm25_results[:top_k], 1):
        scores[item["text"]] = scores.get(item["text"], 0) + 1 / (k + rank)
    
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    final_results = []
    
    for doc_text, score in sorted_items[:top_k]:
        metadata = {}
        for item in vector_results + bm25_results:
            if item["text"] == doc_text:
                metadata = item["metadata"]
                break
        final_results.append({"text": doc_text, "score": score, "metadata": metadata})
    
    return final_results

# ====================== MAIN RAG FUNCTION ======================

def rag_answer3(question: str) -> str:
    """
    Input: A single question (string)
    Output: A single answer (string)
    """
    # 1. Handle empty question immediately
    if not question.strip():
        return "Please provide a valid question."

    # 2. Load Environment & Clients
    load_dotenv(BASE_DIR / ".env")
    API_Key = os.getenv("AZURE_OPENAI_API_KEY")

    embedding_client = AzureOpenAI(
        azure_endpoint="https://api-iw.azure-api.net/sig-embedding/openai/deployments/text-embedding-3-small/embeddings?api-version=2024-10-21",
        api_key=API_Key, api_version="2024-10-21",
    )

    GPT_client = AzureOpenAI(
        azure_endpoint="https://api-iw.azure-api.net/sig-shared-jpeast/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview",
        api_key=API_Key, api_version="2025-01-01-preview",
    )
    
    # 3. Setup ChromaDB
    CHROMA_PATH = str(BASE_DIR / (os.getenv("CHROMA_PATH") or "chroma_db/chroma_db"))
    COLLECTION_NAME = "Innowing_db"
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    
    # 4. Setup BM25 (Cached from documents/ folder)
    DOCS_PATH = str(BASE_DIR / "documents")
    bm25, all_bm25_docs = build_bm25_index_from_documents(DOCS_PATH)
    
    top_k = 5

    # 5. Generate Question Variations
    PROMPT_STR1 = "You are an AI language model assistant. Your task is to generate 3 to 5 different versions of the given user question to help retrieve relevant documents from a database.Rules:Generate 3-5 variations using synonyms or different phrasing, focused on the core intent.Output as a plain, newline-separated list."
    
    messages = [
        {"role": "system", "content": PROMPT_STR1},
        {"role": "user", "content": f"Question: {question}"}
    ]

    variations_response = GPT_client.chat.completions.create(
        model="gpt-4o-mini", messages=messages
    ).choices[0].message.content

    variations = [q.strip() for q in variations_response.split('\n') if q.strip()]
    variations = [question] + variations  # Include original question

    # 6. Execute Search for ALL variations
    all_vector_results = []
    all_bm25_results = []

    for variation in variations:
        # --- A: Vector Search (Semantic) ---
        response = embedding_client.embeddings.create(
            input=variation.replace("\n", " "),
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        )
        query_embedding = response.data[0].embedding
        
        results = collection.query(
            query_embeddings=[query_embedding], n_results=top_k * 2,
            include=["documents", "metadatas", "distances"]
        )
        
        for doc, meta, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            all_vector_results.append({
                "text": doc, "score": 1 - distance, "metadata": meta
            })
        
        # --- B: BM25 Search (Keyword) ---
        tokenized_query = simple_tokenize(variation)
        scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k * 2]
        
        for idx in top_indices:
            if scores[idx] > 0:
                all_bm25_results.append({
                    "text": all_bm25_docs[idx]["text"],
                    "score": float(scores[idx]),
                    "metadata": all_bm25_docs[idx]["metadata"]
                })

    # 7. Combine, Deduplicate, and Fuse (RRF)
    # We do this ONCE outside the loop using all accumulated results
    combined_vector = combine_scores(all_vector_results)
    print(f"Combined Vector Search results: {len(combined_vector)} unique documents")
    combined_bm25 = combine_scores(all_bm25_results)
    print(f"Combined BM25 Search results: {len(combined_bm25)} unique documents")
    final_results = reciprocal_rank_fusion(combined_vector, combined_bm25, top_k=top_k)
    
    for i, doc in enumerate(final_results, 1):
        print(f"Final Doc {i}: Score={doc['score']:.4f}, URL={doc['metadata'].get('url', doc['metadata'].get('image_path', 'N/A'))}, Text Preview={doc['text'][:100]}{'...' if len(doc['text']) > 100 else ''}")
    context_docs = [
        {
            "text": doc["text"],
            "url": doc["metadata"].get("url", doc["metadata"].get("image_path", "")),
            "score": doc["score"]
        }
        for doc in final_results
    ]  
    print(f"Final RRF results: {len(final_results)} unique documents")


    
    # 8. Generate Final Answer
    return format_answer(context_docs, question, GPT_client)    
         

def format_answer(context_docs: List[Dict], question: str, gpt_client) -> str:
    """Format context and generate answer using GPT"""
    if not context_docs:
        return "I don't have enough information to answer this question."

    context_parts = []
    for i, doc in enumerate(context_docs, 1):
        context_parts.append(f"[Document {i}] {doc['text']}")
        if doc.get('url'):
            context_parts.append(f"Source: {doc['url']}")
        context_parts.append("")
    
    context_str = "\n".join(context_parts)
    
    PROMPT_STR2 = f"""You are a helpful AI assistant. Answer the following question based ONLY on the provided context.

If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question."

Context:
{context_str}

Question: {question}

Answer concisely and accurately, citing specific parts of the context when relevant."""

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant that answers questions based on provided context."},
        {"role": "user", "content": PROMPT_STR2}
    ]
    
    response = gpt_client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, temperature=0.3
    )
    
    return response.choices[0].message.content

def generate_rag_answers(questions: List[str]) -> List[str]:
    """
    Input: List of questions (strings)
    Output: zip of (question, answer) pairs
    
    Example usage:
        from rag import generate_rag_answers
        answers = generate_rag_answers([
            "What are the current SIGs in InnoWings?",
            "Tell me about recent Tech Talks in InnoAcademy."
        ])
        print(answers)
    """
    answers = []
    for question in questions:
        print(f"🤖 Answering: {question[:80]}{'...' if len(question) > 80 else ''}")
        answer = rag_answer3(question)
        answers.append(answer)
    return zip(questions, answers)

print(list(generate_rag_answers(["On the wall on the back of the Brainstorming Area,  there are 3 words written in big letters which are HKU, Engineering, and what..?"])))

