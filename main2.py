from dotenv import load_dotenv
import os
import chromadb
from openai import AzureOpenAI
from opentelemetry import context
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json
import time
from typing import List, Dict
from pathlib import Path
import hashlib
import asyncio
import aiohttp
import numpy as np
from rank_bm25 import BM25Okapi
from functools import lru_cache


BASE_DIR = Path(__file__).resolve().parent

# ====================== LOAD ENVIRONMENT ======================
load_dotenv(BASE_DIR.parent / ".env")

API_Key = os.getenv("AZURE_OPENAI_API_KEY")

if not API_Key:
    raise RuntimeError("Missing Azure OpenAI credentials. Set AZURE_OPENAI_API_KEY in .env or environment.")

# ====================== HELPER RAG FUNCTION ======================
 
# ====================== CORE RAG FUNCTION ======================
def rag_answer(question: str) -> str:
    """
    Input: A single question (string)
    Output: A single answer (string)
    """
    
    # ====================== LOAD ENVIRONMENT ======================
    load_dotenv(BASE_DIR / ".env")

    API_Key = os.getenv("AZURE_OPENAI_API_KEY")

    embedding_client = AzureOpenAI(
        azure_endpoint="https://api-iw.azure-api.net/sig-embedding/openai/deployments/text-embedding-3-small/embeddings?api-version=2024-10-21",
        api_key=API_Key,
        api_version="2024-10-21",
    )

    GPT_client = AzureOpenAI(
        azure_endpoint="https://api-iw.azure-api.net/sig-shared-jpeast/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview",
        api_key=API_Key,
        api_version="2025-01-01-preview",
    )
    
    CHROMA_PATH = str(BASE_DIR / (os.getenv("CHROMA_PATH") or "chroma_db/chroma_db"))
    COLLECTION_NAME = "Innowing_db"

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    
    #question = "What is the Innovation Academy?"
    
    top_k = 5

    if not question.strip():
        results = collection.peek(limit=top_k)
        context_docs = [{"text": doc, "url": meta.get("url", ""), "score": 1.0}
                for doc, meta in zip(results["documents"], results["metadatas"])]

    response = embedding_client.embeddings.create(
        input=question.replace("\n", " "),
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    )
    query_embedding = response.data[0].embedding


    ## STRATEGY 1 : MULTI QUERYING
    ## STRATEGY 2 : QUERY REWRITING

    # # generate 3-5 variation of question using LLM
    PROMPT_STR1 = f"You are an AI language model assistant. Your task is to generate 3 to 5 different versions of the given user question to help retrieve relevant documents from a database.Rules:Generate 3-5 variations using synonyms or different phrasing, focused on the core intent.Output as a plain, newline-separated list."
    PROMPT_STR2 = f"You are a search engine. In order to obtain information for answering the query, please provide at least three rewritten queries. Do not answer the rewritten queries. Don't output any words other than the rewritten queries. Output as a plain, newline-separated list."
    
    messages = [
        {"role": "system", "content": PROMPT_STR1},
        {"role": "user", "content": f"Question: {question}"}
    ]

    variations_response = GPT_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    ).choices[0].message.content

    print("Generated Query Variations:\n", variations_response)

    def parallel_search(variations: List[str]) -> List[Dict]:
        context_docs = []
        for variation in variations:
            response = embedding_client.embeddings.create(
                input=variation.replace("\n", " "),
                model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            )
            query_embedding = response.data[0].embedding
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            for doc, meta, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                score = 1 - distance
                context_docs.append({
                    "text": doc,
                    "score": round(score, 4)
                })
        return context_docs
    
    context_docs = parallel_search(variations_response.split("\n"))
    
    # reduce duplication and sort by score
    unique_docs = {}
    for doc in context_docs:
        if doc["text"] not in unique_docs or doc["score"] > unique_docs[doc["text"]]["score"]:
            unique_docs[doc["text"]] = doc

    context_docs = list(unique_docs.values())
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
            context_parts.append(
                f"Document {i} :\n{doc['text']}"
            )
    
    context = "\n\n---\n\n".join(context_parts)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant for HKU InnoWings and InnoAcademy. "
                "Answer the question using ONLY the provided context documents. "
                "If the answer cannot be found in the context, say: "
                "'I don't have enough information based on the provided documents.' "
                "Be concise, accurate, and professional. Always cite the source URL when possible."
            )
        },
        {
            "role": "user",
            "content": f"Context documents:\n{context}\n\nQuestion: {question}"
        }
    ]

    # answer = GPT_client.chat.completions.create(
    #         model="gpt-4o-mini",
    #         messages=messages
    #     ).choices[0].message.content
    
    print("here is the answer: ", answer)
    return answer

def rag_answer2(question: str) -> str:
    """
    Input: A single question (string)
    Output: A single answer (string)
    """
    
    # ====================== LOAD ENVIRONMENT ======================
    load_dotenv(BASE_DIR / ".env")

    API_Key = os.getenv("AZURE_OPENAI_API_KEY")

    embedding_client = AzureOpenAI(
        azure_endpoint="https://api-iw.azure-api.net/sig-embedding/openai/deployments/text-embedding-3-small/embeddings?api-version=2024-10-21",
        api_key=API_Key,
        api_version="2024-10-21",
    )

    GPT_client = AzureOpenAI(
        azure_endpoint="https://api-iw.azure-api.net/sig-shared-jpeast/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview",
        api_key=API_Key,
        api_version="2025-01-01-preview",
    )
    
    CHROMA_PATH = str(BASE_DIR / (os.getenv("CHROMA_PATH") or "chroma_db/chroma_db"))
    COLLECTION_NAME = "Innowing_db"

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
        
    top_k = 5

    if not question.strip():
        results = collection.peek(limit=top_k)
        context_docs = [{"text": doc, "url": meta.get("url", ""), "score": 1.0}
                for doc, meta in zip(results["documents"], results["metadatas"])]

    response = embedding_client.embeddings.create(
        input=question.replace("\n", " "),
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    )
    query_embedding = response.data[0].embedding

    # STRATEGY 4 : SINGLE QUERY
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    for doc, meta, distance in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        # Convert distance to similarity score (Chroma uses cosine distance by default)
        score = 1 - distance
        context_docs.append({
            "text": doc,
            "score": round(score, 4)
        })

    # print("Retrieved Context Documents:")
    # for idx, doc in enumerate(context_docs, 1):
    #     print(f"{idx}. {doc['text'][:100]}... (Score: {doc['score']})")
    
    context_parts = []
    for i, doc in enumerate(context_docs, 1):
            context_parts.append(
                f"Document {i} :\n{doc['text']}"
            )

    context = "\n\n---\n\n".join(context_parts)
    
    # print("\nConstructed Context for LLM:")
    # print(context)
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant for HKU InnoWings and InnoAcademy. "
                "Answer the question using ONLY the provided context documents. "
                "If the answer cannot be found in the context, say: "
                "'I don't have enough information based on the provided documents.' "
                "Be concise, accurate, and professional. Always cite the source URL when possible."
            )
        },
        {
            "role": "user",
            "content": f"Context documents:\n{context}\n\nQuestion: {question}"
        }
    ]

    answer = GPT_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        ).choices[0].message.content
    
    print("here is the answer: ", answer)
    return answer


def simple_tokenize(text: str) -> List[str]:
    """Simple tokenizer - splits on spaces and removes punctuation"""
    import string
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.split()

@lru_cache(maxsize=1)
def rag_answer3(question: str) -> str:
    """
    Input: A single question (string)
    Output: A single answer (string)
    """
    
    # ====================== LOAD ENVIRONMENT ======================
    load_dotenv(BASE_DIR / ".env")

    API_Key = os.getenv("AZURE_OPENAI_API_KEY")

    embedding_client = AzureOpenAI(
        azure_endpoint="https://api-iw.azure-api.net/sig-embedding/openai/deployments/text-embedding-3-small/embeddings?api-version=2024-10-21",
        api_key=API_Key,
        api_version="2024-10-21",
    )

    GPT_client = AzureOpenAI(
        azure_endpoint="https://api-iw.azure-api.net/sig-shared-jpeast/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview",
        api_key=API_Key,
        api_version="2025-01-01-preview",
    )
    
    CHROMA_PATH = str(BASE_DIR / (os.getenv("CHROMA_PATH") or "chroma_db/chroma_db"))
    COLLECTION_NAME = "Innowing_db"

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)
    
    top_k = 5

    if not question.strip():
        results = collection.peek(limit=top_k)
        context_docs = [{"text": doc, "url": meta.get("url", ""), "score": 1.0}
                for doc, meta in zip(results["documents"], results["metadatas"])]

    all_docs_data = collection.get(include=["documents", "metadatas"])
    all_documents = all_docs_data["documents"]
    all_metadatas = all_docs_data["metadatas"]
    
    # Build BM25 index
    tokenized_docs = [simple_tokenize(doc) for doc in all_documents]
    bm25 = BM25Okapi(tokenized_docs)


    PROMPT_STR1 = f"You are an AI language model assistant. Your task is to generate 3 to 5 different versions of the given user question to help retrieve relevant documents from a database.Rules:Generate 3-5 variations using synonyms or different phrasing, focused on the core intent.Output as a plain, newline-separated list."

    messages = [
        {"role": "system", "content": PROMPT_STR1},
        {"role": "user", "content": f"Question: {question}"}
    ]

    variations_response = GPT_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    ).choices[0].message.content

    variations = [q.strip() for q in variations_response.split('\n') if q.strip()]
    variations = [question] + variations

    vector_results = []
    bm25_results = []

    for variation in variations:
        # 3a: Vector search (semantic)
        response = embedding_client.embeddings.create(
            input=variation.replace("\n", " "),
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        )
        query_embedding = response.data[0].embedding
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,  # Retrieve more for better fusion
            include=["documents", "metadatas", "distances"]
        )
        
        for doc, meta, distance in zip(results["documents"][0], 
                                       results["metadatas"][0], 
                                       results["distances"][0]):
            vector_results.append({
                "text": doc,
                "score": 1 - distance,  # Convert distance to similarity
                "metadata": meta
            })
        
        tokenized_query = simple_tokenize(variation)
        scores = bm25.get_scores(tokenized_query)
        
        # Get top-k BM25 results
        top_indices = np.argsort(scores)[::-1][:top_k * 2]
        for idx in top_indices:
            if scores[idx] > 0:  # Only include if there's a match
                bm25_results.append({
                    "text": all_documents[idx],
                    "score": scores[idx],
                    "metadata": all_metadatas[idx]
                })
        
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
            
            # Average the scores
            combined = []
            for doc_text, scores in doc_scores.items():
                combined.append({
                    "text": doc_text,
                    "score": sum(scores) / len(scores),
                    "metadata": doc_metadata[doc_text]
                })
            
            combined.sort(key=lambda x: x["score"], reverse=True)
            return combined

        all_vector_results = combine_scores(vector_results)
        all_bm25_results = combine_scores(bm25_results)

        def reciprocal_rank_fusion(vector_results, bm25_results, k=60):
            """Combine rankings using RRF"""
            scores = {}
            
            # Score vector results
            for rank, item in enumerate(vector_results[:top_k], 1):
                scores[item["text"]] = scores.get(item["text"], 0) + 1 / (k + rank)
            
            # Score BM25 results
            for rank, item in enumerate(bm25_results[:top_k], 1):
                scores[item["text"]] = scores.get(item["text"], 0) + 1 / (k + rank)
            
            # Sort and return
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            final_results = []
            for doc_text, score in sorted_items[:top_k]:
                # Find metadata (from either result set)
                metadata = {}
                for item in vector_results + bm25_results:
                    if item["text"] == doc_text:
                        metadata = item["metadata"]
                        break
                final_results.append({
                    "text": doc_text,
                    "score": score,
                    "metadata": metadata
                })
            
            return final_results
        
        # Apply fusion
        final_results = reciprocal_rank_fusion(all_vector_results, all_bm25_results)
        context_docs = [
                {
                    "text": doc["text"],
                    "url": doc["metadata"].get("url", ""),
                    "score": doc["score"]
                }
                for doc in final_results
        ]  
        return format_answer(context_docs, question, GPT_client)    
         
def format_answer(context_docs: List[Dict], question: str, gpt_client) -> str:
    """Format context and generate answer using GPT"""
    
    # Build context string
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
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3
    )
    
    return response.choices[0].message.content
    
# ====================== PUBLIC API FUNCTION ======================
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
        answer = rag_answer(question)
        answers.append(answer)
    return zip(questions, answers)

generate_rag_answers(["What are the public spaces in InnoWings?"])