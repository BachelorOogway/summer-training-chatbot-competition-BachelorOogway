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
from langchain_text_splitters import RecursiveCharacterTextSplitter



# ====================== LOAD ENVIRONMENT ======================
load_dotenv()

API_Key = os.getenv("AZURE_OPENAI_API_KEY")

if not API_Key:
    raise RuntimeError("Missing Azure OpenAI credentials. Set AZURE_OPENAI_API_KEY in .env or environment.")


# ====================== CORE RAG FUNCTION ======================
def rag_answer(question: str) -> str:
    """
    Input: A single question (string)
    Output: A single answer (string)
    """

    BASE_DIR = Path.cwd().parent
    
    SEED_URLS = [
    "https://innowings.engg.hku.hk/",
    "https://innoacademy.engg.hku.hk/",   
    ]
    
    ALLOWED_DOMAINS = {"innowings.engg.hku.hk", "innoacademy.engg.hku.hk"}

    # Check if a URL belongs to Innowing
    def is_internal_link(url: str) -> bool:
        parsed = urlparse(url)
        return parsed.netloc in ALLOWED_DOMAINS or not parsed.netloc  # allow relative linksc
    
    HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/133.0 Safari/537.36"
    }
    MAX_PAGES = 20          # Safety limit - increase if needed
    DELAY = 1.0              # Seconds between requests (polite enough for this small site) 
    
            # Performs the most basic cleaning: removes only JavaScript and CSS, then extracts readable text while reducing extra blank lines.
    def simple_clean_text(soup: BeautifulSoup) -> str:
        # Very basic cleaning - remove script/style only
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Remove excessive blank lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    # Downloads a single page and returns a dictionary containing the URL and its cleaned text (or an error message)
    def scrape_page(url: str) -> Dict[str, str]:
            try:
                resp = requests.get(url, headers=HEADERS, timeout=15)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, "html.parser")
                text = simple_clean_text(soup)
                return {"url": url, "text": text}
            except Exception as e:
                print(f"❌ Failed {url}: {e}")
                return {"url": url, "text": f"[ERROR: {e}]"}
            
    print("🚀 Starting crawling...\n")

    visited: List[str] = []
    queue: List[str] = SEED_URLS[:]
    documents: List[Dict[str, str]] = []

    while queue and len(documents) < MAX_PAGES:
        url = queue.pop(0)
        if url in visited:
            continue

        print(f"📄 [{len(documents)+1}/{MAX_PAGES}] Scraping → {url}")
        doc = scrape_page(url)
        
        if doc["text"].strip():
            documents.append(doc)
        
        visited.append(url)

        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")
            for a in soup.find_all("a", href=True):
                full_url = urljoin(url, a["href"])
                if is_internal_link(full_url) and full_url not in visited:
                    queue.append(full_url)
        except:
            pass  # ignore link extraction errors

        time.sleep(DELAY)

    print(f"\n✅ Crawl finished! Collected {len(documents)} pages.")
    
    from dotenv import load_dotenv

    load_dotenv("../.env")
    DATASET = BASE_DIR / (os.getenv("DATASET") or "data.json")

    # Save to data.json
    with open(DATASET, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"💾 Saved to {DATASET}.")
        
    """embedding_generation"""
    # Load .env from parent folder
    load_dotenv(BASE_DIR / ".env")

    DATASET = BASE_DIR / (os.getenv("DATASET") or "data.json")  # Changed to data.json as requested
    EMBEDDING_MODEL = str(BASE_DIR / (os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"))
    CHROMA_PATH = BASE_DIR / (os.getenv("CHROMA_PATH") or "chroma_db/chroma_db")

    if os.path.exists(DATASET):
        print(f"📂 Loading documents from {DATASET}...")
        with open(DATASET, "r", encoding="utf-8") as f:
            documents: List[Dict] = json.load(f)
        print(f"   Loaded {len(documents)} raw pages.")
    else:
        raise FileNotFoundError(
            f"❌ {DATASET} not found. Please run your scraper first to generate data.json (or set OUTPUT_FILE env var)."
        )   

    
    print("\n🔪 Chunking documents using RecursiveCharacterTextSplitter (best-practice settings)...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        length_function=len,
        separators=[""],  # split only on character count
    )

    all_texts: List[str] = []
    all_metadatas: List[Dict] = []
    all_ids: List[str] = []

    for doc in documents:
        url = doc["url"]
        text = doc.get("text", "").strip()
        if not text:
            continue

        chunks = text_splitter.split_text(text)

        for chunk_idx, chunk in enumerate(chunks):
            # Deterministic ID (hash of URL + chunk index) → safe re-ingest / upsert
            id_str = f"{url}:{chunk_idx}"
            chunk_id = hashlib.md5(id_str.encode("utf-8")).hexdigest()

            all_texts.append(chunk)
            all_metadatas.append({
                "url": url,
                "source": "HKU InnoWings / InnoAcademy",
                "chunk_index": chunk_idx,
                "total_chunks_on_page": len(chunks),   # useful for debugging / future hierarchical RAG
            })
            all_ids.append(chunk_id)

    print(f"Created {len(all_texts)} chunks from {len(documents)} documents.")
    
    API_Key = os.getenv("AZURE_OPENAI_API_KEY")
    if not API_Key:
        raise RuntimeError("Missing Azure OpenAI credentials.")

    client = AzureOpenAI(
        azure_endpoint="https://api-iw.azure-api.net/sig-embedding/openai/deployments/text-embedding-3-small/embeddings?api-version=2024-10-21",
        api_key=API_Key,
        api_version="2024-10-21",
    )

    def get_embedding(text: str) -> List[float]:
        response = client.embeddings.create(
            input=[text.replace("\n", " ")],
            model=EMBEDDING_MODEL,
        )
        return response.data[0].embedding

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    COLLECTION_NAME = "Innowing_db"

    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
        print(f"🗑️  Deleted existing collection '{COLLECTION_NAME}' for fresh ingest.")
    except:
        pass

    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    print("🧬 Generating embeddings and storing in ChromaDB (this may take a while)...")

    BATCH_SIZE = 50

    for i in range(0, len(all_texts), BATCH_SIZE):
        batch_texts = all_texts[i : i + BATCH_SIZE]
        batch_metadatas = all_metadatas[i : i + BATCH_SIZE]
        batch_ids = all_ids[i : i + BATCH_SIZE]

        # Generate embeddings for the batch
        embeddings = [get_embedding(text) for text in batch_texts]

        # Use .upsert() for idempotent ingestion
        collection.upsert(
            embeddings=embeddings,
            documents=batch_texts,
            ids=batch_ids
        )

        print(f"  [{i + len(batch_texts)}/{len(all_texts)}] Added batch to ChromaDB")

    print("\n🎉 Ingestion complete!")
    print(f"   ChromaDB collection '{COLLECTION_NAME}' saved to: {CHROMA_PATH}")
    print("   ✅ Ready for RAG! Use the same embedding model for queries.")
    
    load_dotenv(BASE_DIR / ".env")

    API_Key = os.getenv("AZURE_OPENAI_API_KEY")

    if not API_Key:
        raise RuntimeError("Missing Azure OpenAI credentials. Set AZURE_OPENAI_API_KEY in .env or environment.")

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

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    context_docs = []
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

generate_rag_answers(["What are the current SIGs in InnoWings?"])