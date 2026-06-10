from dotenv import load_dotenv
import os
import math
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

import chromadb
import numpy as np
from openai import AzureOpenAI
from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).resolve().parent

# ──────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT SETUP
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing Azure OpenAI credentials. Set AZURE_OPENAI_API_KEY in .env.")


# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTION 1: QUESTION EXPANSION
#
# A single question may miss relevant chunks if the wording doesn't match how
# the source text was written.  We ask GPT to rephrase the question in several
# ways so that retrieval casts a wider semantic net.
#
# Example:
#   Original : "What SIGs does InnoWings have?"
#   Variation: "Which special interest groups are part of InnoWings?"
#   Variation: "List the current InnoWings SIG programmes."
# ──────────────────────────────────────────────────────────────────────────────

def _generate_question_variations(original: str, num_variations: int = 4) -> List[str]:
    """
    Calls GPT-4o-mini to produce num_variations alternative phrasings of
    original.  The original is always prepended so it is never dropped.

    Why this matters: embedding models encode *meaning*, not keywords.  Two
    phrasings that mean the same thing may still produce slightly different
    vectors.  Querying with multiple phrasings increases the probability that
    every relevant chunk is retrieved by at least one query.
    """
    GPT_client = AzureOpenAI(
        azure_endpoint="https://api-iw.azure-api.net/sig-shared-jpeast/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview",
        api_key=API_KEY,
        api_version="2025-01-01-preview",
    )

    response = GPT_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        messages=[
            {
                "role": "system",
                "content": (
                    f"Generate {num_variations} different phrasings of the given question. "
                    "Return one per line, no numbering, no extra text."
                ),
            },
            {"role": "user", "content": f"Original: {original}"},
        ],
    )

    lines = [
        line.strip()
        for line in response.choices[0].message.content.strip().split("\n")
        if line.strip()
    ]
    # Prepend the original so it always participates in retrieval
    return [original] + lines[:num_variations]


# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTION 2: SIBLING CHUNK RESOLUTION
#
# Root cause of the "plaque unveiling" failure:
#   An image description was split into 2 chunks at indexing time.
#   Chunk 0 describes people, poses, and visual atmosphere — its embedding
#   vector is dominated by those semantics and scores well for queries about
#   events and ceremonies only by coincidence.
#   Chunk 1 contains the actual event grid with names and dates — but because
#   chunk 0 is retrieved first, chunk 1 never enters the candidate pool.
#
# Fix: after retrieval, inspect every retrieved chunk's metadata.  If it came
# from a multi-chunk image (total_chunks > 1), fetch ALL sibling chunks from
# ChromaDB by matching on image_name.  Siblings are injected into chunk_scores
# with the same score as the chunk that triggered their discovery, so the
# ranking algorithms treat them as equally relevant candidates.
#
# This is correct behaviour: if any part of an image description is relevant
# to the query, all parts of that description should be considered, because
# the facts (dates, event names) and the context (what the image shows) are
# inseparable.
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_sibling_chunks(
    chunk_scores: Dict[str, float],
    chunk_hits:   Dict[str, int],
    chunk_meta:   Dict[str, dict],
    collection,
) -> None:
    """
    Mutates chunk_scores, chunk_hits, and chunk_meta in-place.

    For every retrieved chunk that is part of a multi-chunk image
    (metadata field total_chunks > 1), fetches all sibling chunks from
    ChromaDB and adds them to the candidate pool if not already present.

    Siblings receive the same cumulative score as the chunk that triggered
    their discovery — they are treated as equally plausible candidates and
    left to the ranking algorithms to order.
    """
    # Collect image names that need sibling resolution.
    # We only act on content_type == "image" chunks that have siblings.
    images_to_resolve = {}
    for chunk, meta in chunk_meta.items():
        if (
            meta.get("content_type") == "image"
            and meta.get("total_chunks", 1) > 1
        ):
            image_name = meta.get("image_name")
            if image_name and image_name not in images_to_resolve:
                # Record the highest score seen for any chunk of this image,
                # so siblings inherit a meaningful relevance signal.
                images_to_resolve[image_name] = chunk_scores[chunk]

    if not images_to_resolve:
        return

    # Fetch every document in the collection along with its metadata.
    # ChromaDB has no server-side WHERE filter on arbitrary metadata fields,
    # so we fetch all and filter in Python.  For typical corpus sizes
    # (hundreds to low thousands of chunks) this is fast enough.
    all_docs = collection.get(include=["documents", "metadatas"])

    for doc, meta in zip(all_docs["documents"], all_docs["metadatas"]):
        image_name = (meta or {}).get("image_name")
        if image_name not in images_to_resolve:
            continue
        if doc in chunk_scores:
            continue  # already in the candidate pool

        # Inject sibling with the score of the chunk that was already retrieved.
        # This makes it a genuine competitor without artificially inflating it
        # above the chunk that earned its place through embedding similarity.
        inherited_score = images_to_resolve[image_name]
        chunk_scores[doc] = inherited_score
        chunk_hits[doc]   = 1
        chunk_meta[doc]   = meta or {}


# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTION 3: MULTI-QUERY RETRIEVAL
#
# Each question variation is embedded and used to query ChromaDB independently.
# We accumulate two things for every retrieved chunk:
#
#   chunk_scores: total cosine similarity across all queries that found it.
#                 Dividing by num_queries later gives the *average* similarity,
#                 which is the semantic relevance score used by Algorithms 1 & 2.
#
#   chunk_hits: how many distinct queries retrieved this chunk at all.
#               Used by Algorithm 1 to reward *consistency* of retrieval.
#
# We also collect metadata (source URL, image annotations) for the web+image
# database so downstream algorithms can use it if needed.
# ──────────────────────────────────────────────────────────────────────────────

def _multi_query_retrieve(
    questions: List[str],
    collection,
    embedding_client,
    top_k: int = 15,
) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, dict]]:
    """
    Returns
    -------
    chunk_scores : {chunk_text: cumulative_cosine_similarity}
    chunk_hits   : {chunk_text: number_of_queries_that_retrieved_it}
    chunk_meta   : {chunk_text: metadata_dict}  — source URL, alt-text, etc.
    """
    chunk_scores: Dict[str, float] = defaultdict(float)
    chunk_hits:   Dict[str, int]   = defaultdict(int)
    chunk_meta:   Dict[str, dict]  = {}

    for question in questions:
        # Embed this question variation
        embedding_response = embedding_client.embeddings.create(
            input=question.replace("\n", " "),
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        )
        query_vector = embedding_response.data[0].embedding

        # Query ChromaDB; request metadatas so we can surface URLs / image info
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "distances", "metadatas"],
        )

        for doc, dist, meta in zip(
            results["documents"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            # ChromaDB returns cosine *distance* (0 = identical, 2 = opposite).
            # Convert to similarity in [0, 1] so higher = more relevant.
            similarity = 1 - dist

            chunk_scores[doc] += similarity
            chunk_hits[doc]   += 1   # count this query as a "vote" for the chunk

            # Store metadata once per unique chunk (overwrite is fine; it's stable)
            if doc not in chunk_meta:
                chunk_meta[doc] = meta or {}

    return chunk_scores, chunk_hits, chunk_meta


# ──────────────────────────────────────────────────────────────────────────────
# ALGORITHM 1 : WEIGHTED SCORE COMBINATION
#
# Goal: rank chunks by *how confidently* the retrieval system found them.
#
# Two independent signals:
#
#   avg_similarity (weight 0.7)
#       Average cosine similarity across ALL queries.
#       A chunk scoring 0.9 on every query is much more reliable than one
#       scoring 0.9 on a single query and 0.0 on the rest.
#
#   coverage (weight 0.3)
#       Fraction of queries that retrieved this chunk at all (hits / total).
#       A chunk found by 5/5 queries is robust; one found by 1/5 is fragile.
# ──────────────────────────────────────────────────────────────────────────────

def _algorithm_1_weighted(
    chunk_scores: Dict[str, float],
    chunk_hits:   Dict[str, int],
    num_queries:  int,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Ranks chunks by a weighted combination of average similarity and query
    coverage.  Both signals are in [0, 1], so the final score is too.
    """
    ranked = []
    for chunk, total_similarity in chunk_scores.items():
        # Average similarity: how similar was this chunk to the typical query?
        avg_similarity = total_similarity / num_queries

        # Coverage: what fraction of query variations retrieved this chunk?
        # High coverage means the chunk is relevant regardless of phrasing —
        # a strong signal of genuine relevance rather than lucky keyword overlap.
        coverage = chunk_hits[chunk] / num_queries

        final_score = 0.7 * avg_similarity + 0.3 * coverage
        ranked.append((chunk, final_score))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ──────────────────────────────────────────────────────────────────────────────
# ALGORITHM 2 : BM25 HYBRID (SEMANTIC + KEYWORD)
#
# Goal: handle technical queries where exact terminology matters.
#
# Web-scraped content about a specialised domain (e.g. InnoWings SIGs, image
# captions) often contains unique proper nouns and abbreviations.  Pure
# embedding similarity can miss these because the model generalises meaning.
# BM25 is a classical keyword-matching algorithm that rewards exact term
# overlap, weighted by how *rare* those terms are across the corpus (IDF).
#
# Hybrid scoring:
#   final = 0.7 × avg_semantic  +  0.3 × normalised_BM25
#
# Key fix vs. the original: we score each query variation *separately* through
# BM25 and average the scores.  Concatenating all variations into one query
# would inflate term frequencies and corrupt IDF — each variation must be
# treated as an independent evidence source.
# ──────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """
    Lowercases and splits on non-alphanumeric characters.
    Shared by Algorithm 2 and Algorithm 3 so tokenisation is consistent.
    """
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def _algorithm_2_bm25_hybrid(
    chunk_scores: Dict[str, float],
    questions:    List[str],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Combines dense (embedding) similarity with sparse (BM25) keyword matching.
    BM25 scores are averaged across query variations and normalised to [0, 1]
    before blending with the semantic scores.
    """
    chunks = list(chunk_scores.keys())
    tokenized_chunks = [_tokenize(c) for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)

    # Score each query variation independently to preserve BM25's IDF integrity.
    # Averaging the per-query score vectors gives each query equal voting power.
    per_query_bm25 = np.array([
        bm25.get_scores(_tokenize(q)) for q in questions
    ])                                          # shape: (num_queries, num_chunks)
    avg_bm25 = per_query_bm25.mean(axis=0)     # shape: (num_chunks,)

    # Normalise to [0, 1] so BM25 and semantic scores are on the same scale
    max_bm25 = avg_bm25.max()
    norm_bm25 = avg_bm25 / max_bm25 if max_bm25 > 0 else avg_bm25

    num_queries = len(questions)
    ranked = []
    for i, chunk in enumerate(chunks):
        # Average semantic similarity across all queries (same derivation as Algo 1)
        avg_semantic = chunk_scores[chunk] / num_queries

        hybrid_score = 0.7 * avg_semantic + 0.3 * norm_bm25[i]
        ranked.append((chunk, hybrid_score))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


# ──────────────────────────────────────────────────────────────────────────────
# ALGORITHM 3 : MAXIMAL MARGINAL RELEVANCE (MMR)
#
# Goal: prevent the top 5 results from being near-duplicate chunks.
#
# The problem with Algorithms 1 and 2:
#   Both rank chunks purely by relevance.  For a web-scraped database, the same
#   fact often appears in multiple chunks (e.g. the same event described in a
#   headline, a paragraph, and an image caption).  Returning 5 near-identical
#   chunks wastes the context window and gives the LLM no new information after
#   the first chunk.
#
# MMR solution (Carbonell & Goldstein, 1998):
#   Greedily build the result set one chunk at a time.  At each step, pick the
#   chunk that best balances:
#     - Relevance to the original query           (weight λ)
#     - Dissimilarity to already-selected chunks  (weight 1−λ)
#
#   MMR score = λ × relevance(c) − (1−λ) × max_similarity(c, selected)
#
#   λ = 0.6 → slight lean toward relevance while still enforcing diversity.
#   λ = 1.0 → pure relevance (degenerates to Algorithm 1).
#   λ = 0.0 → pure diversity (ignores relevance entirely).
#
# For a database with image annotations and web content, diversity is especially
# valuable: we want one chunk about the event text, one with the image caption,
# one with the URL reference, etc. — not five paraphrases of the same paragraph.
# ──────────────────────────────────────────────────────────────────────────────

def _cosine_from_lists(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two plain Python lists."""
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x ** 2 for x in a))
    norm_b = math.sqrt(sum(x ** 2 for x in b))
    return dot / (norm_a * norm_b + 1e-9)


def _algorithm_3_mmr(
    chunk_scores:    Dict[str, float],
    chunk_meta:      Dict[str, dict],
    embedding_client,
    num_queries:     int,
    lambda_val:      float = 0.6,
    top_k:           int   = 5,
) -> List[Tuple[str, float]]:
    """
    Selects a diverse, relevant set of chunks using Maximal Marginal Relevance.

    Steps
    -----
    1. Normalise relevance scores to [0, 1].
    2. Apply a small bonus for chunks with rich metadata (URL, image alt-text).
    3. Embed every candidate chunk once (needed to measure chunk-to-chunk sim).
    4. Greedily select chunks, each time picking the one with the highest MMR
       score given the chunks already selected.
    """
    chunks = list(chunk_scores.keys())

    # Normalise relevance scores to [0, 1]
    max_score = max(chunk_scores.values()) or 1.0
    relevance = {c: chunk_scores[c] / (max_score * num_queries) for c in chunks}

    # Small metadata bonus: chunks that carry a source URL or image annotation
    # are more informative for a web+image database, so nudge their relevance up.
    for chunk in chunks:
        meta = chunk_meta.get(chunk, {})
        if meta.get("source_url") or meta.get("alt_text") or meta.get("image_caption"):
            relevance[chunk] = min(1.0, relevance[chunk] + 0.05)

    # Embed all candidate chunks to compute pairwise cosine similarity.
    # Chunks are truncated to 2000 chars to stay within the embedding token limit.
    def embed(text: str) -> List[float]:
        resp = embedding_client.embeddings.create(
            input=text[:2000].replace("\n", " "),
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        )
        return resp.data[0].embedding

    chunk_embeddings = {c: embed(c) for c in chunks}

    # Greedy MMR selection loop
    selected:  List[Tuple[str, float]] = []
    remaining: List[str]               = chunks.copy()

    while len(selected) < top_k and remaining:
        if not selected:
            # First pick: simply the most relevant chunk
            best = max(remaining, key=lambda c: relevance[c])
        else:
            # MMR score = λ × relevance  −  (1−λ) × max_sim_to_any_selected_chunk
            # The penalty term discourages selecting a chunk that paraphrases
            # something already in the result set.
            def mmr_score(c: str) -> float:
                max_sim = max(
                    _cosine_from_lists(chunk_embeddings[c], chunk_embeddings[s])
                    for s, _ in selected
                )
                return lambda_val * relevance[c] - (1 - lambda_val) * max_sim

            best = max(remaining, key=mmr_score)

        selected.append((best, relevance[best]))
        remaining.remove(best)

    return selected


# ──────────────────────────────────────────────────────────────────────────────
# ENSEMBLE : BORDA COUNT
#
# Each of the three algorithms produces an independent ranking from a different
# perspective:
#   Algo 1 (Weighted)  — rewards consistent, high-confidence retrieval
#   Algo 2 (BM25)      — rewards exact keyword matches for technical terms
#   Algo 3 (MMR)       — rewards relevance AND diversity
#
# Borda count is a rank-aggregation method: a chunk ranked 1st out of N gets
# N−1 points, 2nd gets N−2, etc.  Summing across all three algorithms surfaces
# chunks that are strong across *multiple* criteria, not just one.
#
# MMR receives a reduced weight (0.8×) in the Borda sum because it optimises
# diversity rather than pure relevance — it should influence the final set but
# not dominate it.
# ──────────────────────────────────────────────────────────────────────────────

def _ensemble_borda(
    weighted_list: List[Tuple[str, float]],
    bm25_list:     List[Tuple[str, float]],
    mmr_list:      List[Tuple[str, float]],
    top_k: int = 5,
) -> List[str]:
    """
    Aggregates three ranked lists into a single ranking using weighted Borda
    count.  Returns the top top_k chunk texts.
    """
    all_chunks = set()
    for lst in (weighted_list, bm25_list, mmr_list):
        all_chunks.update(chunk for chunk, _ in lst)

    n = len(all_chunks)
    borda: Dict[str, float] = defaultdict(float)

    # Borda points: rank 0 (best) → n−1 points, rank 1 → n−2, …
    for rank, (chunk, _) in enumerate(weighted_list):
        borda[chunk] += (n - rank)           # full weight: relevance + coverage signal

    for rank, (chunk, _) in enumerate(bm25_list):
        borda[chunk] += (n - rank)           # full weight: keyword-matching signal

    for rank, (chunk, _) in enumerate(mmr_list):
        borda[chunk] += (n - rank) * 0.8    # reduced weight: MMR optimises diversity,
                                             # not relevance, so it must not dominate

    # Tie-break using raw scores from each algorithm (scaled small to avoid
    # overriding the rank-based points)
    for chunk, score in weighted_list:
        borda[chunk] += score * 0.1
    for chunk, score in bm25_list:
        borda[chunk] += score * 0.1
    for chunk, score in mmr_list:
        borda[chunk] += score * 0.1

    sorted_chunks = sorted(borda.items(), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in sorted_chunks[:top_k]]


# ──────────────────────────────────────────────────────────────────────────────
# MAIN RAG FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def rag_answer(question: str) -> str:
    """
    Full RAG pipeline: question → retrieved chunks → LLM answer.

    Configuration via environment variables
    ----------------------------------------
    RAG_ALGORITHM        : "weighted" | "bm25" | "mmr" | "ensemble" (default)
    RAG_EXPAND_QUESTIONS : "true" (default) | "false"
    EMBEDDING_MODEL      : embedding model name (default: text-embedding-3-small)
    CHROMA_PATH          : path to the ChromaDB directory
    """
    algorithm     = os.getenv("RAG_ALGORITHM", "ensemble").lower()
    use_expansion = os.getenv("RAG_EXPAND_QUESTIONS", "true").lower() == "true"

    # ── Clients ──────────────────────────────────────────────────────────────

    embedding_client = AzureOpenAI(
        azure_endpoint="https://api-iw.azure-api.net/sig-embedding/openai/deployments/text-embedding-3-small/embeddings?api-version=2024-10-21",
        api_key=API_KEY,
        api_version="2024-10-21",
    )
    GPT_client = AzureOpenAI(
        azure_endpoint="https://api-iw.azure-api.net/sig-shared-jpeast/deployments/gpt-4o-mini/chat/completions?api-version=2025-01-01-preview",
        api_key=API_KEY,
        api_version="2025-01-01-preview",
    )

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    CHROMA_PATH = str(BASE_DIR / (os.getenv("CHROMA_PATH") or "chroma_db/chroma_db"))

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection    = chroma_client.get_collection(name="Innowing_db")

    # ── Step 1: Question expansion ───────────────────────────────────────────
    # More query variations → wider semantic coverage → fewer missed chunks.
    query_list = (
        _generate_question_variations(question, num_variations=4)
        if use_expansion
        else [question]
    )

    # ── Step 2: Multi-query retrieval ────────────────────────────────────────
    # Each variation independently queries ChromaDB.  We accumulate similarity
    # scores, hit counts, and metadata across all queries.
    chunk_scores, chunk_hits, chunk_meta = _multi_query_retrieve(
        query_list, collection, embedding_client, top_k=15
    )

    # ── Step 2b: Sibling chunk resolution ────────────────────────────────────
    # If any retrieved chunk is part of a multi-chunk image, inject all sibling
    # chunks into the candidate pool.  This prevents the split-image problem
    # where chunk 0 (visual description) is retrieved but chunk 1 (event names
    # and dates) is never seen by the ranking algorithms.
    _resolve_sibling_chunks(chunk_scores, chunk_hits, chunk_meta, collection)

    # ── Step 3: Ranking ──────────────────────────────────────────────────────
    if algorithm == "weighted":
        top_chunks = [
            c for c, _ in _algorithm_1_weighted(chunk_scores, chunk_hits, len(query_list))
        ]
    elif algorithm == "bm25":
        top_chunks = [
            c for c, _ in _algorithm_2_bm25_hybrid(chunk_scores, query_list)
        ]
    elif algorithm == "mmr":
        top_chunks = [
            c for c, _ in _algorithm_3_mmr(
                chunk_scores, chunk_meta, embedding_client, len(query_list)
            )
        ]
    else:  # "ensemble" — default and recommended
        weighted = _algorithm_1_weighted(chunk_scores, chunk_hits, len(query_list))
        bm25     = _algorithm_2_bm25_hybrid(chunk_scores, query_list)
        mmr      = _algorithm_3_mmr(
            chunk_scores, chunk_meta, embedding_client, len(query_list)
        )
        top_chunks = _ensemble_borda(weighted, bm25, mmr)

    # ── Step 4: Generate answer ───────────────────────────────────────────────
    # The top chunks are formatted as numbered context documents and injected
    # into the prompt.  The system message instructs the model to stay grounded
    # in those documents and to treat image description text as factual —
    # without this instruction GPT tends to hedge on facts it reads inside
    # image descriptions, treating them as subjective observations rather than
    # structured data.
    context = "\n\n---\n\n".join(
        f"Document {i+1}:\n{chunk}" for i, chunk in enumerate(top_chunks)
    )

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
            "content": f"Context documents:\n{context}\n\nQuestion: {question}",
        },
    ]

    answer = GPT_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    ).choices[0].message.content

    return answer


# ──────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────────────────────

def generate_rag_answers(questions: List[str]):
    """
    Input:  list of question strings
    Output: zip of (question, answer) pairs

    Example
    -------
    from rag import generate_rag_answers

    for q, a in generate_rag_answers(["What SIGs does InnoWings have?"]):
        print(q, "->", a)
    """
    answers = []
    for question in questions:
        print(f"Answering: {question[:80]}{'...' if len(question) > 80 else ''}")
        answers.append(rag_answer(question))
    return zip(questions, answers)


if __name__ == "__main__":
    for q, a in generate_rag_answers([
        "Which SIG should I join if I enjoy swimming?"
    ]):
        print(f"\nQ: {q}\nA: {a}")