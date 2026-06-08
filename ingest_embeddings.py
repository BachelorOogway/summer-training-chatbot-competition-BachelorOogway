#!/usr/bin/env python3
"""
Chunk scraped website text and image annotations, generate embeddings, and store in ChromaDB.

Examples:
    python ingest_embeddings.py
    python ingest_embeddings.py --fresh
    python ingest_embeddings.py --text-only
    python ingest_embeddings.py --images-only
    python ingest_embeddings.py --save-chunks chunks_manifest.json
    python ingest_embeddings.py --dataset data.json --image-file image_annotations.json
    python ingest_embeddings.py --chunk-overlap 200
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chromadb
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AzureOpenAI

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_DATASET = BASE_DIR / "data.json"
DEFAULT_IMAGE_FILE = BASE_DIR / "image_annotations.json"
DEFAULT_CHROMA_PATH = BASE_DIR / "chroma_db" / "chroma_db"
DEFAULT_COLLECTION = "Innowing_db"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_ENDPOINT = (
    "https://api-iw.azure-api.net/sig-embedding/openai/deployments/"
    "text-embedding-3-small/embeddings?api-version=2024-10-21"
)
DEFAULT_SOURCE_LABEL = "HKU InnoWings / InnoAcademy"


@dataclass
class ChunkRecord:
    text: str
    chunk_id: str
    metadata: Dict[str, Any]


@dataclass
class IngestConfig:
    dataset_path: Path = DEFAULT_DATASET
    image_file: Path = DEFAULT_IMAGE_FILE
    chroma_path: Path = DEFAULT_CHROMA_PATH
    collection_name: str = DEFAULT_COLLECTION
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    chunk_size: int = 3000
    chunk_overlap: int = 200
    batch_size: int = 50
    include_text: bool = True
    include_images: bool = True
    fresh: bool = False
    save_chunks_path: Optional[Path] = None


def load_environment() -> str:
    load_dotenv(BASE_DIR / ".env")
    load_dotenv(BASE_DIR.parent / ".env")

    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing Azure OpenAI credentials. Set AZURE_OPENAI_API_KEY in .env or environment."
        )
    return api_key


def make_chunk_id(prefix: str, identifier: str, chunk_index: int) -> str:
    raw = f"{prefix}:{identifier}:{chunk_index}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def build_text_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_web_documents(
    documents: Sequence[Dict[str, Any]],
    splitter: RecursiveCharacterTextSplitter,
) -> List[ChunkRecord]:
    records: List[ChunkRecord] = []

    for doc in documents:
        url = doc.get("url", "").strip()
        text = doc.get("text", "").strip()
        if not text:
            continue

        chunks = splitter.split_text(text)
        for chunk_index, chunk in enumerate(chunks):
            records.append(
                ChunkRecord(
                    text=chunk,
                    chunk_id=make_chunk_id("web", url, chunk_index),
                    metadata={
                        "content_type": "web",
                        "source": DEFAULT_SOURCE_LABEL,
                        "url": url,
                        "chunk_index": chunk_index,
                        "total_chunks": len(chunks),
                        "chunk_size": splitter._chunk_size,
                        "chunk_overlap": splitter._chunk_overlap,
                    },
                )
            )

    return records


def _extract_image_text(entry: Dict[str, Any]) -> str:
    if "chunks" in entry:
        return "\n\n".join(entry["chunks"])

    annotation = entry.get("annotation", "")
    if isinstance(annotation, list) and annotation:
        return str(annotation[0])
    if isinstance(annotation, str):
        return annotation
    return ""


def chunk_image_annotations(
    annotations: Dict[str, Any],
    splitter: RecursiveCharacterTextSplitter,
) -> List[ChunkRecord]:
    records: List[ChunkRecord] = []

    for image_name, entry in annotations.items():
        if not isinstance(entry, dict):
            continue

        image_path = str(entry.get("path", image_name))
        full_text = _extract_image_text(entry).strip()
        if not full_text:
            continue

        if "chunks" in entry and entry["chunks"]:
            chunks = [c.strip() for c in entry["chunks"] if c and c.strip()]
        else:
            chunks = splitter.split_text(full_text)

        for chunk_index, chunk in enumerate(chunks):
            records.append(
                ChunkRecord(
                    text=chunk,
                    chunk_id=make_chunk_id("image", image_path, chunk_index),
                    metadata={
                        "content_type": "image",
                        "source": DEFAULT_SOURCE_LABEL,
                        "image_name": image_name,
                        "image_path": image_path,
                        "chunk_index": chunk_index,
                        "total_chunks": len(chunks),
                        "chunk_size": splitter._chunk_size,
                        "chunk_overlap": splitter._chunk_overlap,
                    },
                )
            )

    return records


def load_json_file(path: Path, label: str) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_chunks(config: IngestConfig) -> Tuple[List[ChunkRecord], Dict[str, int]]:
    splitter = build_text_splitter(config.chunk_size, config.chunk_overlap)
    all_records: List[ChunkRecord] = []
    stats = {"web_pages": 0, "web_chunks": 0, "images": 0, "image_chunks": 0}

    if config.include_text:
        documents = load_json_file(config.dataset_path, "Dataset")
        if not isinstance(documents, list):
            raise ValueError(f"Dataset must be a JSON list: {config.dataset_path}")

        web_records = chunk_web_documents(documents, splitter)
        all_records.extend(web_records)
        stats["web_pages"] = len(documents)
        stats["web_chunks"] = len(web_records)
        print(
            f"Loaded {len(documents)} web pages -> {len(web_records)} chunks "
            f"(size={config.chunk_size}, overlap={config.chunk_overlap})"
        )

    if config.include_images:
        annotations = load_json_file(config.image_file, "Image annotations")
        if not isinstance(annotations, dict):
            raise ValueError(f"Image annotations must be a JSON object: {config.image_file}")

        image_records = chunk_image_annotations(annotations, splitter)
        all_records.extend(image_records)
        stats["images"] = len(annotations)
        stats["image_chunks"] = len(image_records)
        print(
            f"Loaded {len(annotations)} images -> {len(image_records)} chunks "
            f"(size={config.chunk_size}, overlap={config.chunk_overlap})"
        )

    if not all_records:
        raise RuntimeError("No chunks were produced. Check input files and flags.")

    print(f"Total chunks ready for embedding: {len(all_records)}")
    return all_records, stats


def save_chunks_manifest(records: Sequence[ChunkRecord], path: Path) -> None:
    manifest = [
        {
            "id": record.chunk_id,
            "text": record.text,
            "metadata": record.metadata,
        }
        for record in records
    ]
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(f"Saved chunk manifest to {path}")


def create_embedding_client(api_key: str) -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=DEFAULT_EMBEDDING_ENDPOINT,
        api_key=api_key,
        api_version="2024-10-21",
    )


def embed_texts(
    client: AzureOpenAI,
    texts: Sequence[str],
    model: str,
) -> List[List[float]]:
    response = client.embeddings.create(
        input=[text.replace("\n", " ") for text in texts],
        model=model,
    )
    return [item.embedding for item in response.data]


def setup_collection(
    chroma_path: Path,
    collection_name: str,
    fresh: bool,
) -> chromadb.Collection:
    chroma_path.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(chroma_path))

    if fresh:
        try:
            chroma_client.delete_collection(name=collection_name)
            print(f"Deleted existing collection '{collection_name}'")
        except Exception:
            pass

    return chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_records(
    collection: chromadb.Collection,
    records: Sequence[ChunkRecord],
    client: AzureOpenAI,
    model: str,
    batch_size: int,
) -> None:
    print("Generating embeddings and storing in ChromaDB...")

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        embeddings = embed_texts(client, [record.text for record in batch], model)

        collection.upsert(
            ids=[record.chunk_id for record in batch],
            documents=[record.text for record in batch],
            metadatas=[record.metadata for record in batch],
            embeddings=embeddings,
        )

        end = start + len(batch)
        print(f"  [{end}/{len(records)}] upserted")


def ingest(config: IngestConfig) -> Dict[str, Any]:
    api_key = load_environment()
    records, stats = collect_chunks(config)

    if config.save_chunks_path:
        save_chunks_manifest(records, config.save_chunks_path)

    collection = setup_collection(
        config.chroma_path,
        config.collection_name,
        config.fresh,
    )
    client = create_embedding_client(api_key)
    upsert_records(
        collection,
        records,
        client,
        config.embedding_model,
        config.batch_size,
    )

    summary = {
        **stats,
        "total_chunks": len(records),
        "collection_name": config.collection_name,
        "chroma_path": str(config.chroma_path),
        "embedding_model": config.embedding_model,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
    }

    print("\nIngestion complete")
    print(f"  Collection: {summary['collection_name']}")
    print(f"  Chroma path: {summary['chroma_path']}")
    print(f"  Total chunks: {summary['total_chunks']}")
    print(f"  Web chunks: {summary['web_chunks']}")
    print(f"  Image chunks: {summary['image_chunks']}")
    print(f"  Chunk size: {summary['chunk_size']}")
    print(f"  Chunk overlap: {summary['chunk_overlap']}")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> IngestConfig:
    parser = argparse.ArgumentParser(
        description="Chunk text and image data, embed with Azure OpenAI, and store in ChromaDB.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(os.getenv("DATASET", DEFAULT_DATASET)),
        help="Path to scraped website JSON (default: data.json)",
    )
    parser.add_argument(
        "--image-file",
        type=Path,
        default=Path(os.getenv("IMAGE_ANNOTATIONS", DEFAULT_IMAGE_FILE)),
        help="Path to image annotations JSON (default: image_annotations.json)",
    )
    parser.add_argument(
        "--chroma-path",
        type=Path,
        default=Path(os.getenv("CHROMA_PATH", DEFAULT_CHROMA_PATH)),
        help="ChromaDB persistence directory",
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("COLLECTION_NAME", DEFAULT_COLLECTION),
        help="Chroma collection name",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        help="Azure embedding deployment name",
    )
    parser.add_argument("--chunk-size", type=int, default=3000)
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Number of overlapping characters between consecutive chunks (must be smaller than --chunk-size)",
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete and recreate the collection before ingest",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Ingest only website text from data.json",
    )
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Ingest only image annotations",
    )
    parser.add_argument(
        "--save-chunks",
        type=Path,
        default=None,
        help="Optional path to save all generated chunks as JSON",
    )

    args = parser.parse_args(argv)

    if args.text_only and args.images_only:
        parser.error("Use only one of --text-only or --images-only")
    if args.chunk_size <= 0:
        parser.error("--chunk-size must be greater than 0")
    if args.chunk_overlap < 0:
        parser.error("--chunk-overlap cannot be negative")
    if args.chunk_overlap >= args.chunk_size:
        parser.error("--chunk-overlap must be smaller than --chunk-size")

    include_text = not args.images_only
    include_images = not args.text_only

    return IngestConfig(
        dataset_path=args.dataset.resolve(),
        image_file=args.image_file.resolve(),
        chroma_path=args.chroma_path.resolve(),
        collection_name=args.collection,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
        include_text=include_text,
        include_images=include_images,
        fresh=args.fresh,
        save_chunks_path=args.save_chunks.resolve() if args.save_chunks else None,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        config = parse_args(argv)
        ingest(config)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())