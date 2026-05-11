import os
import glob
import uuid
import logging
import frontmatter

from qdrant_client.http import models
from langchain_ollama import OllamaEmbeddings

from src.smart_campus_assistant.config.settings import settings
from src.smart_campus_assistant.clients.qdrant_client import kb_client

logger = logging.getLogger(__name__)

def sync_knowledge_base(data_dir: str = "data/knowledge"):
    """
    Scans the data/knowledge directory for Markdown files, parses their YAML frontmatter,
    embeds the content via Ollama, and synchronizes them with the Qdrant Vector Database.
    Automatically creates the Qdrant collection if it doesn't exist.
    """
    logger.info("Starting Knowledge Base synchronization...")
    
    if not os.path.exists(data_dir):
        logger.warning(f"Knowledge directory '{data_dir}' does not exist. Creating it...")
        os.makedirs(data_dir, exist_ok=True)
        return

    md_files = glob.glob(os.path.join(data_dir, "**", "*.md"), recursive=True)
    
    if not md_files:
        logger.info(f"No markdown files found in '{data_dir}'. Skipping sync.")
        return

    documents = []
    metadatas = []
    ids = []

    for filepath in md_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                post = frontmatter.load(f)

            # Deterministic UUID prevents duplicating the same file on restarts
            rel_path = os.path.relpath(filepath, data_dir)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, rel_path))

            meta = post.metadata
            meta["source_file"] = rel_path
            
            # Use the body content + title for the semantic embedding
            title = meta.get("title", "")
            content = post.content.strip()
            
            if not content:
                continue
                
            full_text = f"{title}\n\n{content}"

            documents.append(full_text)
            metadatas.append(meta)
            ids.append(point_id)

        except Exception as e:
            logger.error(f"Failed to process {filepath}: {e}")

    if documents:
        try:
            # 1. Initialize Ollama Embedder
            logger.info(f"Generating vectors using Ollama model: {settings.OLLAMA_EMBED_MODEL}...")
            embedder = OllamaEmbeddings(
                model=settings.OLLAMA_EMBED_MODEL,
                base_url=settings.OLLAMA_BASE_URL
            )
            
            # Generate all vectors in one batch
            vectors = embedder.embed_documents(documents)
            
            # 2. Check and Create Qdrant Collection if missing
            collection_name = kb_client.collection_name
            if not kb_client.client.collection_exists(collection_name):
                # Dynamically get the dimension size from the first embedded vector
                vector_size = len(vectors[0])
                logger.info(f"Collection '{collection_name}' not found. Creating it with vector dimension {vector_size}...")
                
                kb_client.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=vector_size, 
                        distance=models.Distance.COSINE
                    )
                )
            
            # 3. Upsert using the modern PointStruct approach
            logger.info("Upserting documents into Qdrant...")
            points = [
                models.PointStruct(id=p_id, vector=vector, payload=meta)
                for p_id, vector, meta in zip(ids, vectors, metadatas)
            ]
            
            kb_client.client.upsert(
                collection_name=collection_name,
                points=points
            )
            
            logger.info(f"Successfully synced {len(documents)} documents to the Vector Database.")
            
        except Exception as e:
            logger.error(f"Failed to upsert documents to Qdrant: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sync_knowledge_base()