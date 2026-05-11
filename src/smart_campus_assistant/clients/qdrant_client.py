import logging
from typing import Dict, Any, List
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_ollama import OllamaEmbeddings

# Import your settings where QDRANT_URL and QDRANT_API_KEY are defined
from src.smart_campus_assistant.config.settings import settings

logger = logging.getLogger(__name__)

class CampusKnowledgeClient:
    def __init__(self):
        # Initialize Qdrant using credentials from settings.py
        self.client = QdrantClient(
            url=getattr(settings, "QDRANT_URL", "http://localhost:6333"),
            api_key=getattr(settings, "QDRANT_API_KEY", None),
        )
        self.collection_name = "smart-campus-assistant"
        
        # Initialize the Ollama embedder to convert text queries into vectors
        self.embedder = OllamaEmbeddings(
            model=settings.OLLAMA_EMBED_MODEL,
            base_url=settings.OLLAMA_BASE_URL
        )

    def search_knowledge(self, query_text: str, filters: Dict[str, Any] = None, limit: int = 3):
        """
        Performs a hybrid search: Exact Metadata Filtering + Semantic Vector Search.
        """
        must_conditions = []
        
        # Dynamically build the Qdrant Filter payload based on LLM inputs
        if filters:
            for key, val in filters.items():
                if val is None:
                    continue
                # If the filter is a list (e.g., room_id: ["5.7", "2.2"])
                if isinstance(val, list) and len(val) > 0:
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchAny(any=val)
                        )
                    )
                # If the filter is a single value (e.g., floor: 2)
                elif isinstance(val, (str, int, bool)):
                    must_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=val)
                        )
                    )
                    
        # Construct the final filter object
        qdrant_filter = models.Filter(must=must_conditions) if must_conditions else None

        try:
            # 1. Convert the text query into a vector using Ollama
            query_vector = self.embedder.embed_query(query_text)

            # 2. Search the database using the generated vector and the correct method
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=qdrant_filter,
                limit=limit
            )
            return results.points
        except Exception as e:
            logger.error(f"Qdrant Search Error: {e}")
            return []

# Expose a singleton instance
kb_client = CampusKnowledgeClient()