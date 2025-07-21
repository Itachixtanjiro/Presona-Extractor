import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List
from src.config import EMBEDDING_MODEL
from src.logger import log

# --- Model Loading ---
# For a custom fine-tuned model, you would save your model locally and point
# EMBEDDING_MODEL in config.py to the directory path, e.g., "./models/my-finetuned-model"
log.info(f"Loading embedding model: {EMBEDDING_MODEL}")
try:
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    log.info("Embedding model loaded successfully.")
except Exception as e:
    log.error(f"Failed to load embedding model: {e}", exc_info=True)
    embedder = None

def generate_embeddings(chunks: List[str], normalize: bool = True) -> np.ndarray:
    """
    Generates embeddings for a list of text chunks.
    
    Args:
        chunks (List[str]): The text chunks to embed.
        normalize (bool): If True, normalizes embeddings for Cosine Similarity.
    
    Returns:
        np.ndarray: The generated embeddings.
    """
    if embedder is None:
        raise RuntimeError("Embedding model is not available.")
        
    log.info(f"Generating embeddings for {len(chunks)} chunks...")
    # The model produces float32 vectors, which is what FAISS expects
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    
    if normalize:
        log.info("Normalizing embeddings for Cosine Similarity search.")
        faiss.normalize_L2(embeddings)
        
    log.info(f"Embeddings generated with shape: {embeddings.shape}")
    return embeddings