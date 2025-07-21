import faiss
import pickle
import os
import numpy as np
from typing import List, Tuple
from src.config import (
    INDEX_FILE, META_FILE, VECTOR_DIM, VECTOR_INDEX_TYPE, 
    IVF_NLIST, PQ_M, MIN_TRAIN_SIZE, TOP_K_RAG
)
from src.logger import log

def _create_index() -> faiss.Index:
    """Creates a new FAISS index based on the configured type."""
    log.info(f"Creating new FAISS index of type: {VECTOR_INDEX_TYPE}")
    
    if VECTOR_INDEX_TYPE == "IVFFlat":
        quantizer = faiss.IndexFlatIP(VECTOR_DIM)
        index = faiss.IndexIVFFlat(quantizer, VECTOR_DIM, IVF_NLIST, faiss.METRIC_INNER_PRODUCT)
    elif VECTOR_INDEX_TYPE == "IVFPQ":
        quantizer = faiss.IndexFlatIP(VECTOR_DIM)
        index = faiss.IndexIVFPQ(quantizer, VECTOR_DIM, IVF_NLIST, PQ_M, 8)
    else:
        log.warning(f"Invalid VECTOR_INDEX_TYPE '{VECTOR_INDEX_TYPE}'. Defaulting to IndexFlatIP.")
        index = faiss.IndexFlatIP(VECTOR_DIM)
        
    return index

def load_or_initialize_vector_store() -> Tuple[faiss.Index, List[str]]:
    """
    Loads the FAISS index and metadata. If they don't exist, initializes new ones.
    """
    if os.path.exists(INDEX_FILE) and os.path.exists(META_FILE):
        log.info(f"Loading FAISS index from {INDEX_FILE}")
        index = faiss.read_index(INDEX_FILE)
        
        log.info(f"Loading metadata from {META_FILE}")
        with open(META_FILE, "rb") as f:
            metadata = pickle.load(f)
            
        log.info(f"Loaded index with {index.ntotal} vectors.")
    else:
        log.info("Initializing new vector store.")
        index = _create_index()
        metadata = []
        
    return index, metadata

def save_vector_store(index: faiss.Index, metadata: List[str]):
    """Saves the FAISS index and metadata to disk."""
    log.info(f"Saving FAISS index to {INDEX_FILE}")
    faiss.write_index(index, INDEX_FILE)
    
    log.info(f"Saving metadata to {META_FILE}")
    with open(META_FILE, "wb") as f:
        pickle.dump(metadata, f)
    log.info("Vector store saved successfully.")

def _is_ivf_index(index: faiss.Index) -> bool:
    """Checks if the index is an IVF type."""
    return isinstance(index, (faiss.IndexIVFFlat, faiss.IndexIVFPQ))

def add_to_vector_store(index: faiss.Index, metadata: List[str], new_embeddings: np.ndarray, new_chunks: List[str]):
    """
    Adds new embeddings and chunks to the vector store, handling training if necessary.
    """
    if new_embeddings.shape[0] == 0:
        log.warning("No new embeddings to add.")
        return index

    # --- Training Logic for IVF Indexes ---
    if _is_ivf_index(index) and not index.is_trained:
        log.info("Index is an IVF type but is not trained.")
        
        # Check if we have enough data to perform the initial training
        if new_embeddings.shape[0] >= MIN_TRAIN_SIZE:
            log.info(f"Sufficient data ({new_embeddings.shape[0]}) available for initial training.")
            index.train(new_embeddings)
            log.info("Initial training complete.")
        else:
            # This is a critical state. We have an untrained index and not enough data to train it.
            # We cannot add to it. The best strategy is to log this and wait for more data.
            # For this application, we will raise an error to make the problem visible.
            # A more advanced system might queue the data.
            log.error(f"Cannot add to untrained IVF index. Need at least {MIN_TRAIN_SIZE} vectors for initial training, but only have {new_embeddings.shape[0]}.")
            log.error("Please upload more documents to meet the minimum training size.")
            # Returning the original, untrained index. No data will be added.
            return index

    # --- Add new data to the (now guaranteed to be trained, if IVF) index ---
    log.info(f"Adding {new_embeddings.shape[0]} new vectors to the index.")
    index.add(new_embeddings)
    metadata.extend(new_chunks)
    
    return index

def search_vector_store(index: faiss.Index, metadata: List[str], query_embedding: np.ndarray) -> List[str]:
    """
    Searches the vector store for the most relevant context for a given query embedding.
    """
    if index.ntotal == 0:
        return []

    if hasattr(index, "nprobe"):
        index.nprobe = 10

    log.info(f"Searching for {TOP_K_RAG} relevant documents.")
    distances, indices = index.search(query_embedding, TOP_K_RAG)
    
    valid_indices = [i for i in indices[0] if 0 <= i < len(metadata)]
    
    return [metadata[i] for i in valid_indices]