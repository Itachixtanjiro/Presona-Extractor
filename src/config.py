import os
from dotenv import load_dotenv

load_dotenv()

# --- DIRECTORIES ---
UPLOAD_DIR = "uploads"
VECTOR_STORE_DIR = "vector_store"
MODELS_DIR = "models"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- FAISS + METADATA ---
INDEX_FILE = os.path.join(VECTOR_STORE_DIR, "index.faiss")
META_FILE = os.path.join(VECTOR_STORE_DIR, "metadata.pkl")

# --- CHUNKING PARAMS ---
CHUNKING_STRATEGY = "advanced"
ADVANCED_CHUNK_SIZE = 10
ADVANCED_CHUNK_OVERLAP = 2
RNN_CHUNK_SIZE = 512

# --- EMBEDDING PARAMS ---
# Point to the fine-tuned model directory
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
EMBEDDING_MODEL = os.path.join(MODELS_DIR, f"finetuned-{EMBEDDING_MODEL_NAME}")
VECTOR_DIM = 768

# --- VECTOR STORE PARAMS ---
VECTOR_INDEX_TYPE = "IVFFlat"
IVF_NLIST = 100
PQ_M = 96
MIN_TRAIN_SIZE = 200
# Number of context passages to retrieve for RAG
TOP_K_RAG = 3 

# --- MISTRAL API ---
MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL = "mistral-medium"

# --- LOGGING ---
LOG_FILE = "extraction.log"
LOG_LEVEL = "DEBUG"