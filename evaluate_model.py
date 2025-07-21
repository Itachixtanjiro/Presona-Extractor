import os
import faiss
import numpy as np
import json
import time
from sentence_transformers import SentenceTransformer
from src.preprocessing import read_docx, read_pdf, clean_text, _regex_sent_tokenize
from src.config import EMBEDDING_MODEL
from src.llm_extraction import _call_mistral
from src.logger import log

# --- CONFIGURATION ---
FINE_TUNED_MODEL_PATH = f"./models/finetuned-{EMBEDDING_MODEL.split('/')[-1]}"
BASE_MODEL_PATH = EMBEDDING_MODEL
CORPUS_DIR = "rules_problem_statement"
TEST_QUERIES = [
    "What are the main challenges for a Revenue Manager?"
]
TOP_K = 5

def load_corpus(docs_dir: str) -> list[str]:
    """Loads all sentences from documents in a directory."""
    log.info(f"Loading corpus from: {docs_dir}")
    all_sentences = []
    for filename in os.listdir(docs_dir):
        file_path = os.path.join(docs_dir, filename)
        text = ""
        if filename.endswith(".docx"):
            text = read_docx(file_path)
        elif filename.endswith(".pdf"):
            text = read_pdf(file_path)
        else:
            continue
        all_sentences.extend(_regex_sent_tokenize(clean_text(text)))
    log.info(f"Loaded {len(all_sentences)} sentences into the corpus.")
    return all_sentences

def create_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """Creates a FAISS index for searching."""
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def search(query: str, model: SentenceTransformer, index: faiss.Index, corpus: list[str], top_k: int) -> list[str]:
    """Encodes a query and searches the index for the top_k most similar sentences."""
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)
    _, indices = index.search(query_embedding, top_k)
    return [corpus[i] for i in indices[0]]

def get_extractive_summary(query: str, context: str) -> dict:
    """Uses the LLM to extract persona, pain points, and goals from the retrieved context."""
    prompt = f"""
You are an expert analyst. Based on the following QUERY and the provided CONTEXT, answer the questions.

QUERY:
"{query}"

CONTEXT:
\"\"\"{context}\"\"\"

QUESTIONS:
1. Who is the persona being addressed in the context? (e.g., "CMO", "Revenue Manager")
2. What specific pain points for this persona are mentioned in the context?
3. What are the goals or actions suggested to resolve these pain points?

Reply with only a JSON object with keys "persona", "pain_points" (a list of strings), and "goals_actions" (a list of strings).
If no relevant information is found, return an empty value for that key.
Example:
{{"persona": "Revenue Manager", "pain_points": ["Difficulty in forecasting sales."], "goals_actions": ["Implement a new billing solution."]}}
"""
    try:
        response_text = _call_mistral(prompt)
        return json.loads(response_text)
    except Exception as e:
        log.error(f"Failed to get extractive summary: {e}")
        return {"persona": "Error", "pain_points": [str(e)], "goals_actions": []}

def run_evaluation():
    """Main function to run the evaluation process."""
    log.info("--- Starting Model Evaluation ---")
    corpus = load_corpus(CORPUS_DIR)
    if not corpus:
        log.error("Corpus is empty. Aborting evaluation."); return

    log.info(f"Loading base model: {BASE_MODEL_PATH}")
    base_model = SentenceTransformer(BASE_MODEL_PATH)
    
    if not os.path.exists(FINE_TUNED_MODEL_PATH):
        log.error(f"Fine-tuned model not found at: {FINE_TUNED_MODEL_PATH}. Run 'python fine_tune_model.py' first."); return
    log.info(f"Loading fine-tuned model: {FINE_TUNED_MODEL_PATH}")
    tuned_model = SentenceTransformer(FINE_TUNED_MODEL_PATH)

    log.info("Generating embeddings for the corpus with both models...")
    base_embeddings = base_model.encode(corpus); tuned_embeddings = tuned_model.encode(corpus)
    base_index = create_faiss_index(base_embeddings); tuned_index = create_faiss_index(tuned_embeddings)

    for query in TEST_QUERIES:
        print("\n" + "="*80); print(f"QUERY: {query}"); print("="*80 + "\n")
        
        # --- Process Base Model ---
        base_results = search(query, base_model, base_index, corpus, TOP_K)
        base_context = "\n".join(base_results)
        base_summary = get_extractive_summary(query, base_context)
        time.sleep(1)
        
        print(f"--- BASE MODEL ({BASE_MODEL_PATH}) ---")
        print("  [Extracted Insights]:")
        print(f"    - Persona: {base_summary.get('persona', 'N/A')}")
        print(f"    - Pain Points: {base_summary.get('pain_points', 'N/A')}")
        print(f"    - Goals/Actions: {base_summary.get('goals_actions', 'N/A')}")
        print("\n  [Retrieved Sentences]:")
        for i, res in enumerate(base_results): print(f"    {i+1}. {res}")
            
        # --- Process Fine-Tuned Model ---
        tuned_results = search(query, tuned_model, tuned_index, corpus, TOP_K)
        tuned_context = "\n".join(tuned_results)
        tuned_summary = get_extractive_summary(query, tuned_context)
        time.sleep(1)

        print(f"\n--- FINE-TUNED MODEL ({FINE_TUNED_MODEL_PATH}) ---")
        print("  [Extracted Insights]:")
        print(f"    - Persona: {tuned_summary.get('persona', 'N/A')}")
        print(f"    - Pain Points: {tuned_summary.get('pain_points', 'N/A')}")
        print(f"    - Goals/Actions: {tuned_summary.get('goals_actions', 'N/A')}")
        print("\n  [Retrieved Sentences]:")
        for i, res in enumerate(tuned_results): print(f"    {i+1}. {res}")
            
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    run_evaluation()