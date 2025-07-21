import os
import time
import re
import docx
import pandas as pd
from typing import List

# Constants
CHUNK_LIMIT = 1000

# ---------- Text Reading & Cleaning ----------
def read_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def regex_sent_tokenize(text: str) -> List[str]:
    return re.split(r'(?<=[.!?]) +', text)

# ---------- RAG Base Chunking ----------
def rag_style_chunking(text: str, word_limit: int = CHUNK_LIMIT) -> List[str]:
    words = text.split()
    return [f"[DOC{i}] {' '.join(words[i:i+word_limit])}" for i in range(0, len(words), word_limit)]

# ---------- Model Simulations ----------
def simulate_longformer_on_chunks(chunks: List[str]) -> float:
    time.sleep(0.1)
    return 0.85  # Simulated F1 or accuracy

def simulate_reformer_on_chunks(chunks: List[str]) -> float:
    time.sleep(0.1)
    return 0.83

def simulate_quantized_rag(chunks: List[str]) -> float:
    time.sleep(0.05)
    return 0.80

# ---------- Benchmark Function ----------
def benchmark_models_on_doc(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"✅ Processing: {file_path}")
    text = read_docx(file_path)
    text = clean_text(text)

    benchmark_rows = []

    # --- RAG chunking ---
    rag_chunks = rag_style_chunking(text)

    # --- Strategy 1: RAG + Longformer ---
    start = time.time()
    score_1 = simulate_longformer_on_chunks(rag_chunks)
    t1 = time.time() - start

    # --- Strategy 2: RAG + Reformer ---
    start = time.time()
    score_2 = simulate_reformer_on_chunks(rag_chunks)
    t2 = time.time() - start

    # --- Strategy 3: RAG + Quantization ---
    start = time.time()
    score_3 = simulate_quantized_rag(rag_chunks)
    t3 = time.time() - start

    benchmark_rows.append({
        "File": os.path.basename(file_path),
        "Chunks": len(rag_chunks),
        "RAG+Longformer Score": score_1,
        "RAG+Longformer Time (s)": round(t1, 4),
        "RAG+Reformer Score": score_2,
        "RAG+Reformer Time (s)": round(t2, 4),
        "RAG+Quantized Score": score_3,
        "RAG+Quantized Time (s)": round(t3, 4),
    })

    return pd.DataFrame(benchmark_rows)

# ---------- Run Benchmark for All Files ----------
all_results = []

directory_path = "TEST"
for file_name in os.listdir(directory_path):
    if not file_name.endswith(".docx"):
        continue
    full_path = os.path.abspath(os.path.join(directory_path, file_name))
    print(f"📂 Resolved path: {full_path}")
    print(f"📄 Benchmarking: {file_name}")
    try:
        df = benchmark_models_on_doc(full_path)
        all_results.append(df)
    except Exception as e:
        print(f"❌ Failed to process {file_name}: {e}")

# Combine all results
if all_results:
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv("rag_combinations_benchmark.csv", index=False)
    import ace_tools_open as tools
    tools.display_dataframe_to_user(name="RAG Combo Model Benchmark", dataframe=final_df)
else:
    print("⚠️ No valid DOCX files were processed.")
