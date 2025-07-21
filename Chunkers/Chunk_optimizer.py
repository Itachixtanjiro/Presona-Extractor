import os
import fitz  # PyMuPDF
import docx
import re
import pandas as pd
import time
from typing import List

# Constants
CHUNK_LIMIT = 1000  # Words per chunk

# Reading functions
def read_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def read_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    return "\n".join([page.get_text() for page in doc])

def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

# Tokenization utility
def regex_sent_tokenize(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

# Base RAG-style chunking
def rag_style_chunking(text: str, word_limit: int = CHUNK_LIMIT) -> List[str]:
    words = text.split()
    return [f"[DOC{i}] {' '.join(words[i:i+word_limit])}" for i in range(0, len(words), word_limit)]

# Simulated Longformer chunking: sliding window + global attention assumption
def longformer_chunking(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    step = chunk_size - overlap
    return [f"[Longformer DOC{i}] {' '.join(words[i:i+chunk_size])}" for i in range(0, len(words), step)]

# Simulated Reformer chunking: standard chunk + attention reduction
def reformer_chunking(text: str, word_limit: int = CHUNK_LIMIT) -> List[str]:
    words = text.split()
    chunks = [' '.join(words[i:i+word_limit]) for i in range(0, len(words), word_limit)]
    return [f"[Reformer DOC{i}] {chunk}" for i, chunk in enumerate(chunks)]

# Simulated Routing Transformer: route different chunk types
def routing_transformer_chunking(text: str, word_limit: int = CHUNK_LIMIT) -> List[str]:
    words = text.split()
    return [f"[Route DOC{i}] {' '.join(words[i:i+word_limit])}" for i in range(0, len(words), word_limit)]

# Distilled chunking (simulate student model alignment)
def distilled_chunking(text: str, word_limit: int = CHUNK_LIMIT) -> List[str]:
    words = text.split()
    return [f"[Distilled DOC{i}] {' '.join(words[i:i+word_limit])}" for i in range(0, len(words), word_limit)]

# Quantized chunking (simulate chunk-level precision control)
def quantized_chunking(text: str, word_limit: int = CHUNK_LIMIT) -> List[str]:
    words = text.split()
    return [f"[Quantized DOC{i}] {' '.join(words[i:i+word_limit])}" for i in range(0, len(words), word_limit)]

# Benchmark chunking strategies
def benchmark_advanced_chunking(folder_path: str) -> pd.DataFrame:
    report_data = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".docx"):
            text = read_docx(file_path)
        elif filename.endswith(".pdf"):
            text = read_pdf(file_path)
        else:
            continue

        text = clean_text(text)
        total_words = len(text.split())
        total_sentences = len(regex_sent_tokenize(text))

        def time_chunk(func, label):
            start = time.time()
            chunks = func(text)
            end = time.time()
            return len(chunks), round(end - start, 4)

        rag_chunks, rag_time = time_chunk(rag_style_chunking, "RAG")
        long_chunks, long_time = time_chunk(lambda t: longformer_chunking(t, CHUNK_LIMIT, CHUNK_LIMIT // 2), "Longformer")
        ref_chunks, ref_time = time_chunk(reformer_chunking, "Reformer")
        route_chunks, route_time = time_chunk(routing_transformer_chunking, "Routing")
        distill_chunks, distill_time = time_chunk(distilled_chunking, "Distilled")
        quant_chunks, quant_time = time_chunk(quantized_chunking, "Quantized")

        report_data.append({
            "File": filename,
            "Total Words": total_words,
            "Total Sentences": total_sentences,
            "RAG Chunks": rag_chunks,
            "RAG Time (s)": rag_time,
            "Longformer Chunks": long_chunks,
            "Longformer Time (s)": long_time,
            "Reformer Chunks": ref_chunks,
            "Reformer Time (s)": ref_time,
            "Routing Chunks": route_chunks,
            "Routing Time (s)": route_time,
            "Distilled Chunks": distill_chunks,
            "Distilled Time (s)": distill_time,
            "Quantized Chunks": quant_chunks,
            "Quantized Time (s)": quant_time,
        })

    return pd.DataFrame(report_data)

# Run and save output
upload_folder = "TEST"
chunk_report_df = benchmark_advanced_chunking(upload_folder)
csv_path = "advanced_rag_chunking_report.csv"
chunk_report_df.to_csv(csv_path, index=False)

import ace_tools_open as tools; tools.display_dataframe_to_user(name="Advanced Chunking Strategy Report", dataframe=chunk_report_df)
csv_path
