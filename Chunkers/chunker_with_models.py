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

# Chunking strategies
def word_chunking(text: str, word_limit: int = CHUNK_LIMIT) -> List[str]:
    words = text.split()
    return [' '.join(words[i:i+word_limit]) for i in range(0, len(words), word_limit)]

def regex_sent_tokenize(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

def sentence_chunking(text: str, sent_limit: int = 30) -> List[str]:
    sentences = regex_sent_tokenize(text)
    return [' '.join(sentences[i:i+sent_limit]) for i in range(0, len(sentences), sent_limit)]

def sliding_window_chunking(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    step = chunk_size - overlap
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), step)]

# Simulated RNN tokenization (e.g., sentence-by-sentence accumulation)
def rnn_style_chunking(text: str, max_words: int = CHUNK_LIMIT) -> List[str]:
    sentences = regex_sent_tokenize(text)
    chunks, current_chunk = [], []
    count = 0
    for sentence in sentences:
        words = sentence.split()
        if count + len(words) > max_words:
            chunks.append(' '.join(current_chunk))
            current_chunk = words
            count = len(words)
        else:
            current_chunk.extend(words)
            count += len(words)
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

# Simulated RAG-style retrieval (static partitioning + unique ID)
def rag_style_chunking(text: str, word_limit: int = CHUNK_LIMIT) -> List[str]:
    chunks = word_chunking(text, word_limit)
    return [f"[DOC{i}] {chunk}" for i, chunk in enumerate(chunks)]

# Chunking with strategy benchmarking
def benchmark_chunking_methods(folder_path: str) -> pd.DataFrame:
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

        # Word Chunking
        t0 = time.time()
        word_chunks = word_chunking(text)
        t1 = time.time()

        # Sentence Chunking
        sentence_chunks = sentence_chunking(text)
        t2 = time.time()

        # Sliding Window
        sliding_chunks = sliding_window_chunking(text, CHUNK_LIMIT, CHUNK_LIMIT // 2)
        t3 = time.time()

        # RNN-style
        rnn_chunks = rnn_style_chunking(text)
        t4 = time.time()

        # RAG-style
        rag_chunks = rag_style_chunking(text)
        t5 = time.time()

        report_data.append({
            "File": filename,
            "Total Words": total_words,
            "Total Sentences": total_sentences,
            "Word Chunks": len(word_chunks),
            "Word Chunk Time (s)": round(t1 - t0, 4),
            "Sentence Chunks": len(sentence_chunks),
            "Sentence Chunk Time (s)": round(t2 - t1, 4),
            "Sliding Window Chunks": len(sliding_chunks),
            "Sliding Time (s)": round(t3 - t2, 4),
            "RNN Chunks": len(rnn_chunks),
            "RNN Time (s)": round(t4 - t3, 4),
            "RAG Chunks": len(rag_chunks),
            "RAG Time (s)": round(t5 - t4, 4),
        })

    return pd.DataFrame(report_data)

# Run benchmarking on uploaded folder
upload_folder = "TEST"
chunk_report_df = benchmark_chunking_methods(upload_folder)

# Save CSV
csv_path = "chunking_strategy_report.csv"
chunk_report_df.to_csv(csv_path, index=False)

import ace_tools_open as tools; tools.display_dataframe_to_user(name="Chunking Strategy Comparison Report", dataframe=chunk_report_df)
csv_path
