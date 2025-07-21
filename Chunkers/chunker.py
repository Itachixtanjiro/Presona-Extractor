import os
import fitz  # PyMuPDF
import docx
import time
from typing import List
import pandas as pd
import re
import ace_tools_open as tools

# Constants
CHUNK_LIMIT = 1000  # Adjustable token/word limit per chunk

# Helper functions
def read_docx(file_path: str) -> str:
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def read_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    return "\n".join([page.get_text() for page in doc])

def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def regex_sent_tokenize(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

# Chunking strategies
def word_chunking(text: str, word_limit: int) -> List[str]:
    words = text.split()
    return [' '.join(words[i:i + word_limit]) for i in range(0, len(words), word_limit)]

def sentence_chunking(text: str, sentence_limit: int) -> List[str]:
    sentences = regex_sent_tokenize(text)
    return [' '.join(sentences[i:i + sentence_limit]) for i in range(0, len(sentences), sentence_limit)]

def sliding_window_chunking(text: str, chunk_size: int, stride: int) -> List[str]:
    words = text.split()
    chunks = []
    for i in range(0, len(words) - chunk_size + 1, stride):
        chunks.append(' '.join(words[i:i + chunk_size]))
    return chunks

# Analyze performance of each chunking method
def process_documents_with_analysis(folder_path: str, word_limit: int = 1000) -> pd.DataFrame:
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

        # Apply word chunking
        start_time = time.time()
        word_chunks = word_chunking(text, word_limit)
        word_time = time.time() - start_time

        # Apply sentence chunking
        start_time = time.time()
        sentence_chunks = sentence_chunking(text, 30)
        sentence_time = time.time() - start_time

        # Apply sliding window chunking
        start_time = time.time()
        sliding_chunks = sliding_window_chunking(text, word_limit, int(word_limit / 2))
        sliding_time = time.time() - start_time

        report_data.append({
            "File": filename,
            "Total Words": total_words,
            "Total Sentences": total_sentences,
            "Word Chunks": len(word_chunks),
            "Word Chunk Time (s)": round(word_time, 4),
            "Sentence Chunks": len(sentence_chunks),
            "Sentence Chunk Time (s)": round(sentence_time, 4),
            "Sliding Window Chunks": len(sliding_chunks),
            "Sliding Window Time (s)": round(sliding_time, 4)
        })
    return pd.DataFrame(report_data)

# Process all files and display optimization report
folder_path = "TEST"
chunk_report_df_optimized = process_documents_with_analysis(folder_path)
tools.display_dataframe_to_user(name="Chunking Optimization Report", dataframe=chunk_report_df_optimized)

# Return chunk-level DataFrame for later use
def generate_final_chunks(folder_path: str, chunking_strategy: str = 'word', chunk_size: int = 1000) -> pd.DataFrame:
    chunks_data = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if file.endswith(".docx"):
            raw_text = read_docx(file_path)
        elif file.endswith(".pdf"):
            raw_text = read_pdf(file_path)
        else:
            continue
        raw_text = clean_text(raw_text)

        if chunking_strategy == 'word':
            chunks = word_chunking(raw_text, chunk_size)
        elif chunking_strategy == 'sentence':
            chunks = sentence_chunking(raw_text, chunk_size)
        elif chunking_strategy == 'sliding':
            stride = int(chunk_size / 2)
            chunks = sliding_window_chunking(raw_text, chunk_size, stride)
        else:
            continue

        for i, chunk in enumerate(chunks):
            chunks_data.append({
                "file_name": file,
                "chunk_index": i,
                "chunk_strategy": chunking_strategy,
                "word_count": len(chunk.split()),
                "text": chunk
            })
    return pd.DataFrame(chunks_data)

# Example of final chunk processing with default word-based chunking
final_chunks_df = generate_final_chunks(folder_path, chunking_strategy='word', chunk_size=1000)
tools.display_dataframe_to_user(name="Final Chunks (Word-Based)", dataframe=final_chunks_df)

