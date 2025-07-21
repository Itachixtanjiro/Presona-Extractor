import re
import docx
import fitz
from typing import List, Callable
from src.config import CHUNKING_STRATEGY, ADVANCED_CHUNK_SIZE, ADVANCED_CHUNK_OVERLAP, RNN_CHUNK_SIZE
from src.logger import log

def read_docx(path: str) -> str:
    """Reads text from a .docx file."""
    log.info(f"Reading DOCX file from: {path}")
    return "\n".join(p.text for p in docx.Document(path).paragraphs)

def read_pdf(path: str) -> str:
    """Reads text from a .pdf file."""
    log.info(f"Reading PDF file from: {path}")
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

def clean_text(text: str) -> str:
    """Removes extra whitespace from text."""
    return re.sub(r"\s+", " ", text).strip()

def _regex_sent_tokenize(text: str) -> List[str]:
    """Splits text into sentences using regex."""
    sentences = re.split(r'(?<=[.!?]) +', text)
    return [s.strip() for s in sentences if s.strip()]

def _rnn_style_chunking(text: str) -> List[str]:
    """
    Chunks text by grouping sentences together up to a max word limit.
    This is a fallback for very large documents.
    """
    log.info(f"Using RNN-style chunking with max_words={RNN_CHUNK_SIZE}")
    sentences = _regex_sent_tokenize(text)
    
    chunks, current_chunk_words = [], []
    current_word_count = 0
    
    for sentence in sentences:
        words = sentence.split()
        if current_word_count + len(words) > RNN_CHUNK_SIZE and current_chunk_words:
            chunks.append(' '.join(current_chunk_words))
            current_chunk_words = []
            current_word_count = 0
            
        current_chunk_words.extend(words)
        current_word_count += len(words)
        
    if current_chunk_words:
        chunks.append(' '.join(current_chunk_words))
        
    return chunks

def _advanced_semantic_chunking(text: str) -> List[str]:
    """
    Creates overlapping chunks based on sentences. This is inspired by RAG principles
    to ensure that context is not lost at chunk boundaries. The overlap helps the
    Reformer-style attention mechanism to have visibility into adjacent chunks.
    Quantization is conceptually applied by creating discrete, well-defined chunks.
    """
    log.info(f"Using advanced semantic chunking with size={ADVANCED_CHUNK_SIZE} and overlap={ADVANCED_CHUNK_OVERLAP} sentences.")
    sentences = _regex_sent_tokenize(text)
    
    if not sentences:
        return []

    chunks = []
    # The step is the non-overlapping part of the chunk
    step = ADVANCED_CHUNK_SIZE - ADVANCED_CHUNK_OVERLAP
    if step <= 0:
        log.warning("Chunk overlap is too large, defaulting to step of 1.")
        step = 1

    for i in range(0, len(sentences), step):
        chunk_sentences = sentences[i : i + ADVANCED_CHUNK_SIZE]
        chunk = " ".join(chunk_sentences)
        chunks.append(chunk)
        # Stop if the last chunk is already smaller than the chunk size
        if i + ADVANCED_CHUNK_SIZE >= len(sentences):
            break
            
    return chunks

def get_chunking_strategy() -> Callable[[str], List[str]]:
    """
    Returns the chunking function based on the strategy set in the config.
    """
    if CHUNKING_STRATEGY == "advanced":
        log.info("Selected 'advanced' chunking strategy.")
        return _advanced_semantic_chunking
    elif CHUNKING_STRATEGY == "rnn":
        log.info("Selected 'rnn' chunking strategy as a fallback.")
        return _rnn_style_chunking
    else:
        log.error(f"Invalid CHUNKING_STRATEGY '{CHUNKING_STRATEGY}'. Defaulting to advanced.")
        return _advanced_semantic_chunking