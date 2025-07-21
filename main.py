import os
import uuid
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, UploadFile, File, HTTPException

# Import modular components
from src.config import UPLOAD_DIR
from src.logger import log
from src.preprocessing import read_docx, read_pdf, clean_text, get_chunking_strategy
from src.vectorization import generate_embeddings
from src.vector_store import (
    load_or_initialize_vector_store, 
    add_to_vector_store, 
    save_vector_store,
    search_vector_store
)
from src.llm_extraction import extract_insight_with_rag
from src.helpers import aggregate_insights, save_json

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Global State ---
log.info("Loading vector store on application startup...")
index, metadata = load_or_initialize_vector_store()
log.info("Vector store loaded.")


@app.post("/extract/")
async def upload_and_extract(file: UploadFile = File(...)):
    """
    FastAPI endpoint to upload a document, extract insights using a RAG pipeline,
    and save the results.
    """
    # Declare upfront that we will be modifying the global index and metadata
    global index, metadata
    
    file_id = uuid.uuid4().hex[:8]
    log.info(f"Starting RAG extraction for file_id: {file_id}")

    # 1. Save and preprocess text
    filename = file.filename.lower()
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}_{filename}")
    try:
        with open(save_path, "wb") as f:
            f.write(await file.read())
        
        if filename.endswith(".docx"):
            raw_text = read_docx(save_path)
        elif filename.endswith(".pdf"):
            raw_text = read_pdf(save_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")
        
        cleaned_text = clean_text(raw_text)
        chunking_function = get_chunking_strategy()
        chunks = chunking_function(cleaned_text)
    except Exception as e:
        log.error(f"Failed during file processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to read or process the document.")

    if not chunks:
        log.warning("No chunks were generated from the document.")
        return {"message": "Extraction complete. No content found in document."}

    # 2. Generate embeddings for the new chunks
    new_embeddings = generate_embeddings(chunks)

    # 3. RAG-based Extraction (Retrieve -> Augment -> Generate)
    log.info(f"Starting RAG extraction for {len(chunks)} chunks.")
    raw_insights = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_chunk = {}
        for i, chunk in enumerate(chunks):
            query_embedding = np.array([new_embeddings[i]])
            retrieved_context = search_vector_store(index, metadata, query_embedding)
            
            future = executor.submit(extract_insight_with_rag, chunk, retrieved_context)
            future_to_chunk[future] = chunk
            
        for future in as_completed(future_to_chunk):
            result = future.result()
            if result:
                raw_insights.append(result)
    log.info(f"Extracted {len(raw_insights)} raw insights using RAG.")

    # 4. Add new chunks to the vector store *after* the RAG process
    index = add_to_vector_store(index, metadata, new_embeddings, chunks)
    save_vector_store(index, metadata)

    # 5. Aggregate and save final JSON
    final_data = aggregate_insights(raw_insights)
    output_path = save_json(final_data, file_id)

    log.info(f"RAG extraction complete for file_id: {file_id}")
    return {
        "message": "RAG extraction complete.",
        "file_saved_as": output_path,
        "entries_extracted": len(final_data),
        "insights_preview": final_data[:5]
    }

if __name__ == "__main__":
    import uvicorn
    log.info("Starting FastAPI server with Uvicorn.")
    uvicorn.run(app, host="0.0.0.0", port=8000)