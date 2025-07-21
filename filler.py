from fastapi import FastAPI, Request, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict
import faiss
import pickle
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from dotenv import load_dotenv
import docx
import fitz  # PyMuPDF for PDFs
import re
import uuid
import json
from io import BytesIO
from sympy import content
from concurrent.futures import ThreadPoolExecutor
import asyncio
from collections import defaultdict
import hashlib
# Load vector store and metadata
vector_store_path = "vector_store"
index = faiss.read_index(os.path.join(vector_store_path, "index.faiss"))
with open(os.path.join(vector_store_path, "metadata.pkl"), "rb") as f:
    metadata_df = pickle.load(f)

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

load_dotenv()
# Load Mistral API credentials
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"

# Preprocess persona keywords
persona_keywords = {}
for _, row in metadata_df.iterrows():
    persona = str(row.get("persona", "")).lower()
    keywords = str(row.get("Chunk", "")).lower().split()
    if persona:
        if persona not in persona_keywords:
            persona_keywords[persona] = set()
        persona_keywords[persona].update(keywords)
# FastAPI App
app = FastAPI()


class QueryRequest(BaseModel):
    query: str

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def extract_text_from_docx(file) -> str:
    content =  file.read()
    file.seek(0)
    doc = docx.Document(BytesIO(content))
    return clean_text("\n".join([para.text for para in doc.paragraphs]))

def extract_text_from_pdf(file) -> str:
    content = file.read()
    file.seek(0)
    pdf = fitz.open(stream=content, filetype="pdf")
    return clean_text("\n".join([page.get_text() for page in pdf]))

executor = ThreadPoolExecutor(max_workers=8)

def rag_reformer_quantized_chunking(text: str, chunk_size: int = 512, overlap: int = 128) -> List[str]:
    words = text.split()
    step = chunk_size - overlap
    return [f"[RAG-RF-QZ-{i}] " + ' '.join(words[i:i+chunk_size]) for i in range(0, len(words), step)]

def hash_entry(persona, pain_point, goal_action):
    return hashlib.md5(f"{persona}|{pain_point}|{goal_action}".encode()).hexdigest()

def extract_structured_entities(chunks, mistral_fn):
    seen_hashes = set()
    grouped_results = defaultdict(lambda: {"pain_points": set(), "goal_actions": set()})
    raw_results = []

    for i, chunk in enumerate(chunks):
        prompt = build_extraction_prompt(chunk)
        try:
            response = mistral_fn(prompt)
            result = parse_llm_response(response)

            persona = result.get("Persona", "").strip()
            pain_point = result.get("Pain_point", "").strip()
            goal_action = result.get("Goal_action", "").strip()

            if not persona or not pain_point or not goal_action:
                continue  # Skip incomplete entries

            entry_hash = hash_entry(persona, pain_point, goal_action)
            if entry_hash not in seen_hashes:
                seen_hashes.add(entry_hash)
                grouped_results[persona]["pain_points"].add(pain_point)
                grouped_results[persona]["goal_actions"].add(goal_action)
                raw_results.append({
                    "chunk_id": f"chunk-{i}",
                    "persona": persona,
                    "pain_point": pain_point,
                    "goal_action": goal_action
                })
        except Exception as e:
            print(f"⚠️ Skipping chunk-{i} due to error: {e}")

    # Clean up results: convert sets to sorted lists
    grouped_json = [
        {
            "persona": persona,
            "pain_points": sorted(list(data["pain_points"])),
            "goal_actions": sorted(list(data["goal_actions"]))
        }
        for persona, data in grouped_results.items()
    ]

    return raw_results, grouped_json


def build_extraction_prompt(chunk: str) -> str:
    return f"""
You are a business assistant. Read the document context and extract relevant **Persona**, **Pain_point**, **Goal_action**, and **Value_message**.

Chunk:
\"\"\"{chunk}\"\"\"

Extract as follows:
Persona: (who is affected or mentioned)
Pain_point: (specific business challenge they are facing)
Goal_action: (what they aim to fix or do about it)

Respond in this format only.
"""


def retrieve_top_chunks(query: str, top_k: int = 5) -> List[str]:
    query_embedding = embedding_model.encode([query])
    D, I = index.search(query_embedding, top_k)
    return metadata_df.iloc[I[0]]["Chunk"].tolist()


def build_prompt(query: str, context_chunks: List[str]) -> str:
    joined_context = "\n\n".join(context_chunks)
    return f"""You are an expert assistant. Respond to the following query using ONLY the provided document context.

Query: "{query}"

Context:
{joined_context}

Extract and return the following:
Persona:
Pain_point:
Goal_action:
Value_message (Avoid bullet points. Write the value message as a concise, actionable paragraph in natural language and generate a concise paragraph, up to 5 lines, summarizing the business actions the persona should take to resolve the pain point):

Only use personas, pain points, and goals if they are present or implied in the context. Do NOT make up information.
"""


def call_mistral(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "mistral-medium",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "max_tokens": 512
    }
    response = requests.post(MISTRAL_ENDPOINT, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def infer_persona(query: str, context_chunks: List[str]) -> str:
    persona_scores = {}
    context_text = " ".join(context_chunks).lower()
    for persona, keywords in persona_keywords.items():
        count = sum(word in context_text for word in keywords)
        persona_scores[persona] = count
    if not persona_scores:
        return None
    best_persona = max(persona_scores, key=persona_scores.get)
    return best_persona.upper() if persona_scores[best_persona] > 0 else None


def parse_llm_response(response_text: str) -> Dict[str, str]:
    fields = {"Persona": "", "Pain_point": "", "Goal_action": "", "Value_message": ""}
    key_map = {k.lower(): k for k in fields.keys()}
    current_key = None
    buffer = []

    for line in response_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for key in key_map:
            if line.lower().startswith(key + ":"):
                if current_key:
                    fields[key_map[current_key]] = " ".join(buffer).strip()
                    buffer = []
                current_key = key
                buffer.append(line.split(":", 1)[1].strip())
                break
        else:
            if current_key:
                buffer.append(line)
    if current_key:
        fields[key_map[current_key]] = " ".join(buffer).strip()
    return fields


@app.post("/Office-GPt")
async def generate_insight(request: QueryRequest):
    query = request.query
    chunks = retrieve_top_chunks(query)
    if not chunks:
        return {"error": "No relevant chunks found."}

    prompt = build_prompt(query, chunks)
    try:
        response_text = call_mistral(prompt)
        parsed = parse_llm_response(response_text)
        inferred = infer_persona(query, chunks)
        if inferred and parsed["Persona"].lower() != inferred.lower():
            parsed["Persona"] = inferred

        return {
            "query": query,
            "persona": parsed["Persona"],
            "pain_point": parsed["Pain_point"],
            "goal_action": parsed["Goal_action"],
            "value_message": parsed["Value_message"]
        }
    except Exception as e:
        return {"error": str(e)}

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    try:
        filename = file.filename.lower()
        if filename.endswith(".docx"):
            raw_text = extract_text_from_docx(file.file)
        elif filename.endswith(".pdf"):
            raw_text = extract_text_from_pdf(file.file)
        else:
            return {"error": "Unsupported file type. Use .docx or .pdf"}

        chunks = rag_reformer_quantized_chunking(raw_text)

        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(executor, extract_structured_entities, i, chunk) for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)

        results = [r for r in results if r]  # Filter out None

        file_id = str(uuid.uuid4())[:8]
        output_path = f"uploads/persona_insights_{file_id}.json"
        os.makedirs("uploads", exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return {
            "message": "Extraction complete.",
            "file_saved_as": output_path,
            "entries_extracted": len(results),
            "preview": results[:3]
        }

    except Exception as e:
        return {"error": str(e)}