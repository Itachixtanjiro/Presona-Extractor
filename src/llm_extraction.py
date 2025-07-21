import requests
import json
import time
from typing import Dict, List
from src.config import MISTRAL_API_KEY, MISTRAL_ENDPOINT, MISTRAL_MODEL
from src.logger import log

def _get_rag_prompt(original_chunk: str, retrieved_context: List[str]) -> str:
    """
    Generates a prompt for the RAG workflow, combining the original chunk
    with context retrieved from the vector store.
    """
    context_str = "\n\n---\n\n".join(retrieved_context)
    
    return f"""
You are a highly specialized analyst focused on identifying business professionals from text. Your task is to extract key information based on an ORIGINAL CHUNK of text, using the provided RETRIEVED CONTEXT for additional information and clarification.

The final output must be a single JSON object with three keys: "persona", "pain_point", and "goal_action".

**CRITICAL INSTRUCTIONS for 'persona' extraction:**
1. The 'persona' MUST be a specific, professional job title.
2. AVOID generic descriptions, roles, or groups.

**Examples:**
- GOOD persona: "Revenue Manager", "Chief Executive Officer", "DevOps Engineer", "Sales Director"
- BAD persona: "a busy reader", "someone who needs software", "potential customers"

---
ORIGINAL CHUNK:
"{original_chunk}"
---
RETRIEVED CONTEXT:
"{context_str}"
---

Based on the ORIGINAL CHUNK and clarified by the RETRIEVED CONTEXT, extract the specific professional persona, their primary pain point, and the goal or action to resolve it.

Reply with only the JSON object.
"""

def _call_mistral(prompt: str) -> str:
    """
    Makes a single API call to the Mistral service with exponential backoff.
    """
    if not MISTRAL_API_KEY:
        raise ValueError("MISTRAL_API_KEY is not set.")
        
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2, # Lower temperature for more deterministic output
        "max_tokens": 512,
        "response_format": {"type": "json_object"}
    }
    
    max_retries = 5
    backoff_factor = 1.5
    
    for attempt in range(max_retries):
        try:
            log.debug(f"Calling Mistral API (Attempt {attempt + 1}/{max_retries})...")
            resp = requests.post(MISTRAL_ENDPOINT, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
            
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            error_details = e.response.text
            log.error(f"HTTP Error {status_code} from Mistral API: {error_details}")
            
            if status_code == 429:
                wait_time = backoff_factor * (2 ** attempt)
                log.warning(f"Rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                log.error("Non-retriable HTTP error. Aborting.")
                raise e
        except requests.RequestException as e:
            log.error(f"Request failed due to a network issue: {e}")
            wait_time = backoff_factor * (2 ** attempt)
            log.warning(f"Retrying in {wait_time:.2f} seconds...")
            time.sleep(wait_time)

    raise requests.RequestException(f"API call failed after {max_retries} retries.")

def extract_insight_with_rag(original_chunk: str, retrieved_context: List[str]) -> Dict:
    """
    Extracts a structured insight using the RAG workflow.
    """
    prompt = _get_rag_prompt(original_chunk, retrieved_context)
    try:
        response_text = _call_mistral(prompt)
        data = json.loads(response_text)
        
        if all(k in data for k in ("persona", "pain_point", "goal_action")):
            log.debug(f"Successfully extracted RAG insight for chunk: {original_chunk[:50]}...")
            return data
        else:
            log.warning(f"RAG LLM response missing required keys for chunk: {original_chunk[:50]}...")
            return {}
            
    except requests.RequestException as e:
        log.error(f"RAG Mistral API call failed after multiple retries: {e}")
    except json.JSONDecodeError as e:
        log.error(f"RAG JSON parsing failed for response: '{response_text[:100]}...'. Error: {e}")
    except Exception as e:
        log.error(f"An unexpected error occurred during RAG extraction: {e}")
        
    return {}