import json
import os
from collections import defaultdict
from typing import List, Dict
from src.config import UPLOAD_DIR
from src.logger import log

def aggregate_insights(insights: List[Dict]) -> List[Dict]:
    """
    Groups extracted insights by persona and deduplicates pain points and goals.
    """
    if not insights:
        return []
        
    log.info(f"Aggregating {len(insights)} raw insights.")
    
    # Use a set to store tuples of (pain_point, goal_action) for automatic deduplication
    grouped = defaultdict(set)
    
    for ins in insights:
        persona = ins.get("persona", "Unknown").strip()
        pain_point = ins.get("pain_point", "").strip()
        goal_action = ins.get("goal_action", "").strip()
        
        if persona and pain_point and goal_action:
            grouped[persona].add((pain_point, goal_action))
            
    # Reformat the grouped data into the final desired structure
    final_output = []
    for persona, pairs in grouped.items():
        persona_insights = [
            {"pain_point": pp, "goal_action": ga} for pp, ga in sorted(list(pairs))
        ]
        final_output.append({"persona": persona, "insights": persona_insights})
        
    log.info(f"Aggregated insights into {len(final_output)} unique personas.")
    return final_output

def save_json(data: List[Dict], file_id: str) -> str:
    """Saves the final data to a JSON file."""
    out_path = os.path.join(UPLOAD_DIR, f"persona_insights_{file_id}.json")
    log.info(f"Saving final JSON output to: {out_path}")
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        
    return out_path