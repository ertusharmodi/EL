import json
import logger
import math
from typing import Dict, List, Tuple, Any

import ollama

_EMBED_MODEL = "nomic-embed-text"
_cached_embeddings: Dict[str, List[float]] = {}
_cached_memory_state: str = ""

def _get_embedding(text: str) -> List[float]:
    try:
        res = ollama.embeddings(model=_EMBED_MODEL, prompt=text)
        return res.get("embedding", [])
    except Exception as e:
        logger.warning(f"  ⚠️  [Memory Retriever] Embedding failed: {e}")
        return []

def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if not vec1 or not vec2:
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0: 
        return 0.0
    return dot / (norm1 * norm2)

def _flatten_memory(mem: dict) -> List[Tuple[str, str, Any]]:
    # Returns [(category, key, value), ...]
    flat = []
    for cat, items in mem.items():
        if not items:
            continue
        if isinstance(items, dict):
            for k, v in items.items():
                flat.append((cat, k, v))
    return flat

def retrieve_relevant(query: str, mem: dict, top_k: int = 3) -> dict:
    global _cached_memory_state, _cached_embeddings
    
    flat_mem = _flatten_memory(mem)
    if not flat_mem:
        return {}
        
    # Check if memory mutated
    mem_str = json.dumps(mem, sort_keys=True)
    if mem_str != _cached_memory_state:
        # Re-embed all facts
        _cached_embeddings.clear()
        for cat, k, v in flat_mem:
            text_rep = f"{k} = {v}"
            _cached_embeddings[text_rep] = _get_embedding(text_rep)
        _cached_memory_state = mem_str
        
    query_emb = _get_embedding(query)
    if not query_emb:
        # Fallback to returning everything if embeddings fail (e.g., model missing)
        return mem
        
    scored = []
    for cat, k, v in flat_mem:
        text_rep = f"{k} = {v}"
        emb = _cached_embeddings.get(text_rep, [])
        score = _cosine_similarity(query_emb, emb)
        scored.append((score, cat, k, v))
        
    scored.sort(key=lambda x: x[0], reverse=True)
    
    results = {}
    valid_count = 0
    for score, cat, k, v in scored:
        if valid_count >= top_k:
            break
        # nomic-embed-text cosine distances usually hover around >0.5 for related concepts
        if score > 0.45:
            if cat not in results:
                results[cat] = {}
            results[cat][k] = v
            valid_count += 1
            
    logger.debug(f"  🧠 Retrieved {valid_count} relevant memories")
    return results
