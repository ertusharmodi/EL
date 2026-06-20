"""
Hybrid memory extraction — regex first, then LLM.

Pipeline:
  1. Regex extraction (fast, deterministic, confidence=1.0)
  2. LLM extraction (natural language, per-item confidence)
  3. Merge & deduplicate (regex wins on key conflicts)
  4. Save items with confidence >= MEMORY_LLM_MIN_CONFIDENCE
"""

from typing import Any, Dict, List, Union

import config
import llm_extractor
import memory_manager
import regex_extractor


def extract_memories_regex(user_message: str) -> List[Dict[str, Any]]:
    """Run regex-only extraction (re-export for tests)."""
    return regex_extractor.extract_memories_regex(user_message)


def extract_memories(user_message: str) -> List[Dict[str, Any]]:
    """
    Run the full hybrid extraction pipeline without saving.
    Returns merged memory items with confidence scores.
    """
    regex_mems = regex_extractor.extract_memories_regex(user_message)
    llm_mems = llm_extractor.extract_memories_llm(user_message, already_extracted=regex_mems)
    return _merge_extractions(regex_mems, llm_mems)


def _merge_extractions(
    regex_mems: List[Dict[str, Any]],
    llm_mems: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge regex and LLM results.
    If duplicate keys exist:
      - Compare values
      - Identical values: keep regex version
      - Additional info: prefer the more complete value
      - List-like preferences: merge values instead of replacing
    """
    merged: Dict[tuple, Dict[str, Any]] = {}

    for mem in regex_mems:
        slot = (mem["category"], mem["key"])
        if slot in merged:
            old_val = merged[slot]["value"]
            new_val = mem["value"]
            final_val = memory_manager.merge_values(mem["category"], mem["key"], old_val, new_val)
            if final_val != old_val:
                print("  🧠 Merge Decision")
                print(f"  Old Value: {old_val}")
                print(f"  New Value: {new_val}")
                print(f"  Final Value: {final_val}")
            merged[slot]["value"] = final_val
        else:
            merged[slot] = mem

    for mem in llm_mems:
        slot = (mem["category"], mem["key"])
        if slot not in merged:
            merged[slot] = mem
        else:
            old_mem = merged[slot]
            old_val = old_mem["value"]
            new_val = mem["value"]
            
            final_val = memory_manager.merge_values(mem["category"], mem["key"], old_val, new_val)
            
            if final_val != old_val:
                print("  🧠 Merge Decision")
                print(f"  Old Value: {old_val}")
                print(f"  New Value: {new_val}")
                print(f"  Final Value: {final_val}")
                
            merged[slot]["value"] = final_val
            
            if final_val == new_val and new_val != old_val:
                merged[slot]["source"] = "llm"
                merged[slot]["confidence"] = mem.get("confidence", old_mem.get("confidence"))
            elif final_val != old_val:
                merged[slot]["source"] = "merged"

    return list(merged.values())


def _format_value(value: Union[str, List[str]]) -> str:
    if isinstance(value, list):
        return ", ".join(value)
    return str(value)


def _passes_confidence(mem: Dict[str, Any]) -> bool:
    return float(mem.get("confidence", 0.0)) >= config.MEMORY_LLM_MIN_CONFIDENCE


def run_extraction_and_save(
    user_message: str,
    stt_confidence: float = 1.0,
) -> None:
    """
    Hybrid extraction pipeline: regex → LLM → merge → save.

    Skips all saves when STT confidence is below MEMORY_STT_MIN_CONFIDENCE.
    Only saves memories whose extraction confidence >= MEMORY_LLM_MIN_CONFIDENCE.
    """
    print(f"🧠 DEBUG: [1] User message received: '{user_message}'")
    print(f"🧠 DEBUG: [2] STT confidence: {stt_confidence}")

    if stt_confidence < config.MEMORY_STT_MIN_CONFIDENCE:
        print("  ⚠️ Memory save skipped due to low STT confidence")
        return

    regex_mems = regex_extractor.extract_memories_regex(user_message)
    print(f"🧠 DEBUG: [3] Regex extraction result: {regex_mems}")

    llm_mems = llm_extractor.extract_memories_llm(user_message, already_extracted=regex_mems)
    print(f"🧠 DEBUG: [4] LLM extraction result: {llm_mems}")

    memories = _merge_extractions(regex_mems, llm_mems)
    print(f"🧠 DEBUG: [5] Merged memories: {memories}")

    if not memories:
        return

    for mem in memories:
        cat = mem["category"]
        key = mem["key"]
        val = mem["value"]
        confidence = float(mem.get("confidence", 0.0))
        source = mem.get("source", "unknown")
        display = _format_value(val)

        print(f"  🧠  Extracted Memory\n  {cat}.{key} = {display}")
        print(f"  🧠  Confidence: {confidence:.0%} ({source})")

        print(f"🧠 DEBUG: [6] Confidence filtering: {confidence} >= {config.MEMORY_LLM_MIN_CONFIDENCE}")
        if not _passes_confidence(mem):
            print(f"  ⚠️   Skipped — confidence below {config.MEMORY_LLM_MIN_CONFIDENCE:.0%}")
            continue

        print(f"🧠 DEBUG: [7] Memory save operation started for {cat}.{key} = {val}")
        action = memory_manager.apply_memory(cat, key, val)
        print(f"🧠 DEBUG: [7] Memory save operation returned action: {action}")

        if action == "saved":
            print(f"  🧠  Saved Memory\n  {cat}.{key} = {display}")
        elif action == "updated":
            print(f"  🧠  Updated Memory\n  {cat}.{key} = {display}")
        elif action == "unchanged":
            print(f"  🧠  Unchanged Memory (Already known)\n  {cat}.{key} = {display}")
