import time
import memory
import intent_classifier
import response_policy

def test():
    memory._load_profile()
    memory_manager = __import__("memory_manager")
    memory_manager.load_memory()
    
    t0 = time.monotonic()
    
    # 1. Intent Router
    intent = intent_classifier.classify_intent("who am i")
    
    # 2. Policy Route
    response = None
    if intent in ("MEMORY_SUMMARY", "MEMORY_QUERY"):
        response, _ = response_policy.apply_policy(intent, "who am i")
        
    t1 = time.monotonic()
    
    print(f"Intent: {intent}")
    print(f"Response: {response}")
    print(f"Time: {(t1 - t0) * 1000:.2f} ms")

if __name__ == "__main__":
    test()
