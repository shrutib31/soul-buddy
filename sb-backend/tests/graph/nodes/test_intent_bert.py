import time
from transformers import pipeline

print("Loading DistilBERT Intent Classifier...")
classifier = pipeline(
    "zero-shot-classification", 
    model="typeform/distilbert-base-uncased-mnli",
    device=-1 # Uses CPU
)

LABEL_MAP = {
    "a polite greeting or hello": "greeting",
    "venting, complaining, or expressing negative emotions": "venting",
    "asking a factual question for information": "seek_information",
    "seeking a deep explanation or understanding": "seek_understanding",
    "asking for a solution to a specific problem": "open_to_solution",
    "requesting to use a tool, calculator, or resource": "try_tool",
    "seeking emotional support, comfort, or empathy": "seek_support"
}

def test_intent(message):
    candidate_labels = list(LABEL_MAP.keys())
    
    start_time = time.time()
    
    result = classifier(message, candidate_labels, multi_label=False)
    
    end_time = time.time()
    
    top_label_phrase = result['labels'][0]
    confidence = result['scores'][0]
    
    intent_slug = LABEL_MAP.get(top_label_phrase)
    
    final_intent = intent_slug if confidence >= 0.25 else "unclear"

    print(f"\n--- Message: \"{message}\" ---")
    print(f"\nFinal Slug: {final_intent} (Score: {confidence:.2f})")
    print(f"Latency:    {(end_time - start_time) * 1000:.2f}ms")

# Runing Tests
if __name__ == "__main__":
    test_cases = [
        "Hi, how are you?",
        "I am a bit scared about my semester exam results.",
        "I am just a bit bored",
        "What is 2+2?",
        "Can you please help me? I feel alone and lonely after my wife died.",
        "I committed a big mistake in my company. How can I solve the issue?",
        "I feel bored right now.",
        "What is ChatGPT?",
        "I am scared, someone might be stalking me. What do I do?"
    ]
    
    print(f"\nTesting {len(test_cases)} intents...")
    for text in test_cases:
        test_intent(text)