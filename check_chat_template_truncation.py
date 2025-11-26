import os
from transformers import AutoTokenizer

MODEL_ID = "google/gemma-3-1b-it"

def main():
    print(f"Loading HF tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Create a long message
    long_content = "repetition " * 50 # 50 words, likely > 10 tokens
    messages = [
        {"role": "user", "content": long_content},
    ]
    
    MAX_LENGTH = 10
    
    print(f"\n--- Test: max_length={MAX_LENGTH}, truncation=False ---")
    # Note: apply_chat_template passes kwargs to tokenizer() or encode() if tokenize=True
    try:
        output = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=False, 
            max_length=MAX_LENGTH, 
            truncation=False
        )
        
        print(f"Output length: {len(output)}")
        print(f"Output IDs: {output}")
        
        if len(output) > MAX_LENGTH:
            print(f"\nResult: Output exceeded max_length ({len(output)} > {MAX_LENGTH}). Truncation did NOT happen.")
        else:
            print(f"\nResult: Output was truncated to {len(output)}.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

