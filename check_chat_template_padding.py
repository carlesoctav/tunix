import os
from transformers import AutoTokenizer

MODEL_ID = "google/gemma-3-1b-it"

def main():
    print(f"Loading HF tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # Short message
    messages = [
        {"role": "user", "content": "Hi"},
    ]
    
    MAX_LENGTH = 20
    
    print(f"\n--- Test: max_length={MAX_LENGTH}, padding='max_length' ---")
    try:
        # Note: padding arguments are typically passed to the tokenizer's encoding method.
        # apply_chat_template usually returns a list of ints if tokenize=True.
        output = tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=False, 
            max_length=MAX_LENGTH, 
            padding='max_length'
        )
        
        print(f"Output length: {len(output)}")
        print(f"Output IDs: {output}")
        
        if len(output) == MAX_LENGTH:
            print(f"\nResult: Output WAS padded to max_length ({len(output)}).")
            # Check if padding tokens (0) are present
            pad_id = tokenizer.pad_token_id
            if pad_id in output:
                 print(f"Verified: Pad token ID {pad_id} found in output.")
            else:
                 print(f"Warning: Pad token ID {pad_id} NOT found (maybe filled with something else?).")
        else:
            print(f"\nResult: Output was NOT padded to max_length (Length: {len(output)}).")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

