import os
from transformers import AutoTokenizer

MODEL_ID = "google/gemma-3-1b-it"

def main():
    print(f"Checking pad token for {MODEL_ID}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        
        print(f"\n--- Tokenizer Config ---")
        print(f"Pad Token: {tokenizer.pad_token}")
        print(f"Pad Token ID: {tokenizer.pad_token_id}")
        print(f"EOS Token: {tokenizer.eos_token}")
        print(f"EOS Token ID: {tokenizer.eos_token_id}")
        
        if tokenizer.pad_token_id == 0:
            print("\nResult: Yes, the pad token ID is 0.")
        else:
            print(f"\nResult: No, the pad token ID is {tokenizer.pad_token_id}.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

