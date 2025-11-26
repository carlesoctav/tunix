import os
from transformers import AutoTokenizer

MODEL_ID = "google/gemma-3-1b-it"

def main():
    print(f"Checking padding side for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    print(f"Tokenizer padding_side: {tokenizer.padding_side}")
    
    if tokenizer.padding_side == "left":
        print("Reason: This tokenizer is configured for left-padding by default.")
    else:
        print("Reason: Unknown.")

if __name__ == "__main__":
    main()
