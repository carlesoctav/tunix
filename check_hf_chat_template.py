import os
from transformers import AutoTokenizer

MODEL_ID = "google/gemma-3-1b-it"

def main():
    print(f"Loading HF tokenizer for {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "model", "content": "I am doing well, thank you!"},
    ]
    
    print("\n--- Test 1: tokenize=True (Default) ---")
    output_1 = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    print(f"Output Type: {type(output_1)}")
    if isinstance(output_1, list):
        print(f"Output sample: {output_1[:5]}... (List of IDs)")
    elif isinstance(output_1, dict):
        print(f"Output keys: {output_1.keys()}")
    else:
        print(f"Output: {output_1}")

    print("\n--- Test 2: tokenize=True, return_dict=True ---")
    try:
        output_2 = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=True)
        print(f"Output Type: {type(output_2)}")
        if isinstance(output_2, dict):
            print(f"Output keys: {output_2.keys()}")
            if 'attention_mask' in output_2:
                print("SUCCESS: 'attention_mask' is present.")
            else:
                print("FAILURE: 'attention_mask' is NOT present.")
        else:
            print(f"Output is not a dict: {output_2}")
            
    except Exception as e:
        print(f"Error with return_dict=True: {e}")

if __name__ == "__main__":
    main()
