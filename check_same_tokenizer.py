import os
import shutil
import numpy as np
from transformers import AutoTokenizer
from tunix.generate import tokenizer_adapter

# Model ID to test (Gemma 2B is a good candidate)
MODEL_ID = "google/gemma-3-1b-it"

def main():
    print(f"Checking consistency between HuggingFace and SentencePiece loaders for {MODEL_ID}...")
    
    # 1. Load via Hugging Face (requires HF_TOKEN env var if gated)
    try:
        print("Loading HuggingFace tokenizer...")
        # We use the Tunix wrapper for HF
        tunix_hf_wrapper = tokenizer_adapter.Tokenizer(
            tokenizer_type='huggingface',
            tokenizer_path=MODEL_ID,
            add_bos=False,
            add_eos=False
        )
        # We also need the raw object to extract the model file
        raw_hf_tokenizer = tunix_hf_wrapper.tokenizer
    except Exception as e:
        print(f"Error loading HF tokenizer: {e}")
        print("Please ensure you have access to the model and HF_TOKEN is set if required.")
        return

    # 2. Extract tokenizer.model for SentencePiece loading
    # We save the pretrained tokenizer to a temp dir to get the .model file
    temp_dir = "temp_tokenizer_check_dir"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    print(f"Saving HF tokenizer to {temp_dir} to extract .model file...")
    raw_hf_tokenizer.save_pretrained(temp_dir)
    
    sp_model_path = os.path.join(temp_dir, "tokenizer.model")
    if not os.path.exists(sp_model_path):
        print("Error: tokenizer.model not found in saved HF files. Comparison requires the SP model file.")
        return

    # 3. Load via SentencePiece using Tunix wrapper
    try:
        print(f"Loading SentencePiece tokenizer from {sp_model_path}...")
        tunix_sp_wrapper = tokenizer_adapter.Tokenizer(
            tokenizer_type='sentencepiece',
            tokenizer_path=sp_model_path,
            add_bos=False,
            add_eos=False
        )
    except Exception as e:
        print(f"Error loading SP tokenizer: {e}")
        return

    # 4. Compare Outputs
    test_cases = [
        "Hello world",
        "Testing 123",
        "   Leading spaces",
        "Trailing spaces   ",
        "Special characters: — • 123",
        "Gemma is a family of lightweight, state-of-the-art open models.",
        "<start_of_turn>model\nHello<end_of_turn>", # Gemma specific tokens
    ]

    print("\n--- Starting Comparison ---")
    all_match = True
    
    for i, text in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {text!r}")
        
        # Encode
        # Note: Tunix wrapper 'encode' for HF calls self._tokenizer.encode(text, **kwargs)
        # For SP it calls self._tokenizer.EncodeAsIds(text, **kwargs)
        
        # We pass add_special_tokens=False to HF to try and match raw SP output 
        # which typically doesn't add BOS/EOS unless configured/requested separately.
        hf_ids = tunix_hf_wrapper.encode(text, add_special_tokens=False)
        sp_ids = tunix_sp_wrapper.encode(text)
        
        print(f"  HF IDs: {hf_ids}")
        print(f"  SP IDs: {sp_ids}")
        
        if hf_ids == sp_ids:
            print("  Result: MATCH")
        else:
            print("  Result: MISMATCH")
            all_match = False

    # 5. Compare Decoding
    print("\n--- Decoding Check (using last case) ---")
    hf_decode = tunix_hf_wrapper.decode(hf_ids)
    sp_decode = tunix_sp_wrapper.decode(sp_ids)
    print(f"  HF Decode: {hf_decode!r}")
    print(f"  SP Decode: {sp_decode!r}")
    
    if hf_decode != sp_decode:
        print("  Decoding Result: MISMATCH")
        all_match = False
    else:
        print("  Decoding Result: MATCH")

    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    if all_match:
        print("\nCONCLUSION: The tokenizers produced IDENTICAL output for all test cases.")
    else:
        print("\nCONCLUSION: The tokenizers produced DIFFERENT output.")
        print("Note: Small differences might handle whitespace or special tokens differently by default.")

if __name__ == "__main__":
    main()
