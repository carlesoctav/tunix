import os
import sys
# Add the project root to the path so we can import from research
sys.path.append(os.getcwd())

from research.datasets_utils import TrainMixDataConfig, HFSource
from datasets import load_dataset
from transformers import AutoTokenizer

def main():
    print("--- Checking Dataset Columns ---")
    # Inspect the dataset to find the correct column
    try:
        ds = load_dataset("allenai/Dolci-Think-SFT", split="train", streaming=True)
        first_item = next(iter(ds))
        print(f"Dataset keys: {first_item.keys()}")
        
        if "messages" in first_item:
            chat_column = "messages"
        elif "conversations" in first_item:
            chat_column = "conversations"
        else:
            # Fallback or pick the first list/dict column
            chat_column = next(k for k, v in first_item.items() if isinstance(v, list))
        print(f"Selected chat column: {chat_column}")
            
    except Exception as e:
        print(f"Error loading dataset inspection: {e}")
        return

    print("\n--- Setting up Config ---")
    
    # Use a known existing tokenizer for the test
    tokenizer_id = "google/gemma-3-1b-it" 
    
    config = TrainMixDataConfig(
        sources=[
            HFSource(
                path="allenai/Dolci-Think-SFT", 
                streaming=True,
            )
        ],
        tokenizer_path=tokenizer_id,
        chat_column=chat_column,
        max_length=2048,
        batch_size=None # No batching for simple iteration inspection
    )

    print("--- Creating Pipeline ---")
    try:
        dataset = config.make()
    except Exception as e:
        print(f"Error creating pipeline: {e}")
        return

    print("--- Iterating & Checking Regex ---")
    print(f"Looking for '<reasoning>' tags replacing '<think>'...\n")

    try:
        # Use islice or just a loop
        count = 0
        for i, batch in enumerate(dataset):
            if count >= 10:
                break
            
            print(batch["input_ids"])
            print(type(batch["input_ids"]))
            print(batch["input_ids"].shape)
            text = batch.get("text", "")
            
            # Check if our regex worked
            has_reasoning = "<reasoning>" in text
            has_think = "<think>" in text
            has_answer = "<answer>" in text
            
            print(f"Example {i+1}:")
            print(f"  Has <reasoning>: {has_reasoning}")
            print(f"  Has <answer>:    {has_answer}")
            print(f"  Has <think>:     {has_think}")
            
            # Print a snippet if we found the tags
            if has_reasoning:
                # Find the reasoning block to display
                start = text.find("<reasoning>")
                end = text.find("</reasoning>")
                snippet = text[start:end+12] # + length of closing tag
                print(f"  Reasoning Snippet: {snippet[:100]}...")

            if has_answer:
                start = text.find("<answer>")
                end = text.find("</answer>")
                # If </answer> is not found (maybe end of string), just take some chars
                if end == -1:
                    end = start + 100
                snippet = text[start:end+9]
                print(f"  Answer Snippet:    {snippet[:100]}...")

            if has_think:
                print("  [WARNING] Found <think> tag! Regex might have failed or pattern mismatch.")
                start = text.find("<think>")
                print(f"  Snippet: {text[start:start+100]}...")
            else:
                print("  No tags found in this example.")
                
            print("-" * 40)
            count += 1
            
    except Exception as e:
        print(f"Error during iteration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
