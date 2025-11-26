import re

def process_text(text):
    # Pattern Explanation:
    # <think>       : Match literal "<think>"
    # (.*?)         : Group 1 (Reasoning). Non-greedy match of content inside.
    # </think>      : Match literal "</think>"
    # \s*           : Match optional whitespace/newlines after the tag.
    # (.*?)         : Group 2 (Answer). Non-greedy match of content after.
    # (?=<end_of_turn>) : Positive Lookahead. Stop matching when "<end_of_turn>" is seen ahead.
    
    pattern = r"<think>(.*?)</think>\s*(.*?)(?=<end_of_turn>)"
    
    # Replacement:
    # Wrap Group 1 in <reasoning> tags
    # Wrap Group 2 in <answer> tags
    replacement = r"<reasoning>\1</reasoning><answer>\2</answer>"
    
    # re.DOTALL is crucial so that '.' matches newlines (reasoning can be multi-line)
    return re.sub(pattern, replacement, text, flags=re.DOTALL)

# --- Test Case ---
input_text = """[BOS]<start_of_turn>user
Who are you?<end_of_turn>
<start_of_turn>model
<think>this user is asking who are you im a llm trained by google so i guess il wil tell that</think>im llm trained by google!<end_of_turn>
<start_of_turn>user
What is 2+2?<end_of_turn>
<start_of_turn>model
2+2=4.<end_of_turn>"""

output_text = process_text(input_text)

print("--- Original ---")
print(input_text)
print("\n--- Processed ---")
print(output_text)

