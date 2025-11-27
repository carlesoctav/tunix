import transformers

tokenizer = transformers.AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
print(f"EOS Token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")
print(f"Pad Token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
print(f"Additional Special Tokens: {tokenizer.additional_special_tokens}")
if "<end_of_turn>" in tokenizer.vocab:
    print(f"<end_of_turn> ID: {tokenizer.vocab['<end_of_turn>']}")
else:
    # Try encoding it
    ids = tokenizer.encode("<end_of_turn>", add_special_tokens=False)
    print(f"<end_of_turn> encoded: {ids}")
