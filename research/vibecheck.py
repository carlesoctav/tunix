import os
import dataclasses
from typing import Any, Dict, Tuple
import jax
import flax.nnx as nnx
from tunix.cli.utils import model as model_lib
from tunix.models.gemma3 import params as gemma3_params_lib
from tunix.sft import checkpoint_manager
from tunix.generate import sampler
import transformers

# Hardcoded configuration to match research/train.py
MODEL_NAME = "gemma3-1b"
BASE_MODEL_PATH = "/home/carlesoctav/personal/tunix/models/gemma1b/gemma3-1b-it"
CHECKPOINT_DIR = "/mnt/carles/models/exp/exp2"
TOKENIZER_NAME = "google/gemma-3-1b-it"

def create_mesh():
    num_devices = jax.device_count()
    mesh_shape = (1, num_devices)
    return jax.make_mesh(
        mesh_shape,
        ("fsdp", "tp"),
        axis_types=(jax.sharding.AxisType.Auto,) * 2,
    )

def main():
    print(f"Initializing {MODEL_NAME}...")
    mesh = create_mesh()
    
    # 1. Load Base Model
    print(f"Loading base model from {BASE_MODEL_PATH}...")
    model_params = model_lib.obtain_model_params(MODEL_NAME)
    
    model = gemma3_params_lib.create_model_from_checkpoint(
        BASE_MODEL_PATH, model_params, mesh
    )

    # 2. Apply LoRA
    lora_config = {
        "rank": 16,
        "alpha": 16.0,
        "module_path": ".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj",
    }
    print("Applying LoRA adapter...")
    model = model_lib.apply_lora_to_model(model, mesh, lora_config)

    # 3. Restore Finetuned Weights
    print(f"Restoring checkpoint from {CHECKPOINT_DIR}...")
    ckpt_manager = checkpoint_manager.CheckpointManager(root_directory=CHECKPOINT_DIR)
    step = ckpt_manager.latest_step()
    if step is None:
        raise ValueError(f"No checkpoint found in {CHECKPOINT_DIR}")
    
    print(f"Found latest checkpoint step: {step}")
    ckpt_manager.maybe_restore(model, step=step, restore_only_lora_params=True)
    
    # 4. Setup Tokenizer
    print(f"Loading tokenizer from HuggingFace ({TOKENIZER_NAME})...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 5. Setup Sampler
    print("Setting up sampler...")
    cache_config = sampler.CacheConfig(
        cache_size=8192,
        num_layers=model_params.num_layers,
        num_kv_heads=model_params.num_kv_heads,
        head_dim=model_params.head_dim
    )
    
    my_sampler = sampler.Sampler(
        model,
        tokenizer,
        cache_config,
    )
    
    print("\n" + "="*50)
    print("       Tunix Vibecheck (Gemma 3 LoRA)       ")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("User > ")
            if user_input.lower() in ("/exit", "/quit"):
                break
                
            messages = [{"role": "user", "content": user_input}]
            
            # Apply chat template to format input
            prompt_str = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            prompts = [prompt_str]
            
            # Determine EOS tokens (standard EOS + <end_of_turn>)
            eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
            eos_ids = [tokenizer.eos_token_id]
            if isinstance(eot_id, int):
                eos_ids.append(eot_id)
            
            # Generate
            out = my_sampler(
                prompts, 
                max_generation_steps=2048, 
                echo=False, 
                top_p=0.95, 
                temperature=0.7,
                eos_tokens=eos_ids
            )
            
            print(f"Model > {out.text[0]}")
            print("-" * 30)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
