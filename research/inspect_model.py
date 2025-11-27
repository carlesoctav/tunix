import os
import jax
import flax.nnx as nnx
from tunix.cli.utils import model as model_lib
from tunix.models.gemma3 import params as gemma3_params_lib
from tunix.sft import checkpoint_manager
import transformers

# Hardcoded configuration
MODEL_NAME = "gemma3-1b"
BASE_MODEL_PATH = "/home/carlesoctav/personal/tunix/models/gemma1b/gemma3-1b-it"
CHECKPOINT_DIR = "/mnt/carles/models/exp/exp2"

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
    
    # 4. Inspect State
    print("Inspecting Model State Structure...")
    state = nnx.state(model)
    
    # Print first few keys to understand structure
    flat_state = jax.tree_util.tree_leaves_with_path(state)
    print(f"Total parameters: {len(flat_state)}")
    
    with open("tunix_keys.txt", "w") as f:
        for path, val in flat_state:
            # Handle diverse key types
            path_parts = []
            for p in path:
                if hasattr(p, 'key'):
                    path_parts.append(str(p.key))
                elif hasattr(p, 'name'):
                    path_parts.append(str(p.name))
                else:
                    path_parts.append(str(p))
            path_str = "/".join(path_parts)
            f.write(f"{path_str}: {val.shape}\n")
            if len(path_parts) < 50: # Print first 50 to stdout
                 pass # controlled below

    # Print first 20 lines of the file
    print("First 20 keys:")
    with open("tunix_keys.txt", "r") as f:
        for i, line in enumerate(f):
            if i < 20:
                print(line.strip())
            else:
                break

    # Check specific LoRA keys
    print("\nChecking for LoRA keys...")
    lora_found = False
    with open("tunix_keys.txt", "r") as f:
         for line in f:
            if "lora" in line.lower():
                print(f"Found LoRA param: {line.strip()}")
                lora_found = True
                if "down_proj" in line:
                     pass # Don't break immediately to see more variety

    
    if not lora_found:
        print("WARNING: No LoRA keys found in state!")

    # 5. Check HF Keys
    print("\nChecking HF Model Keys (loading empty/config if possible)...")
    try:
        config = transformers.AutoConfig.from_pretrained("google/gemma-3-1b-it", trust_remote_code=True)
        print("HF Architecture:", config.architectures)
        
        # Try to instantiate dummy model
        # 1B parameters is ~2GB-4GB, should fit in RAM
        print("Instantiating HF model on CPU (might take a moment)...")
        model_hf = transformers.AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        
        print("\nFirst 20 HF Keys:")
        count = 0
        for name, _ in model_hf.named_parameters():
            print(f"{name}")
            count += 1
            if count >= 20:
                break
        
        # Check for norms in HF keys
        print("\nChecking for norm keys in HF...")
        norm_keys = [name for name, _ in model_hf.named_parameters() if "norm" in name and "layers.0" in name]
        for nk in norm_keys:
            print(nk)

    except Exception as e:
        print(f"Could not load HF info: {e}")

if __name__ == "__main__":
    main()
