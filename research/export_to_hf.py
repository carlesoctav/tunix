import os
import shutil
import json
import jax
import jax.numpy as jnp
import numpy as np
import torch
from safetensors.torch import save_file
from flax import nnx
from tunix.cli.utils import model as model_lib
from tunix.models.gemma3 import params as gemma3_params_lib
from tunix.sft import checkpoint_manager
import transformers
from huggingface_hub import HfApi, create_repo

# Configuration
MODEL_NAME = "gemma3-1b"
BASE_MODEL_PATH = "/home/carlesoctav/personal/tunix/models/gemma1b/gemma3-1b-it"
CHECKPOINT_DIR = "/mnt/carles/models/exp/exp2"
OUTPUT_DIR = "./exported_model"
HF_REPO_ID = "carlesoctav/gemma3-1b-vibecheck" # Replace with desired repo
HF_TOKEN = os.environ.get("HF_TOKEN")

def create_mesh():
    num_devices = jax.device_count()
    mesh_shape = (1, num_devices)
    return jax.make_mesh(
        mesh_shape,
        ("fsdp", "tp"),
        axis_types=(jax.sharding.AxisType.Auto,) * 2,
    )

def get_path_str(path):
    parts = []
    for p in path:
        if hasattr(p, 'key'):
            parts.append(str(p.key))
        elif hasattr(p, 'name'):
            parts.append(str(p.name))
        else:
            parts.append(str(p))
    return "/".join(parts)

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
    
    # 4. Extract State and Convert/Merge
    print("Extracting state and merging LoRA...")
    state = nnx.state(model)
    flat_state = jax.tree_util.tree_leaves_with_path(state)
    
    # Convert to dict for easier access
    param_dict = {}
    for path, val in flat_state:
        key = get_path_str(path)
        # Ensure value is on CPU numpy
        arr = np.array(val)
        # Convert bfloat16 to float32 because torch.tensor(numpy_bf16) is not supported
        if arr.dtype.name == 'bfloat16':
            arr = arr.astype(np.float32)
        param_dict[key] = arr

    hf_state_dict = {}
    
    # Helper to get merged weight
    def get_merged(base_key, lora_a_key, lora_b_key, transpose_lora_b=False):
        w = param_dict[base_key]
        if lora_a_key in param_dict and lora_b_key in param_dict:
            a = param_dict[lora_a_key]
            b = param_dict[lora_b_key]
            alpha = lora_config["alpha"]
            rank = lora_config["rank"]
            scaling = alpha / rank
            
            # LoRA logic depends on layer type
            # For einsums in Gemma 3:
            # q_einsum: w(N, D, H), a(D, R), b(R, N, H)
            if "q_einsum" in base_key:
                 # b is (R, N, H) -> reshape (R, N*H)
                 # a is (D, R)
                 # a @ b -> (D, N*H) -> reshape (D, N, H) -> transpose (N, D, H)
                 # Actually: einsum('dr,rnh->ndh') matches.
                 delta = np.einsum('dr,rnh->ndh', a, b)
                 w = w + delta * scaling
            elif "kv_einsum" in base_key:
                 # w(2, 1, D, H)
                 # a(D, R)
                 # b(R, 2, 1, H)
                 # einsum('dr,rckh->ckdh')
                 delta = np.einsum('dr,rckh->ckdh', a, b)
                 w = w + delta * scaling
            elif "attn_vec_einsum" in base_key:
                 pass
            else:
                 # Linear layers (gate, up, down)
                 # w(In, Out) or similar
                 # a(In, R), b(R, Out) -> a @ b
                 delta = a @ b
                 w = w + delta * scaling
        return w

    # Mapping Loop
    num_layers = model_params.num_layers
    
    # Embeddings
    # Note: We convert to float32 above, but for saving we might want BF16 if supported
    # But safe_tensors/torch usually handles float32 fine. We can cast back to BF16 in torch if desired.
    def to_torch(arr):
        t = torch.tensor(arr)
        # Convert back to bfloat16 for saving to save space/match original precision
        return t.to(torch.bfloat16).contiguous()

    hf_state_dict["model.embed_tokens.weight"] = to_torch(param_dict["embedder/input_embedding/value"])
    
    # Final Norm
    hf_state_dict["model.norm.weight"] = to_torch(param_dict["final_norm/scale/value"])
    
    for i in range(num_layers):
        layer_prefix = f"layers/{i}"
        hf_prefix = f"model.layers.{i}"
        
        # Norms
        hf_state_dict[f"{hf_prefix}.input_layernorm.weight"] = to_torch(param_dict[f"{layer_prefix}/pre_attention_norm/scale/value"])
        hf_state_dict[f"{hf_prefix}.post_attention_layernorm.weight"] = to_torch(param_dict[f"{layer_prefix}/post_attention_norm/scale/value"])
        hf_state_dict[f"{hf_prefix}.pre_feedforward_layernorm.weight"] = to_torch(param_dict[f"{layer_prefix}/pre_ffw_norm/scale/value"])
        hf_state_dict[f"{hf_prefix}.post_feedforward_layernorm.weight"] = to_torch(param_dict[f"{layer_prefix}/post_ffw_norm/scale/value"])
        
        # Attention Norms
        hf_state_dict[f"{hf_prefix}.self_attn.q_norm.weight"] = to_torch(param_dict[f"{layer_prefix}/attn/_query_norm/scale/value"])
        hf_state_dict[f"{hf_prefix}.self_attn.k_norm.weight"] = to_torch(param_dict[f"{layer_prefix}/attn/_key_norm/scale/value"])
        
        # Attention Projections
        # Q
        q_base = f"{layer_prefix}/attn/q_einsum/w/value"
        q_a = f"{layer_prefix}/attn/q_einsum/w_lora_a/value"
        q_b = f"{layer_prefix}/attn/q_einsum/w_lora_b/value"
        w_q = get_merged(q_base, q_a, q_b) # (N, D, H)
        # HF: (N*H, D)
        # Transpose (0, 2, 1) -> (N, H, D). Reshape (-1, D).
        w_q_hf = w_q.transpose(0, 2, 1).reshape(-1, w_q.shape[1])
        hf_state_dict[f"{hf_prefix}.self_attn.q_proj.weight"] = to_torch(w_q_hf)
        
        # KV
        kv_base = f"{layer_prefix}/attn/kv_einsum/w/value"
        kv_a = f"{layer_prefix}/attn/kv_einsum/w_lora_a/value"
        kv_b = f"{layer_prefix}/attn/kv_einsum/w_lora_b/value"
        w_kv = get_merged(kv_base, kv_a, kv_b) # (2, NumKV, D, H) -> (2, 1, 1152, 256)
        
        # K: w_kv[0] -> (1, D, H). HF: (1*H, D)
        w_k = w_kv[0]
        w_k_hf = w_k.transpose(0, 2, 1).reshape(-1, w_k.shape[1])
        hf_state_dict[f"{hf_prefix}.self_attn.k_proj.weight"] = to_torch(w_k_hf)
        
        # V: w_kv[1] -> (1, D, H). HF: (1*H, D)
        w_v = w_kv[1]
        w_v_hf = w_v.transpose(0, 2, 1).reshape(-1, w_v.shape[1])
        hf_state_dict[f"{hf_prefix}.self_attn.v_proj.weight"] = to_torch(w_v_hf)
        
        # O Proj (No LoRA in config)
        o_base = f"{layer_prefix}/attn/attn_vec_einsum/w/value"
        w_o = param_dict[o_base] # (N, H, D)
        # HF: (D, N*H)
        # Reshape to (N*H, D). Transpose to (D, N*H)
        w_o_hf = w_o.reshape(-1, w_o.shape[2]).T
        hf_state_dict[f"{hf_prefix}.self_attn.o_proj.weight"] = to_torch(w_o_hf)
        
        # MLP
        # Gate
        g_base = f"{layer_prefix}/mlp/gate_proj/kernel/value"
        g_a = f"{layer_prefix}/mlp/gate_proj/kernel_lora_a/value"
        g_b = f"{layer_prefix}/mlp/gate_proj/kernel_lora_b/value"
        w_g = get_merged(g_base, g_a, g_b) # (In, Out)
        # HF: (Out, In)
        hf_state_dict[f"{hf_prefix}.mlp.gate_proj.weight"] = to_torch(w_g.T)
        
        # Up
        u_base = f"{layer_prefix}/mlp/up_proj/kernel/value"
        u_a = f"{layer_prefix}/mlp/up_proj/kernel_lora_a/value"
        u_b = f"{layer_prefix}/mlp/up_proj/kernel_lora_b/value"
        w_u = get_merged(u_base, u_a, u_b) # (In, Out)
        # HF: (Out, In)
        hf_state_dict[f"{hf_prefix}.mlp.up_proj.weight"] = to_torch(w_u.T)
        
        # Down
        d_base = f"{layer_prefix}/mlp/down_proj/kernel/value"
        d_a = f"{layer_prefix}/mlp/down_proj/kernel_lora_a/value"
        d_b = f"{layer_prefix}/mlp/down_proj/kernel_lora_b/value"
        w_d = get_merged(d_base, d_a, d_b) # (In, Out)
        # HF: (Out, In)
        hf_state_dict[f"{hf_prefix}.mlp.down_proj.weight"] = to_torch(w_d.T)

    # 5. Save
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    
    print(f"Saving SafeTensors to {OUTPUT_DIR}...")
    save_file(hf_state_dict, os.path.join(OUTPUT_DIR, "model.safetensors"))
    
    # 6. Config and Tokenizer
    print("Saving config and tokenizer...")
    tokenizer = transformers.AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    config = transformers.AutoConfig.from_pretrained("google/gemma-3-1b-it", trust_remote_code=True)
    config.save_pretrained(OUTPUT_DIR)

    # 7. Upload
    print(f"Uploading to HuggingFace Hub ({HF_REPO_ID})...")
    api = HfApi(token=HF_TOKEN)
    try:
        create_repo(HF_REPO_ID, token=HF_TOKEN, exist_ok=True, repo_type="model")
    except Exception as e:
        print(f"Repo creation failed (might exist): {e}")

    api.upload_folder(
        folder_path=OUTPUT_DIR,
        repo_id=HF_REPO_ID,
        repo_type="model"
    )
    print("Upload complete!")

if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    main()
