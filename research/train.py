import dataclasses
import inspect
from typing import Any, Dict, Optional, Tuple, Union
import jax
import jax.numpy as jnp
import optax
from absl import app
from tunix.cli.utils import model as model_lib
from tunix.sft import peft_trainer
from tunix.sft import utils
from research.datasets_utils import TrainMixDataConfig, HFSource

from tunix.sft.peft_trainer import TrainingConfig, MetricsLoggerOptions
from dotenv import load_dotenv
load_dotenv()


@dataclasses.dataclass
class MeshConfig:
    shape: Tuple[int, ...]
    axis_names: Tuple[str, ...]

@dataclasses.dataclass
class LoraConfig:
    rank: int
    alpha: float
    weight_qtype: str
    tile_size: int
    module_path: str

@dataclasses.dataclass
class ModelConfig:
    model_name: str
    model_id: str
    model_source: str
    model_download_path: str
    intermediate_ckpt_dir: str
    mesh: MeshConfig
    lora_config: Optional[LoraConfig] = None
    model_display: bool = False

@dataclasses.dataclass
class TokenizerConfig:
    tokenizer_path: str
    tokenizer_type: str
    add_bos: bool = True
    add_eos: bool = True

@dataclasses.dataclass
class OptimizerConfig:
    opt_type: str = "adamw"
    learning_rate: float = 1e-5
    schedule_type: Optional[str] = None
    schedule_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    opt_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

@dataclasses.dataclass
class TrainArgs:
    model_config: ModelConfig
    tokenizer_config: TokenizerConfig
    optimizer_config: OptimizerConfig
    training_config: TrainingConfig
    train_mix_data_config: TrainMixDataConfig

def _extract_kwargs(func, config_dict):
    """Extracts arguments from config_dict that match func's signature."""
    sig = inspect.signature(func)
    kwargs = {}
    for param in sig.parameters.values():
        if param.name in config_dict:
            kwargs[param.name] = config_dict[param.name]
    return kwargs

def create_learning_rate(optimizer_config: OptimizerConfig) -> Union[float, optax.Schedule]:
    """Creates a learning rate or schedule."""
    if optimizer_config.schedule_type:
        try:
            schedule_fn = getattr(optax.schedules, optimizer_config.schedule_type)
        except AttributeError:
            raise ValueError(f"Schedule type '{optimizer_config.schedule_type}' not found in optax.schedules")
        
        # Combine direct kwargs with flattened config for extraction
        config_data = {**dataclasses.asdict(optimizer_config), **optimizer_config.schedule_kwargs}
        
        sched_kwargs = _extract_kwargs(schedule_fn, config_data)
        return schedule_fn(**sched_kwargs)
    
    return optimizer_config.learning_rate

def create_optimizer(optimizer_config: OptimizerConfig) -> optax.GradientTransformation:
    """Creates the optimizer based on the config."""
    opt_type = optimizer_config.opt_type
    
    # Get LR or Schedule
    learning_rate = create_learning_rate(optimizer_config)
    
    try:
        opt_func = getattr(optax, opt_type.lower())
    except AttributeError:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")

    config_data = {**dataclasses.asdict(optimizer_config), **optimizer_config.opt_kwargs}
    opt_kwargs = _extract_kwargs(opt_func, config_data)
    
    opt_kwargs['learning_rate'] = learning_rate
    
    return opt_func(**opt_kwargs)

def create_mesh(mesh_config: dict) -> jax.sharding.Mesh:
    """Creates a JAX mesh from configuration dict."""
    shape = mesh_config["shape"]
    axis_names = mesh_config["axis_names"]
    
    num_devices = jax.device_count()
    import numpy as np
    if np.prod(shape) > num_devices:
         print(f"Warning: Requested mesh shape {shape} requires {np.prod(shape)} devices, but found {num_devices}. Adjusting to (1, 1).")
         shape = (1, 1)
         axis_names = ("fsdp", "tp")

    return jax.make_mesh(
        tuple(shape),
        tuple(axis_names),
        axis_types=(jax.sharding.AxisType.Auto,) * len(tuple(axis_names)),
    )

class TrainPipeline:
    def __init__(self, args: TrainArgs):
        self.args = args

    def run(self):
        model_config_dict = dataclasses.asdict(self.args.model_config)
        tokenizer_config_dict = dataclasses.asdict(self.args.tokenizer_config)
        
        mesh = create_mesh(model_config_dict["mesh"])
        
        model, tokenizer_path = model_lib.create_model(
            model_config_dict, tokenizer_config_dict, mesh
        )

        if model is None:
            raise ValueError("Model is None")

        # tokenizer_config_dict['tokenizer_path'] = tokenizer_path
        #
        # tokenizer = model_lib.create_tokenizer(
        #     tokenizer_config_dict, tokenizer_path
        # )

        optimizer = create_optimizer(self.args.optimizer_config)
        
        trainer = peft_trainer.PeftTrainer(
            model,
            optimizer,
            self.args.training_config,
        )

        def gen_model_input_fn(x: peft_trainer.TrainingInput):
            pad_mask = x.input_tokens != 0
            positions = utils.build_positions_from_mask(pad_mask)
            attention_mask = utils.make_causal_attn_mask(pad_mask)
            return {
                'input_tokens': x,
                'input_mask': jnp.ones(x.shape),
                'positions': positions,
                'attention_mask': attention_mask,
            }

        trainer = trainer.with_gen_model_input_fn(gen_model_input_fn)
        train_ds = self.args.train_mix_data_config.make()

        with mesh:
            trainer.train(train_ds, None)

def main():
    metrics_opts = MetricsLoggerOptions(
        log_dir="./logs",
        flush_every_n_steps=20
    )

    args = TrainArgs(
        model_config=ModelConfig(
            model_name="gemma3-1b-it",
            model_id="/home/carlesoctav/personal/tunix/models/gemma1b/gemma3-1b-it",
            model_source="local",
            model_download_path="./models/gemma1b",
            intermediate_ckpt_dir="./models/gemma1b-int",
            mesh=MeshConfig(
                shape=(4, 1),
                axis_names=("fsdp", "tp")
            ),
            lora_config=LoraConfig(
                rank=16,
                alpha=2.0,
                weight_qtype="nf4",
                tile_size=256,
                module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj"
            ),
            model_display=False
        ),
        tokenizer_config = TokenizerConfig(
            tokenizer_path = "no need", 
            tokenizer_type = "sentencepiece"
        ),
        optimizer_config=OptimizerConfig(
            opt_type="adamw",
            learning_rate=1e-5,
            schedule_type="warmup_cosine_decay_schedule",
            schedule_kwargs={
                "init_value": 0.0,
                "peak_value": 3e-5,
                "warmup_steps": 5,
                "decay_steps": 10,
                "end_value": 0.0
            },
            opt_kwargs={
                "weight_decay": 0.1
            }
        ),
        training_config=TrainingConfig(
            eval_every_n_steps=None,
            max_steps=None,
            metrics_logging_options=metrics_opts, 
            checkpoint_root_directory = "/mnt/carles/models/exp1"
        ),
        train_mix_data_config=TrainMixDataConfig(
            sources=[HFSource(
                path="allenai/Dolci-Think-SFT", 
                streaming = True,
            )],
            tokenizer_path="google/gemma-3-1b-it",
            batch_size=16
        )
    )
    pipeline = TrainPipeline(args)
    pipeline.run()

if __name__ == "__main__":
    load_dotenv()
    main()
