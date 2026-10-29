from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import json
import sys
import types
from pathlib import Path


def _import_v4_recipe():
    bridge = importlib.import_module("megatron.bridge")
    recipes_root = Path(bridge.__file__).resolve().parent / "recipes"

    vlm_package_name = "megatron.bridge.data.vlm_datasets"
    vlm_package = types.ModuleType(vlm_package_name)
    vlm_package.__path__ = []
    vlm_package.__spec__ = importlib.machinery.ModuleSpec(
        vlm_package_name, loader=None, is_package=True
    )
    sys.modules[vlm_package_name] = vlm_package
    hf_provider_name = f"{vlm_package_name}.hf_provider"
    hf_provider = types.ModuleType(hf_provider_name)
    hf_provider.HFDatasetConversationProvider = type(
        "HFDatasetConversationProvider", (), {}
    )
    sys.modules[hf_provider_name] = hf_provider

    for package_name, package_path in (
        ("megatron.bridge.recipes", recipes_root),
        ("megatron.bridge.recipes.deepseek", recipes_root / "deepseek"),
    ):
        package = types.ModuleType(package_name)
        package.__file__ = str(package_path / "__init__.py")
        package.__package__ = package_name
        package.__path__ = [str(package_path)]
        package.__spec__ = importlib.machinery.ModuleSpec(
            package_name, loader=None, is_package=True
        )
        sys.modules[package_name] = package

    return importlib.import_module(
        "megatron.bridge.recipes.deepseek.deepseek_v4"
    ).deepseek_v4_flash_no_mtp_sft_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DeepSeek-V4-Flash MAGIC LoRA SFT on Megatron Bridge"
    )
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--save", required=True)
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--train-iters", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-warmup-iters", type=int, default=5)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--pipeline-parallel-size", type=int, default=2)
    parser.add_argument("--expert-parallel-size", type=int, default=8)
    parser.add_argument("--lora-dim", type=int, default=24)
    parser.add_argument("--lora-alpha", type=int, default=24)
    parser.add_argument("--run-name", default="DeepSeek-V4-Flash_MAGIC_LoRA_SFT_gate50")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipe = _import_v4_recipe()
    from megatron.bridge.peft.lora import LoRA
    from megatron.bridge.training.config import FinetuningDatasetConfig

    cfg = recipe(hf_path=args.hf_path)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = args.pipeline_parallel_size
    cfg.model.expert_model_parallel_size = args.expert_parallel_size
    cfg.model.context_parallel_size = 1
    cfg.model.seq_length = args.seq_length
    cfg.model.use_fused_mhc = False
    # PP=2 and 4096-token selective recompute retains enough MoE activations
    # to exhaust an H200 when the expert all-to-all allocates its NCCL buffer.
    # Megatron's native whole-layer recomputation is pipeline-aware and avoids
    # the second-backward graph created by the deprecated query-chunk hook.
    cfg.model.recompute_granularity = "full"
    cfg.model.recompute_method = "uniform"
    cfg.model.recompute_num_layers = 1

    cfg.peft = LoRA(
        target_modules=["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"],
        dim=args.lora_dim,
        alpha=args.lora_alpha,
        dropout=0.0,
        normalize_moe_lora=True,
        share_expert_adapters=True,
    )

    cfg.dataset = FinetuningDatasetConfig(
        seq_length=args.seq_length,
        dataset_root=args.dataset_root,
        dataloader_type="batch",
        seed=5678,
    )
    cfg.dataset.do_test = False
    cfg.train.train_iters = args.train_iters
    cfg.train.global_batch_size = args.global_batch_size
    cfg.train.micro_batch_size = args.micro_batch_size
    cfg.optimizer.lr = args.lr
    cfg.optimizer.optimizer_cpu_offload = False
    cfg.optimizer.optimizer_offload_fraction = 0.0
    cfg.scheduler.lr_warmup_iters = args.lr_warmup_iters
    cfg.scheduler.lr_decay_iters = args.train_iters
    cfg.validation.eval_interval = args.train_iters
    cfg.validation.eval_iters = 1

    cfg.checkpoint.pretrained_checkpoint = args.pretrained_checkpoint
    cfg.checkpoint.save = args.save
    cfg.checkpoint.save_interval = args.save_interval
    cfg.checkpoint.most_recent_k = 2
    cfg.checkpoint.load_optim = False
    # GRPO initializes its own optimizer; carrying the SFT Adam moments is not
    # required for adapter continuity.  On this two-node fabric the sharded
    # optimizer save repeatedly failed in RDMA collectives after training.
    cfg.checkpoint.save_optim = False
    # Save the native distributed adapter checkpoint first.  The synchronous
    # HF conversion performs a cross-rank gather and previously timed out after
    # all 50 training steps had completed, leaving an unusable partial DCP.
    # HF PEFT export is intentionally performed as a separate audited stage.
    cfg.checkpoint.also_save_hf_checkpoint = False
    cfg.checkpoint.hf_source_path = args.hf_path
    cfg.checkpoint.hf_trust_remote_code = True
    cfg.logger.log_interval = 1
    cfg.logger.wandb_project = None
    cfg.logger.wandb_exp_name = args.run_name

    resolved = {
        "adaptation": "lora_sft_then_lora_grpo",
        "not_strict_dongcheng_full_parameter_replica": True,
        "global_batch_size": cfg.train.global_batch_size,
        "micro_batch_size": cfg.train.micro_batch_size,
        "train_iters": cfg.train.train_iters,
        "lr": cfg.optimizer.lr,
        "seq_length": cfg.model.seq_length,
        "tp": cfg.model.tensor_model_parallel_size,
        "pp": cfg.model.pipeline_model_parallel_size,
        "ep": cfg.model.expert_model_parallel_size,
        "recompute_granularity": cfg.model.recompute_granularity,
        "recompute_method": cfg.model.recompute_method,
        "recompute_num_layers": cfg.model.recompute_num_layers,
        "lora_dim": cfg.peft.dim,
        "lora_alpha": cfg.peft.alpha,
        "lora_targets": cfg.peft.target_modules,
        "normalize_moe_lora": cfg.peft.normalize_moe_lora,
        "share_expert_adapters": cfg.peft.share_expert_adapters,
        "pretrained_checkpoint": cfg.checkpoint.pretrained_checkpoint,
        "save": cfg.checkpoint.save,
        "also_save_hf_checkpoint": cfg.checkpoint.also_save_hf_checkpoint,
        "save_optim": cfg.checkpoint.save_optim,
    }
    print(json.dumps(resolved, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return

    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.gpt_step import forward_step

    finetune(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
