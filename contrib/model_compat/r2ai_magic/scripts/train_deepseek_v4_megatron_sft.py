from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import json
import sys
import types
from pathlib import Path


def _import_v4_recipe():
    """Load the V4 recipe without importing unrelated diffusion/VLM recipes."""
    bridge = importlib.import_module("megatron.bridge")
    recipes_root = Path(bridge.__file__).resolve().parent / "recipes"

    vlm_package_name = "megatron.bridge.data.vlm_datasets"
    vlm_package = types.ModuleType(vlm_package_name)
    vlm_package.__path__ = []
    vlm_package.__spec__ = importlib.machinery.ModuleSpec(vlm_package_name, loader=None, is_package=True)
    sys.modules[vlm_package_name] = vlm_package
    hf_provider_name = f"{vlm_package_name}.hf_provider"
    hf_provider = types.ModuleType(hf_provider_name)
    hf_provider.HFDatasetConversationProvider = type("HFDatasetConversationProvider", (), {})
    sys.modules[hf_provider_name] = hf_provider

    for package_name, package_path in (
        ("megatron.bridge.recipes", recipes_root),
        ("megatron.bridge.recipes.deepseek", recipes_root / "deepseek"),
    ):
        package = types.ModuleType(package_name)
        package.__file__ = str(package_path / "__init__.py")
        package.__package__ = package_name
        package.__path__ = [str(package_path)]
        package.__spec__ = importlib.machinery.ModuleSpec(package_name, loader=None, is_package=True)
        sys.modules[package_name] = package

    return importlib.import_module(
        "megatron.bridge.recipes.deepseek.deepseek_v4"
    ).deepseek_v4_flash_no_mtp_sft_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek-V4-Flash full SFT on Megatron Bridge")
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--pretrained-checkpoint", required=True)
    parser.add_argument("--save", required=True)
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--global-batch-size", type=int, default=32)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--train-iters", type=int, default=2093)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lr-warmup-iters", type=int, default=209)
    parser.add_argument("--eval-interval", type=int, default=400)
    parser.add_argument("--eval-iters", type=int, default=2)
    parser.add_argument("--save-interval", type=int, default=2093)
    parser.add_argument("--keep-checkpoints", type=int, default=1)
    parser.add_argument("--run-name", default="DeepSeek-V4-Flash_magic_sft_h200_tp1_pp4_ep8")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipe = _import_v4_recipe()

    from megatron.bridge.training.config import FinetuningDatasetConfig
    cfg = recipe(hf_path=args.hf_path)
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 4
    cfg.model.expert_model_parallel_size = 8
    cfg.model.context_parallel_size = 1
    cfg.model.seq_length = args.seq_length
    cfg.model.use_fused_mhc = False

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
    cfg.scheduler.lr_warmup_iters = args.lr_warmup_iters
    cfg.scheduler.lr_decay_iters = args.train_iters
    cfg.validation.eval_interval = args.eval_interval
    cfg.validation.eval_iters = args.eval_iters

    cfg.checkpoint.pretrained_checkpoint = args.pretrained_checkpoint
    cfg.checkpoint.save = args.save
    cfg.checkpoint.save_interval = args.save_interval
    cfg.checkpoint.most_recent_k = args.keep_checkpoints
    cfg.checkpoint.load_optim = False
    cfg.checkpoint.save_optim = False
    cfg.logger.log_interval = 1
    cfg.logger.wandb_project = None
    cfg.logger.wandb_exp_name = args.run_name

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dataset_root": cfg.dataset.dataset_root,
                    "eval_interval": cfg.validation.eval_interval,
                    "global_batch_size": cfg.train.global_batch_size,
                    "lr": cfg.optimizer.lr,
                    "pretrained_checkpoint": cfg.checkpoint.pretrained_checkpoint,
                    "save": cfg.checkpoint.save,
                    "save_interval": cfg.checkpoint.save_interval,
                    "seq_length": cfg.model.seq_length,
                    "train_iters": cfg.train.train_iters,
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return

    from megatron.bridge.training.finetune import finetune
    from megatron.bridge.training.gpt_step import forward_step

    finetune(config=cfg, forward_step_func=forward_step)


if __name__ == "__main__":
    main()
