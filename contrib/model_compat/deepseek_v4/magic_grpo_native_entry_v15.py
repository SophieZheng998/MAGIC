"""MAGIC DeepSeek-V4 LoRA GRPO entry using Megatron-Core native rollout.

The released DeepSeek-V4 checkpoint uses compressed routed-expert weights.
Megatron Bridge exports updated training weights as BF16, which cannot be
hot-loaded into vLLM's compressed expert kernels.  This entry keeps rollout
on the already-loaded Megatron model so the LoRA adapter is applied exactly
and no quantized/full-weight synchronization is required.

This historical experiment entry patches only
the rollout backend and the previously audited PEFT hyper-connection import.
"""

from __future__ import annotations

import os
import sys
from copy import deepcopy


def _patch_peft_hyper_connection_load() -> None:
    from mcore_bridge.bridge.gpt_bridge import GPTBridge

    original = GPTBridge._set_hyper_connection

    def patched(self, mg_layer, hf_state_dict, layer_idx, to_mcore):
        if to_mcore and getattr(self, "_peft_format", False):
            return None
        return original(self, mg_layer, hf_state_dict, layer_idx, to_mcore)

    GPTBridge._set_hyper_connection = patched


class _MCoreTokenizerAdapter:
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.bos = tokenizer.bos_token_id
        self.eod = tokenizer.eos_token_id

    def tokenize(self, text):
        return self._tokenizer.encode(text, add_special_tokens=False)

    def detokenize(self, token_ids, skip_special_tokens=True):
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)


def _patch_native_rollout() -> None:
    import torch
    import torch.distributed as dist
    from megatron.core.inference.apis import MegatronLLM, SamplingParams
    from megatron.core.inference.config import InferenceConfig
    from swift.infer_engine.protocol import (
        ChatCompletionResponse,
        ChatCompletionResponseChoice,
        ChatMessage,
        RolloutOutput,
        UsageInfo,
    )
    from swift.megatron.trainers.rollout_mixin import MegatronRolloutMixin
    from swift.rlhf_trainers.utils import encode_sample

    def native_init(self):
        self._init_rollout_params()
        self.vllm_mode = "native_megatron"
        self.vllm_gpu_memory_utilization = 0.0
        self.vllm_tensor_parallel_size = 1
        self.use_vllm = False
        self.vllm_use_async_engine = False
        self.enable_offload = False
        self.vllm_version_ge_0_10_2 = True
        self.rollout_enable_lora = False
        self.enable_server_multi_turn = False
        self.base_sync_done = True
        self._cached_vllm_param_names = None
        self._prepare_scheduler()

        # The coordinator drives EP inference on the default NCCL process
        # group from a background thread.  GRPO's foreground ranks must not
        # issue control collectives on that same group while generation is in
        # flight: doing so gives the ranks different NCCL collective orders
        # and deadlocks.  Keep small Python-object control traffic on a
        # dedicated Gloo group instead.
        self._native_control_group = dist.new_group(
            ranks=list(range(dist.get_world_size())), backend="gloo"
        )

        model = self.unwrapped_models[0]
        model.eval()
        tokenizer = _MCoreTokenizerAdapter(self.template.tokenizer)
        config = InferenceConfig(
            buffer_size_gb=2,
            max_requests=32,
            max_tokens=4096,
            num_cuda_graphs=0,
        )
        self._native_llm = MegatronLLM(
            model=model,
            tokenizer=tokenizer,
            inference_config=config,
            use_coordinator=True,
        )
        dist.barrier(group=self._native_control_group)

    def native_generate(self, samples):
        samples = self._preprocess_inputs(samples)
        local_prompt_ids = []
        for sample in samples:
            encoded = encode_sample(sample, self.template, encode_prompt_only=True)
            ids = encoded["input_ids"]
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            if ids and isinstance(ids[0], list):
                ids = ids[0]
            local_prompt_ids.append([int(token) for token in ids])

        control_group = self._native_control_group
        control_world_size = dist.get_world_size(group=control_group)
        control_rank = dist.get_rank(group=control_group)
        gathered = [None for _ in range(control_world_size)]
        dist.all_gather_object(gathered, local_prompt_ids, group=control_group)
        lengths = [len(chunk) for chunk in gathered]
        flat_prompts = [prompt for chunk in gathered for prompt in chunk]

        model = self.unwrapped_models[0]
        model.eval()
        dist.barrier(group=control_group)
        packed = None
        if control_rank == 0:
            params = SamplingParams(
                temperature=float(self.temperature),
                top_k=max(0, int(getattr(self.args, "top_k", 0))),
                top_p=float(getattr(self.args, "top_p", 1.0)),
                return_log_probs=True,
                skip_prompt_log_probs=True,
                num_tokens_to_generate=int(self.max_completion_length),
                termination_id=self.template.tokenizer.eos_token_id,
            )
            records = self._native_llm.generate(flat_prompts, params)
            packed = []
            for record in records:
                tokens = record.generated_tokens
                if isinstance(tokens, torch.Tensor):
                    tokens = tokens.tolist()
                tokens = [int(token) for token in tokens]
                logps = record.generated_log_probs
                if isinstance(logps, torch.Tensor):
                    logps = logps.tolist()
                logps = [float(value) for value in (logps or [])]
                text = record.generated_text
                if text is None:
                    text = self.template.tokenizer.decode(tokens, skip_special_tokens=True)
                packed.append({"tokens": tokens, "logps": logps, "text": text})

        objects = [packed]
        dist.broadcast_object_list(objects, src=0, group=control_group)
        packed = objects[0]
        start = sum(lengths[:control_rank])
        local_records = packed[start : start + lengths[control_rank]]
        outputs = []
        for sample, record in zip(samples, local_records):
            tokens = record["tokens"]
            logps = record["logps"]
            finish_reason = "length" if len(tokens) >= self.max_completion_length else "stop"
            logprob_content = [{"logprob": value} for value in logps] if logps else None
            choice = ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=record["text"]),
                finish_reason=finish_reason,
                logprobs={"content": logprob_content} if logprob_content is not None else None,
                token_ids=tokens,
            )
            response = ChatCompletionResponse(
                model="deepseek-v4-flash-megatron-native",
                choices=[choice],
                usage=UsageInfo(
                    prompt_tokens=len(local_prompt_ids[len(outputs)]),
                    completion_tokens=len(tokens),
                    total_tokens=len(local_prompt_ids[len(outputs)]) + len(tokens),
                ),
                id=sample.request_id,
                prompt_token_ids=local_prompt_ids[len(outputs)],
            )
            outputs.append(
                RolloutOutput(
                    response=response,
                    response_token_ids=[tokens],
                    rollout_logprobs=[logps] if logps else [],
                )
            )
        dist.barrier(group=control_group)
        model.train()
        return self._postprocess_rollout_outputs(samples, outputs)

    MegatronRolloutMixin._init_rollout_engine = native_init
    MegatronRolloutMixin._generate_completions = native_generate


def _patch_native_pipeline_client() -> None:
    """Bypass Swift's server-mode vLLM handshake for native MCore rollout.

    The CLI still needs ``vllm_mode=server`` to satisfy argument validation,
    but this entry never uses a VLLMClient: rollout is performed by the
    Megatron model already owned by the trainer.
    """
    from swift.megatron.pipelines.train.rlhf import MegatronRLHF

    def no_vllm_client(self):
        return None

    MegatronRLHF._prepare_vllm_client = no_vllm_client


if __name__ == "__main__":
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    _patch_peft_hyper_connection_load()
    _patch_native_rollout()
    _patch_native_pipeline_client()
    from swift.megatron import megatron_rlhf_main

    megatron_rlhf_main()
