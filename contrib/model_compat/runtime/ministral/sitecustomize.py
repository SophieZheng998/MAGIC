"""Opt-in runtime compatibility hooks for the MAGIC smoke pipeline."""

import os


if os.environ.get("MAGIC_MINISTRAL_TEXT_ONLY_SYNC_PATCH") == "1":
    from r2ai_magic.rl.vllm_weight_patch import (
        install_verl_010_no_rmpad_entropy_chunk_patch,
        install_vllm_018_pixtral_text_only_patch,
    )

    install_vllm_018_pixtral_text_only_patch()
    install_verl_010_no_rmpad_entropy_chunk_patch(
        int(os.environ.get("MAGIC_ENTROPY_CHUNK_SIZE", "128"))
    )
