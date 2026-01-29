# data/factory.py

from __future__ import annotations

from typing import Optional

from data.pair import PairSentenceRegressionDataModule
from data.lm import PromptMaskedCausalLMDataModule


def build_pairwise_datamodule(
    *,
    dataset_name: str,
    model_name: str,
    max_seq_length: int,
    batch_size: int,
    tokenize_inputs: bool = True,
    combine_fields: bool = False,
    combine_separator_token: str | None = None,
    # perf knobs
    num_workers: int = 0,
    pin_memory: bool = True,
    map_batch_size: int = 1024,
    map_num_proc: int | None = None,
    load_from_cache_file: bool = True,
    keep_in_memory: bool = False,
) -> Optional[PairSentenceRegressionDataModule]:
    # PairSentenceRegressionDataModule validates registry keys internally (via dataset_key),
    # so we can just attempt construction and let it raise if unknown,
    # but to keep factory contract ("None if unsupported"), catch ValueError.
    try:
        return PairSentenceRegressionDataModule(
            dataset_key=dataset_name,
            model_name_or_path=model_name,
            combine_fields=combine_fields,
            combine_separator=combine_separator_token,
            tokenize_inputs=tokenize_inputs,
            max_seq_length=max_seq_length,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            map_batch_size=map_batch_size,
            map_num_proc=map_num_proc,
            load_from_cache_file=load_from_cache_file,
            keep_in_memory=keep_in_memory,
        )
    except ValueError:
        return None


def build_prompt_masked_lm_datamodule(
    *,
    dataset_name: str,
    model_name: str,
    max_seq_length: int,
    train_batch_size: int,
    eval_batch_size: int,
    combine_fields: bool = False,
    combine_separator_token: str | None = None,
    add_eos: bool = True,
    use_chat_template_if_available: bool = True,
    # perf knobs
    num_workers: int = 0,
    pin_memory: bool = True,
    map_batch_size: int = 256,
    map_num_proc: int | None = None,
    load_from_cache_file: bool = True,
    keep_in_memory: bool = False,
) -> Optional[PromptMaskedCausalLMDataModule]:
    try:
        return PromptMaskedCausalLMDataModule(
            dataset_key=dataset_name,
            model_name_or_path=model_name,
            max_seq_length=max_seq_length,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            combine_fields=combine_fields,
            combine_separator=combine_separator_token,
            add_eos=add_eos,
            use_chat_template_if_available=use_chat_template_if_available,
            num_workers=num_workers,
            pin_memory=pin_memory,
            map_batch_size=map_batch_size,
            map_num_proc=map_num_proc,
            load_from_cache_file=load_from_cache_file,
            keep_in_memory=keep_in_memory,
        )
    except ValueError:
        return None
