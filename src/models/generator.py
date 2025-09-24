from typing import Dict, Any, Optional
import os

from transformers import AutoTokenizer

from vllm import LLM, SamplingParams


class VllmGenerator:
    def __init__(
        self,
        model_id: str,
        num_gpus: int,
        seed: int,
        max_length: int,
        gpu_memory_utilization: float,
        instruction: Dict[str, str],
        role_column_name: str,
        content_column_name: str,
        max_new_tokens: int,
        do_sample: bool,
        generation_config: Dict[str, Any],
        device_id: Optional[int],
        master_addr: Optional[str],
        master_port: Optional[int],
        nccl_socket_ifname: Optional[str],
        nccl_ib_disable: Optional[int],
    ) -> None:
        if device_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        if master_addr is not None:
            os.environ["MASTER_ADDR"] = str(master_addr)
        if master_port is not None:
            os.environ["MASTER_PORT"] = str(master_port)
        if nccl_socket_ifname is not None:
            os.environ["NCCL_SOCKET_IFNAME"] = str(nccl_socket_ifname)
        if nccl_ib_disable is not None:
            os.environ["NCCL_IB_DISABLE"] = str(nccl_ib_disable)

        os.environ.setdefault(
            "VLLM_WORKER_MULTIPROC_METHOD",
            "spawn",
        )
        for var in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
            if var in os.environ:
                del os.environ[var]

        tp = 1 if device_id is not None else num_gpus
        self.llm = LLM(
            model=model_id,
            tensor_parallel_size=tp,
            seed=seed,
            trust_remote_code=True,
            max_model_len=max_length,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            is_enable_thinking=False,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.instruction = instruction
        self.role_column_name = role_column_name
        self.content_column_name = content_column_name

        if do_sample:
            self.generation_config = generation_config
        else:
            self.generation_config = {
                "temperature": 0,
                "top_p": 1,
            }

        self.sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            skip_special_tokens=True,
            stop_token_ids=[self.tokenizer.eos_token_id],
            **self.generation_config,
        )

    def __call__(
        self,
        recommendations: str,
    ) -> str:
        generation = self.generate(query=recommendations)
        return generation

    def generate(
        self,
        query: str,
    ) -> str:
        output = self.llm.generate(
            prompts=[query],
            sampling_params=self.sampling_params,
        )
        generation = output[0].outputs[0].text.strip()
        return generation
