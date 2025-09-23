from typing import Optional
import os

import numpy as np

from vllm import LLM


class VllmEmbedding:
    def __init__(
        self,
        model_id: str,
        num_gpus: int,
        seed: int,
        max_length: int,
        gpu_memory_utilization: float,
        instruction: str,
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
            task="embed",
            tensor_parallel_size=tp,
            seed=seed,
            trust_remote_code=True,
            max_model_len=max_length,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        self.instruction = instruction

    def __call__(
        self,
        query: str,
    ) -> np.ndarray:
        embedding = self.embed(query=query)
        return embedding

    def embed(
        self,
        query: str,
    ) -> np.ndarray:
        input_text = self.get_detailed_instruction(query=query)
        output = self.llm.embed(input_text)
        embedding = output[0].outputs.embedding
        embedding = np.array(
            embedding,
            dtype=np.float32,
        )
        return embedding

    def get_detailed_instruction(
        self,
        query: str,
    ) -> str:
        instruction = f"Instruct: {self.instruction}\nQuery:{query}"
        return instruction
