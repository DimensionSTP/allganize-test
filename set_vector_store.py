import dotenv

dotenv.load_dotenv(
    override=True,
)

import os
import warnings

os.environ["HYDRA_FULL_ERROR"] = "1"
warnings.filterwarnings("ignore")

import numpy as np

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.databases import FaissIndex
from src.models import VllmEmbedding


@hydra.main(
    config_path="configs/",
    config_name="main.yaml",
)
def set_vector_store(
    config: DictConfig,
) -> None:
    index: FaissIndex = instantiate(
        config.database,
    )
    embedding: VllmEmbedding = instantiate(
        config.model.embedding,
    )

    queries = index.df[config.target_column_name].tolist()
    embedded = [embedding(query=query) for query in queries]
    embedded = np.array(
        embedded,
        dtype=np.float32,
    )
    index.add(embedded=embedded)
    index.save()


if __name__ == "__main__":
    set_vector_store()
