import dotenv

dotenv.load_dotenv(
    override=True,
)

import os
import warnings

os.environ["HYDRA_FULL_ERROR"] = "1"
warnings.filterwarnings("ignore")

import hydra
from omegaconf import DictConfig

from src.pipelines import pipeline


@hydra.main(
    config_path="configs/",
    config_name="main.yaml",
)
def main(
    config: DictConfig,
) -> None:
    return pipeline(config)


if __name__ == "__main__":
    main()
