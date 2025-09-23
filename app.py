import dotenv

dotenv.load_dotenv(
    override=True,
)

import os
import warnings

os.environ["HYDRA_FULL_ERROR"] = "1"
warnings.filterwarnings("ignore")

import sys

from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

from src.pipelines import app_pipeline


def main(
    config_path: str,
    job_name: str,
    config_name: str,
) -> None:
    def initialize_hydra():
        if GlobalHydra().is_initialized():
            GlobalHydra().clear()
        initialize(
            config_path=config_path,
            job_name=job_name,
        )

    initialize_hydra()
    config = compose(
        config_name=config_name,
        overrides=sys.argv[1:],
    )
    return app_pipeline(config)


if __name__ == "__main__":
    CONFIG_PATH = "configs/"
    JOB_NAME = "streamlit_demo"
    CONFIG_NAME = "main.yaml"
    main(
        config_path=CONFIG_PATH,
        job_name=JOB_NAME,
        config_name=CONFIG_NAME,
    )
