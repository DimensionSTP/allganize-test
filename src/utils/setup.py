from typing import Union

from omegaconf import DictConfig
from hydra.utils import instantiate

from ..managers import *


class SetUp:
    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        self.config = config

    def get_manager(
        self,
        manager_type: str,
    ) -> Union[RecommendationManager, ReportManager]:
        manager: Union[RecommendationManager, ReportManager] = instantiate(
            self.config.manager[manager_type],
        )
        return manager
