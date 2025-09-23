from ..models import VllmGenerator


class ReportManager:
    def __init__(
        self,
        generator: VllmGenerator,
    ) -> None:
        self.generator = generator

    def generate(
        self,
        recommendations: str,
    ) -> str:
        report = self.generator(recommendations=recommendations)
        return report
