from abc import ABC, abstractmethod
import torch


class Embedder(ABC):
    @abstractmethod
    def embed(self, src: torch.Tensor) -> torch.Tensor:
        pass
    @abstractmethod
    def calc_embedding_space(self, data: torch.Tensor):
        pass
    @abstractmethod
    def project_back(self, embedding: torch.Tensor) -> torch.Tensor:
        pass