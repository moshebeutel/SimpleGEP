from typing import Optional
import torch
from sklearn.decomposition import KernelPCA
from simplegep.embeddings.embedder import Embedder


class KernelPCAEmbedder(Embedder):
    def __init__(self, num_basis_elements: int, kernel_type: str = 'rbf', gamma: Optional[float] = None):
        super(KernelPCAEmbedder).__init__()
        self._num_basis_elements = num_basis_elements
        self._kernel_type = kernel_type
        self._gamma = gamma
        self._kernel_pca = KernelPCA(n_components=self._num_basis_elements, kernel=self._kernel_type, gamma=self._gamma,
                                     fit_inverse_transform=True)

    def calc_embedding_space(self, data: torch.Tensor):
        data_np = data.detach().cpu().numpy()
        self._kernel_pca.fit(data_np)

    def embed(self, src: torch.Tensor) -> torch.Tensor:
        src_np = src.detach().cpu().numpy()
        embedding: torch.Tensor = torch.from_numpy(self._kernel_pca.transform(src_np))
        return embedding

    def project_back(self, embedding: torch.Tensor) -> torch.Tensor:
        embedding_np = embedding.detach().cpu().numpy()
        reconstructed: torch.Tensor = torch.from_numpy(self._kernel_pca.inverse_transform(embedding_np))
        return reconstructed


# if __name__ == '__main__':

