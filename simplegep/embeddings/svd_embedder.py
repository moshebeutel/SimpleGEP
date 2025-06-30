import torch
from simplegep.embeddings.embedder import Embedder
from simplegep.embeddings.utils import normalize_return_transform


class SVDEmbedder(Embedder):
    def __init__(self, num_basis_elements: int):
        super(SVDEmbedder).__init__()
        self._num_basis_elements = num_basis_elements
        self._basis_elements = None
        self._U = None
        self._S = None
        self._V = None
        self._center = None
        self._scale = None



    def calc_embedding_space(self, data: torch.Tensor):
        normalized_x, self._center, self._scale = normalize_return_transform(data)
        self._U, self._S, Vh = torch.linalg.svd(normalized_x, full_matrices=False)
        self._V = Vh.mH
        # self._basis_elements = self._U[:, :self._num_basis_elements]
        self._basis_elements = self._V[:, :self._num_basis_elements]

    def embed(self, src: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=False):
            normalized = (src - self._center) / self._scale
            embedding: torch.Tensor = torch.matmul(normalized, self._basis_elements)
        assert torch.all(torch.isnan(embedding) == False), 'SVD embedding contains NaNs'
        return embedding

    def project_back(self, embedding: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast('cuda', enabled=False):
            reconstructed = torch.matmul(embedding, self._basis_elements.t())
        reconstructed_transformed = (reconstructed * self._scale) + self._center
        return reconstructed_transformed

