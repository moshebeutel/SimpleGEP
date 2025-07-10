import torch


class GradsContainer:
    def __init__(self, container_size: int, grads_numel: int):
        assert container_size > 0, f'Expected container_size > 0, got {container_size}'
        assert grads_numel > 0, f'Expected grads_numel > 0, got {grads_numel}'

        self.container_size: int = container_size
        self.grads_numel: int = grads_numel
        self._current_size: int = 0
        self._grads: torch.Tensor or None = None

    def add(self, grad: torch.Tensor):
        assert grad.dim() == 2, f'Expected grads with dim==2, got {grad.dim()}'
        assert grad.shape[1] == self.grads_numel, f'Expected grads with shape[1]=={self.grads_numel}, got {grad.shape[1]}'

        added_grads = grad.shape[0]

        self._grads = grad if self._grads is None else torch.cat([self._grads, grad], dim=0)
        assert (self._current_size + added_grads) == self._grads.shape[0], f'Added {added_grads} grads, expected {self._current_size + added_grads} grads, got {self._grads.shape[0]}'

        if self._grads.shape[0] > self.container_size:
            self._grads = self._grads[-self.container_size:]

        self._current_size = self._grads.shape[0]

    @property
    def grads(self) -> torch.Tensor:
        return self._grads

    @property
    def current_size(self) -> int:
        return self._current_size

def create_grads_history_container(args, grads_numel: int):
    grads_history_container = GradsContainer(args.grads_history_size, grads_numel)
    return grads_history_container