from simplegep.embeddings.kernel_pca_embedder import KernelPCAEmbedder
from simplegep.embeddings.svd_embedder import SVDEmbedder

embedder_hub = {'svd': SVDEmbedder, 'kernel_pca': KernelPCAEmbedder}


def get_embedder(args):
    assert args.embedder in embedder_hub, f"Embedder {args.embedder} not found. Expected one of {embedder_hub.keys()}."
    ctor_func = embedder_hub[args.embedder]
    embedder = ctor_func(args.num_basis, args.kernel_type) if args.embedder == 'kernel_pca' else ctor_func(args.num_basis)
    return embedder