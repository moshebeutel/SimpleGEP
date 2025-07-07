from simplegep.embeddings.kernel_pca_embedder import KernelPCAEmbedder
from simplegep.embeddings.svd_embedder import SVDEmbedder

embedder_hub = {'svd': SVDEmbedder, 'kernel_pca': KernelPCAEmbedder}

def get_embedder(args):
    return embedder_hub[args.embedder](args.num_basis)