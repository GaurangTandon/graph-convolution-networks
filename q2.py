import numpy as np
# import torch

DIM = 3
N = 3 # zero indexed nodes
ITERS = 3

def relu(vec):
    vec = np.copy(vec)
    vec[vec < 0] = 0
    return vec

def f(vec):
    return relu(vec)

adjmatrix = np.zeros((N, N), dtype=np.bool8)

def singlestep(node_vectors_previous: np.ndarray):
    node_vectors = np.zeros((N, DIM))
    W = np.random.random((DIM, DIM))
    B = np.random.random((DIM, DIM))

    for node_idx in range(N):
        pooled_neighbours = adjmatrix[node_idx]
        average_pool = node_vectors[pooled_neighbours].sum(axis=0)
        my_previous_vector = node_vectors_previous[node_idx]
        assert average_pool.shape == my_previous_vector.shape
        assert W.shape == (DIM, DIM)
        assert B.shape == (DIM, DIM)
        assert average_pool.shape == (DIM,)
        assert my_previous_vector.shape == (DIM,)
        neighbor_pool = W @ average_pool
        my_bias = B @ my_previous_vector
        assert neighbor_pool.shape == (DIM,)
        assert my_bias.shape == (DIM,)
        node_vectors[node_idx] = f(neighbor_pool + my_bias)

    return node_vectors

def main():
    adjmatrix[0][1] = 1
    adjmatrix[1][2] = 1

    node_vectors = np.random.random((N, DIM))
    print("Stage 0", node_vectors)
    for i in range(ITERS):
        node_vectors = singlestep(node_vectors)
        print(f"Stage {i + 1}", node_vectors)

if __name__ == "__main__":
    main()