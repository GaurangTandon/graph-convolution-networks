import numpy as np
# import torch

DIM = 3
N = 4 # zero indexed nodes
ITERS = 3

def relu(vec):
    vec = np.copy(vec)
    vec[vec < 0] = 0
    return vec

def f(vec):
    return relu(vec)

adjmatrix = np.zeros((N, N), dtype=np.bool8)

def singlestep(node_vectors_previous: np.ndarray, stage: int=0):
    node_vectors = np.zeros((N, DIM))
    W = np.random.random((DIM, DIM))
    B = np.random.random((DIM, DIM))

    print(fr"\subsection{{Stage {{stage}}\n-----\nW={W}\nB={B}\n")
    for node_idx in range(N):
        pooled_neighbours = adjmatrix[node_idx]
        average_pool = node_vectors_previous[pooled_neighbours].sum(axis=0)
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
        print(f"For node {node_idx}, neighbour pool: {neighbor_pool}; my bias: {my_bias}; new vector={node_vectors[node_idx]}")
    
    print(f"Final result\n{node_vectors}")

    return node_vectors

def main():
    adjmatrix[0][1] = 1
    adjmatrix[1][2] = 1
    adjmatrix[2][3] = 1

    node_vectors = np.random.random((N, DIM))
    print("Stage 0", node_vectors)
    for stage in range(1, ITERS + 1):
        node_vectors = singlestep(node_vectors, stage=stage)

if __name__ == "__main__":
    main()