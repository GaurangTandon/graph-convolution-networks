from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from torch import IntTensor, Tensor

# device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
device_name = 'cpu' # since we have for loops in our training code
device = torch.device(device_name)

class MessagePassing(torch.nn.Module, ABC):
    """
    This is the neighborhood aggregation or message passing scheme.
    See formula here: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html

    We cannot build adjacency matrix because of memory overflow error
    """
    def __init__(self, layers: int, input_dim: int, latent_dim: int, output_dim: int, norm_type: str, is_rnn: bool=False) -> None:
        """
        We must take only these dimensions as our input and train on them
        This is so that if tomorrow our graph adds more nodes to itself, then we should
        still be able to train on it
        """
        super().__init__()
        self.layer_count = layers
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.neighbours_aggregated: Optional[Tensor] = None
        self.neighbour_count: Optional[Tensor] = None
        self.norm_type = norm_type
        self.is_rnn = is_rnn

        self.initialization()

    @abstractmethod
    def initialization(self):
        pass

    @abstractmethod
    def combine(self, old_feature: Tensor, neighbor_aggregated: Tensor, layer_idx: int) -> Tensor:
        pass

    @abstractmethod
    def aggregation(self, neighbor_features: Tensor) -> Tensor:
        pass

    @abstractmethod
    def output(self, feature: Tensor) -> Tensor:
        pass


    def forward(self, features: Tensor, edge_list: IntTensor) -> Tensor:
        """
        features = Node features/the X tensor
        Edge list = list of edges (from a to b)
        """
 
        features = self.encoder(features) 
        adder = (1 if self.is_rnn else 0)
        node_count = features.shape[0]
        degree: List[int] = [0 for _ in range(node_count)]

        def get_normalization_factor(degree_list: List[int], x: int, y: int):
            # +1 because many nodes have degree 0 in citeseer dataset
            dx = degree_list[x] + 1
            dy = degree_list[y] + 1
            if self.norm_type == 'row':
                return dx
            elif self.norm_type == 'col':
                return dy
            elif self.norm_type == 'symmetric':
                return (dx * dy) ** 0.5
            else:
                assert self.norm_type == 'none'
                return 1

        neighbours_list: List[List[int]] = [[] for _ in range(node_count)]
        for x, y in edge_list:
            neighbours_list[y.item()].append(x.item())
            degree[y] += 1

        for layer_idx in range(self.layer_count):
            new_features: List[Tensor] = []

            for node_idx in range(node_count):
                neighbors = neighbours_list[node_idx]
                neighbour_features = torch.stack([features[y] / get_normalization_factor(degree, node_idx, y) for y in neighbors], dim=0) if len(neighbors) > 0 else torch.zeros(1, self.latent_dim + adder, device=device)
                aggregated = self.aggregation(neighbour_features)
                new_features.append(self.combine(features[node_idx], aggregated, layer_idx))
            features = torch.stack(new_features)
        
        return self.output(features)

