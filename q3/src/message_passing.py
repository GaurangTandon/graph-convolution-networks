import torch
from torch.nn.functional import relu
from typing import List, Tuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MessagePassing(torch.nn.Module):
    FEATURES_DIM = 10

    """
    This is the neighborhood aggregation or message passing scheme.
    See formula here: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html

    We cannot build adjacency matrix because of memory overflow error
    """
    def __init__(self, node_count: int, edge_list: List[Tuple[int, int]]) -> None:
        super().__init__()
        self.node_count = node_count
        self.node_features = torch.rand(node_count, self.FEATURES_DIM, dtype=torch.float64, device=device)
        self.edge_list = edge_list
        self.f = relu
        self.W = torch.nn.Linear(self.FEATURES_DIM, self.FEATURES_DIM, bias=False, device=device)
        self.B = torch.nn.Linear(self.FEATURES_DIM, self.FEATURES_DIM, bias=False, device=device)

    def forward(self, *input):
        node_features_previous = self.node_features.clone()
        neighbours_aggregated = torch.zeros(self.node_count, self.FEATURES_DIM, device=device)
        neighbour_count = torch.zeros((self.node_count,), device=device)

        # na to nb direction
        for na, nb in self.edge_list:
            neighbours_aggregated[nb] += node_features_previous[na]
            neighbour_count[nb] += 1

        for node_idx in range(self.node_count):
            aggregated = neighbours_aggregated[node_idx] / neighbour_count[node_idx]
            self.node_features[node_idx] = self.f(self.W(aggregated) + self.B(aggregated))

if __name__ == "__main__":
    mp = MessagePassing(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    mp.forward()
    print(mp.node_features)
    mp.forward()
    print(mp.node_features)
    mp.forward()
    print(mp.node_features)
