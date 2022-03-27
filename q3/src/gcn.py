from typing import Optional

import torch
from torch import Tensor
from torch.nn.functional import softmax, relu

from src.message_passing import MessagePassing, device


class CiteSeerGCN(MessagePassing):
    def initialization(self):
        self.f = relu
        self.W = torch.nn.ModuleList([torch.nn.Linear(self.latent_dim, self.latent_dim, bias=False, device=device) for _ in range(self.layer_count)])
        self.B = torch.nn.ModuleList([torch.nn.Linear(self.latent_dim, self.latent_dim, bias=False, device=device) for _ in range(self.layer_count)])

        self.neighbours_aggregated: Optional[Tensor] = None
        self.neighbour_count: Optional[Tensor] = None

        self.encoder = torch.nn.Linear(self.input_dim, self.latent_dim, device=device)
        self.decoder = torch.nn.Linear(self.latent_dim, self.output_dim, device=device)
    
    def combine(self, old_feature: Tensor, neighbor_aggregated: Tensor, layer_idx: int) -> Tensor:
        my_thing = self.W[layer_idx](neighbor_aggregated)
        neighbor_thing = self.B[layer_idx](old_feature)
        return self.f(my_thing + neighbor_thing)
        
    def aggregation(self, neighbor_features: Tensor):
        return neighbor_features.mean(dim=0)
    
    def output(self, feature: Tensor):
        return softmax(self.decoder(feature), dim=1)
