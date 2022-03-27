import torch
from torch import Tensor
from torch.nn.functional import softmax, relu
from torch.nn import Linear, ModuleList, ParameterList

from src.message_passing import MessagePassing, device


class CiteSeerGIN(MessagePassing):
    def initialization(self):
        self.f = relu
        self.epsilon = ParameterList([torch.nn.parameter.Parameter(torch.Tensor([0.2]), requires_grad=True) for _ in range(self.layer_count)])
        self.mlp = ModuleList([Linear(self.latent_dim, self.latent_dim, device=device) for _ in range(self.layer_count)])

        self.encoder = Linear(self.input_dim, self.latent_dim, device=device)
        self.decoder = Linear(self.latent_dim, self.output_dim, device=device)
    
    def combine(self, old_feature: Tensor, neighbor_aggregated: Tensor, layer_idx: int) -> Tensor:
        combined = (1 + self.epsilon[layer_idx]) * old_feature + neighbor_aggregated
        return self.mlp[layer_idx](combined)
        
    def aggregation(self, neighbor_features: Tensor):
        return neighbor_features.sum(dim=0)
    
    def output(self, feature: Tensor):
        return softmax(self.decoder(feature), dim=1)

