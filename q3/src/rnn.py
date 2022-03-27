import torch
from torch import Tensor, IntTensor
from torch.nn import Linear, ModuleList
from torch.nn.functional import softmax

from src.message_passing import device
from src.dataloader import MAXLEN
from src.message_passing import MessagePassing


class GraphRNN(MessagePassing):
    def __init__(self, layers: int, input_dim: int, latent_dim: int, output_dim: int, norm_type: str) -> None:
        super().__init__(layers, input_dim, latent_dim, output_dim, norm_type, is_rnn=True)

    def initialization(self):
        self.encoder_method = Linear(self.input_dim, self.latent_dim, device=device)
        self.decoder = Linear(self.latent_dim + 1, self.output_dim, device=device)

        self.weights = ModuleList([Linear(self.latent_dim + 1, self.latent_dim + 1) for _ in range(self.layer_count)])
    
    def encoder(self, features: Tensor):
        features = self.encoder_method(features)
        # features: (MAXLEN, LATENTDIM)
        add1 = torch.zeros(1, self.latent_dim)
        features = torch.cat((add1, features), dim=0)
        add2 = torch.zeros(features.shape[0], 1)
        features = torch.cat((add2, features), dim=1)
        # features: (MAXLEN + 1, LATENTDIM + 1)
        features[:][0] = 0
        features[0][0] = 1
        return features

    def aggregation(self, neighbor_features: Tensor) -> Tensor:
        return neighbor_features
    
    def combine(self, old_feature: Tensor, neighbor_aggregated: Tensor, layer_idx: int) -> Tensor:
        """
        Acts like a shift register, shifting information from right neighbor to left neighbor
        """
        old_feature = old_feature.reshape(-1)
        # print(old_feature.shape) # LATENTDIM+1
        if old_feature[0].item() == 1:
            # print(old_feature.shape, old_feature[1:].shape)
            new_feature: Tensor = (self.weights[layer_idx](old_feature) + neighbor_aggregated)
            # add1 = torch.zeros(1)
            # new_feature = torch.cat(new_feature, add1, dim=0)
            new_feature[0] = 1
            # print(new_feature.shape, old_feature.shape)
            return new_feature.reshape(-1)
        else:
            return neighbor_aggregated.reshape(-1)

    def output(self, feature: Tensor) -> Tensor:
        return softmax(self.decoder(feature[0]))

    # need to override forward because different signature
    def forward(self, features: Tensor):
        edge_index = IntTensor([[i, i + 1] for i in range(features.shape[0])])
        self.layers = len(features)
        return super().forward(features, edge_index)