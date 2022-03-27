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
        self.decoder = Linear(self.latent_dim, self.output_dim, device=device)

        self.weights = ModuleList([Linear(self.latent_dim, self.latent_dim) for _ in range(self.layer_count)])
    
    def encoder(self, features: Tensor):
        features = self.encoder_method(features)
        # features: (N, MAXLEN, LATENTDIM)
        add1 = torch.zeros(features.shape[0], 1, self.latent_dim)
        features = torch.cat((add1, features), dim=1)
        add2 = torch.zeros(features.shape[0], features.shape[1], 1)
        features = torch.cat((add2, features), dim=2)
        # features: (N, MAXLEN + 1, LATENTDIM + 1)
        features[:, 0, 0] = 1
        features[:, :, 0] = 0
        return features

    def aggregation(self, neighbor_features: Tensor) -> Tensor:
        return neighbor_features
    
    def combine(self, old_feature: Tensor, neighbor_aggregated: Tensor, layer_idx: int) -> Tensor:
        """
        Acts like a shift register, shifting information from right neighbor to left neighbor
        """
        # print(old_feature.shape): MAXLEN+1, LATENTDIM+1
        if old_feature[-1].item() == 1:
            new_feature: Tensor = (self.weights[layer_idx](old_feature[:-1]) + neighbor_aggregated[:-1])
            new_feature = new_feature.unsqueeze(1)
            new_feature[-1] = 1
            return new_feature
        else:
            return neighbor_aggregated

    def output(self, feature: Tensor) -> Tensor:
        return softmax(self.decoder(feature), dim=1)

    # need to override forward because different signature
    def forward(self, features: Tensor):
        edge_index = IntTensor([[i, i + 1] for i in range(features.shape[0])])
        self.layers = len(features)
        return super().forward(features, edge_index)