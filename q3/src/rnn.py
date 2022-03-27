from src.message_passing import MessagePassing

from torch import Tensor
from torch.nn import Linear, ModuleList
from torch.nn.functional import softmax

from src.message_passing import device

class GraphRNN(MessagePassing):
    def initialization(self):
        self.encoder_method = Linear(self.input_dim, self.latent_dim, device=device)
        self.decoder = Linear(self.latent_dim, self.output_dim, device=device)

        self.weights = ModuleList([Linear(self.latent_dim, self.latent_dim) for _ in range(self.layer_count)])
    
    def encoder(self, features: Tensor):
        features = self.encoder_method(features).unsqueeze(-1)
        features[0, -1] = 1
        features[:, -1] = 0
        return features

    def aggregation(self, neighbor_features: Tensor) -> Tensor:
        return neighbor_features
    
    def combine(self, old_feature: Tensor, neighbor_aggregated: Tensor, layer_idx: int) -> Tensor:
        """
        Acts like a shift register, shifting information from right neighbor to left neighbor
        """
        if old_feature[-1] == 1:
            new_feature: Tensor = (self.weights[layer_idx](old_feature[:-1]) + neighbor_aggregated[:-1])
            new_feature = new_feature.unsqueeze(1)
            new_feature[-1] = 1
            return new_feature
        else:
            return neighbor_aggregated

    def output(self, feature: Tensor) -> Tensor:
        return softmax(self.decoder(feature), dim=1)