from argparse import ArgumentParser
from typing import Optional, List

import torch
from torch import Tensor
from torch.nn.functional import softmax, relu, cross_entropy

from tqdm.auto import trange

from src.message_passing import MessagePassing, device, device_name
from src.dataloader import get_citeseer_dataset


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

def training(node_features: Tensor, edge_list: Tensor, labels: Tensor, train_mask: Tensor, val_mask: Tensor, epochs: int=1):
    # node_count = node_features.shape[0]
    input_dim = node_features.shape[1]
    FEATURE_DIM = 16
    model = CiteSeerGCN(layers=2, input_dim=input_dim, latent_dim=FEATURE_DIM, output_dim=6)
    optimizer = torch.optim.Adam(params=model.parameters())
    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accuracies: List[float] = []
    val_accuracies: List[float] = []

    with trange(epochs, desc="Running training") as it:
        for _epoch_idx in it:
            model.train()
            optimizer.zero_grad()
            output = model(node_features, edge_list)
            output_relevant = output[train_mask]
            label_relevant = labels[train_mask]

            train_accuracy = torch.mean(
                torch.eq(torch.argmax(output_relevant, dim=1), label_relevant).float()
            ).item()

            train_loss = cross_entropy(output_relevant, label_relevant)
            train_loss.backward()
            optimizer.step()

            train_losses.append(train_loss.item())

            with torch.no_grad():
                model.eval()

                output = model(node_features, edge_list)
                output_relevant = output[val_mask]
                label_relevant = labels[val_mask]

                val_accuracy = torch.mean(
                    torch.eq(torch.argmax(output_relevant, dim=1), label_relevant).float()
                ).item()
                val_accuracies.append(val_accuracy)

                val_loss = cross_entropy(output_relevant, label_relevant).item()
                val_losses.append(val_loss)

            it.set_postfix({ "val_loss": val_loss, "train_loss": train_loss.item(), "val_accuracy": val_accuracy, "train_accuracy": train_accuracy })
        
    return val_losses, train_losses, val_accuracies, train_accuracies

def main():
    args = ArgumentParser("TRAINER AAAAA")
    args.add_argument("--epochs", type=int)
    options = args.parse_args()

    dataset = get_citeseer_dataset()
    dataset.download()
    df = dataset.data
    df = df.to(device=device_name)
    # These two are the correct ways to get the data
    # Do not use .get(idx), we don't know what it does
    # print(dataset.data)
    # print(dataset[0])

    features = df.x
    edges: Tensor = df.edge_index.T
    labels = df.y

    # edge_list = [(x, y) for x, y in edges]
    # print(edge_list[:10])

    print(training(features, edges, labels, df.train_mask, df.val_mask, epochs=options.epochs))
        
if __name__ == "__main__":
    main()