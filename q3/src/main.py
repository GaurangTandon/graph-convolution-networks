from argparse import ArgumentParser
from typing import List

import torch
from torch import Tensor
from torch.nn.functional import cross_entropy
from tqdm.auto import trange

from src.dataloader import get_citeseer_dataset
from src.gcn import CiteSeerGCN

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
    args = ArgumentParser("GraphML Toolkit")
    args.add_argument("--task", required=True, choices=['gcn'])
    args.add_argument("--epochs", type=int, default=10)
    options = args.parse_args()

    df = get_citeseer_dataset()

    if options.task == 'gcn':
        print(training(df.x, df.edge_index.T, df.y, df.train_mask, df.val_mask, epochs=options.epochs))
    elif options.task == 'gin':
        pass
    elif options.task == 'rnn':
        pass
        
if __name__ == "__main__":
    main()