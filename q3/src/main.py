from argparse import ArgumentParser
import random
from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn.functional import cross_entropy, one_hot, pad as torch_pad
from tqdm.auto import trange

from src.message_passing import device
from src.dataloader import get_citeseer_dataset, get_imdb_dataset, VOCAB_SIZE, MAXLEN
from src.gcn import CiteSeerGCN
from src.gin import CiteSeerGIN
from src.rnn import GraphRNN

def training(model: torch.nn.Module, node_features: Tensor, edge_list: Tensor, labels: Tensor, train_mask: Tensor, val_mask: Tensor, epochs: int=1):
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

def train_rnn(model: torch.nn.Module, train_data: Tuple[List[List[int]], List[int]], test_data: Tuple[List[List[int]], List[int]], epoch_count: int):
    """
    It requires a completely different loop because IMDB data is structured differently
    """

    optimizer = torch.optim.Adam(params=model.parameters())
    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accuracies: List[float] = []
    val_accuracies: List[float] = []

    train_samples, train_labels = train_data
    test_samples, test_labels = test_data
    def pad(lst: List[int], ml: int):
        return lst + ([0 for _ in range(ml - len(lst))] if ml >= len(lst) else [])
    def converter(samples: List[List[int]]):
        return torch.stack([torch.LongTensor(pad(x, MAXLEN), device=device) for x in samples])
    train_samples = converter(train_samples)
    test_samples = converter(test_samples)
    train_input = one_hot(train_samples, num_classes=VOCAB_SIZE).to(torch.float32)
    train_labels = torch.LongTensor(train_labels)
    test_labels = torch.LongTensor(test_labels)

    with trange(epoch_count) as epoch_iter:
        for epoch in epoch_iter:
            train_loss = 0

            model.train()
            optimizer.zero_grad()

            # (N, MAXLEN, VOCABSIZE)
            print(train_input.shape)
            # (N, )
            print(train_labels.shape)
            result = model(train_input)
            train_accuracy = torch.mean(torch.eq(torch.argmax(result, dim=1), train_labels).float()).item()
            train_accuracies.append(train_accuracy)
            train_loss += cross_entropy(result, train_labels)

            train_loss.backward()
            train_losses.append(train_loss)
            optimizer.step()

            # with torch.no_grad():
            #     model.eval()
            #     val_loss = 0

            #     for x, y in train_data:
            #         result = model(x)
            #         train_loss += 0

            #     val_losses.append(val_loss)
                
            epoch_iter.set_postfix({
                "train_loss": train_loss,
                "train_accuracy": train_accuracy
                # "val_loss": val_loss
            })
    
    return train_accuracies

def main():
    SEED = 42
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)

    args = ArgumentParser("GraphML Toolkit")
    args.add_argument("--task", required=True, choices=['gcn', 'compare', 'rnn'])
    args.add_argument("--epochs", type=int, default=10)
    args.add_argument("--norm_type", type=str, default='none', choices=['none', 'row', 'col', 'symmetric'])
    options = args.parse_args()

    # node_count = node_features.shape[0]
    FEATURE_DIM = 16

    result = None

    print("CLI", options)

    if options.task == 'gcn':
        citeseer_df = get_citeseer_dataset()
        citeseer_input_dim = citeseer_df.x.shape[1]
        model = CiteSeerGCN(layers=2, input_dim=citeseer_input_dim, latent_dim=FEATURE_DIM, output_dim=6, norm_type=options.norm_type)
        result = training(model, citeseer_df.x, citeseer_df.edge_index.T, citeseer_df.y, citeseer_df.train_mask, citeseer_df.val_mask, epochs=options.epochs)
    elif options.task == 'compare':
        citeseer_df = get_citeseer_dataset()
        citeseer_input_dim = citeseer_df.x.shape[1]
        model = CiteSeerGIN(layers=2, input_dim=citeseer_input_dim, latent_dim=FEATURE_DIM, output_dim=6, norm_type=options.norm_type)
        result = training(model, citeseer_df.x, citeseer_df.edge_index.T, citeseer_df.y, citeseer_df.train_mask, citeseer_df.val_mask, epochs=options.epochs)
    elif options.task == 'rnn':
        (x_train, y_train), (x_test, y_test) = get_imdb_dataset()
        model = GraphRNN(layers=2, input_dim=VOCAB_SIZE, latent_dim=64, output_dim=2, norm_type=options.norm_type)
        SAMPLE_COUNT = 100
        train_data = x_train[:SAMPLE_COUNT], y_train[:SAMPLE_COUNT]
        test_data = x_test[:SAMPLE_COUNT], y_test[:SAMPLE_COUNT]
        result = train_rnn(model, train_data, test_data, epoch_count=options.epochs)

    print(result)

if __name__ == "__main__":
    main()