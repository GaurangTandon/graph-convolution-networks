from typing import List, Tuple

import torch

from src.message_passing import MessagePassing
from src.dataloader import get_citeseer_dataset

def training(node_count: int, edge_list: List[Tuple[int, int]], node_labels: List[int], epochs=1):
    model = MessagePassing(node_count, edge_list)

    for epoch in range(epochs):
        output = model()
    
if __name__ == "__main__":
    edge_list, node_labels = get_citeseer_dataset()
    node_count = len(node_labels)
    training(node_count, edge_list, node_labels)
