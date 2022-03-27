from pathlib import Path
from typing import Dict, Tuple, List

from torch_geometric.datasets import Planetoid

from src.message_passing import device_name

def read_nodes(dirpath: str='..\\citeseer'):
    node_id_to_idx: Dict[str, int] = {}
    label: List[str] = []
    with open(f'{dirpath}\\citeseer.content', 'r') as f:
        for idx, line in enumerate(f):
            data = line.strip().split('\t')
            node_id_to_idx[data[0]] = idx
            label.append(data[-1])
            
    return node_id_to_idx, label

def read_edges(node_mapping: Dict[str, int], dirpath: str='..\\citeseer'):
    edge_list: List[Tuple[int, int]] = []
    skipped_count = 0
    with open(f'{dirpath}\\citeseer.cites', 'r') as f:
        for line in f:
            data = line.strip().split('\t')
            if data[1] not in node_mapping or data[0] not in node_mapping:
                skipped_count += 1
                continue
            # edge direction from a to b
            edge_list.append((node_mapping[data[1]], node_mapping[data[0]]))
    print(skipped_count)
            
    return edge_list

def get_citeseer_dataset():
    """
    I wasted a lot of time writing my own data reader from text files
    before realizing this thing already exists in pytorch ._.
    I hate myself
    """
    path = Path.cwd().parent / "citeseer"
    pt = Planetoid(root=str(path), name="CiteSeer")
    pt.download()
    # These two are the correct ways to get the data
    # Do not use .get(idx), we don't know what it does
    # print(dataset.data)
    # print(dataset[0])
    return pt.data.to(device=device_name)

    # class_labels = [
    #     "Agents",
    #     "AI",
    #     "DB",
    #     "IR",
    #     "ML",
    #     "HCI"
    # ]
    le = preprocessing.LabelEncoder()
    node_id_maps, node_labels = read_nodes()
    node_labels = le.fit_transform(node_labels)
    edge_list = read_edges(node_id_maps)
    return edge_list, node_labels
    