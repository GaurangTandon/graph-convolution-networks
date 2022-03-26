from typing import Dict, Tuple, List
from sklearn import preprocessing

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

def get_citeseer_dataset() -> Tuple[List[Tuple[int, int]], List[int]]:
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
    
if __name__ == "__main__":
    edge_list, node_labels = get_citeseer_dataset()
    print(edge_list[:10])
    print(node_labels[:10])