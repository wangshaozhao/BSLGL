import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import os
import csv


# return cora dataset as pytorch geometric Data object together with 60/20/20 split, and list of cora IDs
def get_cora_casestudy(SEED=0):
    data_X, data_Y, data_citeid, data_edges = parse_cora()
    # data_X = sklearn.preprocessing.normalize(data_X, norm="l1")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.

    # load data
    data_name = 'cora'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset = Planetoid('dataset', data_name,
                        transform=T.NormalizeFeatures())
    data = dataset[0]

    data.x = torch.tensor(data_X).float()
    data.edge_index = torch.tensor(data_edges).long()
    data.y = torch.tensor(data_Y).long()
    data.num_nodes = len(data_Y)

    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
    data.val_id = np.sort(
        node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(data.num_nodes * 0.8):])

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(data.num_nodes)])

    return data, data_citeid


def parse_cora():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path1 = os.path.join(project_root, 'dataset', 'cora_orig', 'cora.content')
    #path = 'dataset/cora_orig/cora'
    idx_features_labels = np.genfromtxt(
        path1, dtype=np.dtype(str))
    data_X = idx_features_labels[:, 1:-1].astype(np.float32)
    labels = idx_features_labels[:, -1]
    class_map = {x: i for i, x in enumerate(['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                                            'Probabilistic_Methods', 'Reinforcement_Learning', 'Rule_Learning', 'Theory'])}
    data_Y = np.array([class_map[l] for l in labels])
    data_citeid = idx_features_labels[:, 0]
    idx = np.array(data_citeid, dtype=np.dtype(str))
    idx_map = {j: i for i, j in enumerate(idx)}
    path2 = os.path.join(project_root, 'dataset', 'cora_orig', 'cora.cites')
    edges_unordered = np.genfromtxt(
        path2, dtype=np.dtype(str))
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(
        edges_unordered.shape)
    data_edges = np.array(edges[~(edges == None).max(1)], dtype='int')
    data_edges = np.vstack((data_edges, np.fliplr(data_edges)))
    return data_X, data_Y, data_citeid, np.unique(data_edges, axis=0).transpose()


import os
import csv

def read_enhanced_text_column(csv_file_path):
    enhanced_texts = []
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            enhanced_texts.append(row['enhanced_text'])
    return enhanced_texts


def read_enhanced_text_column_1(csv_file_path):
    """
    Read enhanced text from CSV file and extract "Title...Abstract..." part
    """
    enhanced_texts = []
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        # Verify required fields exist
        required_fields = {'enhanced_text'}
        if not required_fields.issubset(reader.fieldnames):
            missing = required_fields - set(reader.fieldnames)
            raise ValueError(f"CSV file missing required fields: {missing}")
        for idx, row in enumerate(reader):
            try:
                full_text = row['enhanced_text']
                # Extract Title and Abstract parts
                title_end = full_text.find('Abstract:')
                if title_end == -1:
                    raise ValueError(f"Row {idx} missing 'Abstract:' marker")
                # Extract up to Abstract end (assuming Keywords come after Abstract)
                keywords_start = full_text.find('Keywords:', title_end)
                if keywords_start != -1:
                    extracted = full_text[:keywords_start].strip()
                else:
                    extracted = full_text.strip()

                enhanced_texts.append(extracted)
            except Exception as e:
                raise ValueError(f"Error processing row {idx}: {str(e)}") from e

    return enhanced_texts


def read_enhanced_text_column_2(csv_file_path):
    """
    Read enhanced text from CSV file and extract "Title...Keywords..." part
    """
    enhanced_texts = []

    # Verify file exists
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"File not found: {csv_file_path}")

    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        # Verify required fields exist
        required_fields = {'enhanced_text'}
        if not required_fields.issubset(reader.fieldnames):
            missing = required_fields - set(reader.fieldnames)
            raise ValueError(f"CSV file missing required fields: {missing}")

        for idx, row in enumerate(reader):
            try:
                full_text = row['enhanced_text']

                # Extract Title to Keywords part
                title_end = full_text.find('Abstract:')
                keywords_start = full_text.find('Keywords:', title_end if title_end != -1 else 0)

                if keywords_start == -1:
                    raise ValueError(f"Row {idx} missing 'Keywords:' marker")

                # Skip Abstract part, connect Title and Keywords
                if title_end != -1:
                    title_part = full_text[:title_end].strip()
                    keywords_part = full_text[keywords_start:].strip()
                    extracted = f"{title_part}\n{keywords_part}"
                else:
                    # If no Abstract, take to end of file
                    extracted = full_text.strip()

                enhanced_texts.append(extracted)
            except Exception as e:
                raise ValueError(f"Error processing row {idx}: {str(e)}") from e
    return enhanced_texts

def get_raw_text_cora(feature_type='orig', use_text=False, seed=0):
    data, data_citeid = get_cora_casestudy(seed)
    if not use_text:
        return data, None

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path1 = os.path.join(project_root, 'dataset', 'cora_orig', 'mccallum', 'cora', 'papers')
    with open(path1)as f:
        lines = f.readlines()
    pid_filename = {}
    for line in lines:
        pid = line.split('\t')[0]
        fn = line.split('\t')[1]
        fn = fn.replace(':','_')
        pid_filename[pid] = fn

    path2 = os.path.join(project_root, 'dataset', 'cora_orig', 'mccallum', 'cora', 'extractions')
    text = []
    for pid in data_citeid:
        fn = pid_filename[pid]
        path = os.path.join(path2, fn)
        with open(path) as f:
            lines = f.read().splitlines()

        for line in lines:
            if 'Title:' in line:
                ti = line
            if 'Abstract:' in line:
                ab = line
        text.append(ti+'\n'+ab)
    if feature_type == 'orig':
        return data, text
    if feature_type == 'enhanced':
        csv_path = os.path.join(project_root, 'enhanced_texts','enhanced_cora.csv')
        text = read_enhanced_text_column(csv_path)
        return data, text
    if feature_type == 'enhanced_impartial':
        csv_path = os.path.join(project_root, 'enhanced_texts','enhanced_cora.csv')
        text = read_enhanced_text_column_1(csv_path)
        return data, text
    if feature_type == 'enhanced_smoothing':
        csv_path = os.path.join(project_root, 'enhanced_texts','enhanced_cora.csv')
        text = read_enhanced_text_column_2(csv_path)
        return data, text


if __name__ == "__main__":
    pass
