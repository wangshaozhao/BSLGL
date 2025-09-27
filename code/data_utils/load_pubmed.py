import numpy as np
# adapted from https://github.com/jcatw/scnn
import torch
import random

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from sklearn.preprocessing import normalize, StandardScaler
import json
import pandas as pd

# return pubmed dataset as pytorch geometric Data object together with 60/20/20 split, and list of pubmed IDs
def get_pubmed_casestudy(corrected=False, SEED=0):
    _, data_X, data_Y, data_pubid, data_edges = parse_pubmed()
    data_X = normalize(data_X, norm="l1")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)  # Numpy module.
    random.seed(SEED)  # Python random module.

    # load data
    data_name = 'PubMed'
    # path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset = Planetoid('dataset', data_name, transform=T.NormalizeFeatures())
    data = dataset[0]

    # replace dataset matrices with the PubMed-Diabetes data, for which we have the original pubmed IDs
    data.x = torch.tensor(data_X)
    data.edge_index = torch.tensor(data_edges)
    data.y = torch.tensor(data_Y)

    # split data
    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)

    data.train_id = np.sort(node_id[:int(data.num_nodes * 0.6)])
    data.val_id = np.sort(
        node_id[int(data.num_nodes * 0.6):int(data.num_nodes * 0.8)])
    data.test_id = np.sort(node_id[int(data.num_nodes * 0.8):])

    if corrected:
        is_mistake = np.loadtxt(
            'pubmed_casestudy/pubmed_mistake.txt', dtype='bool')
        data.train_id = [i for i in data.train_id if not is_mistake[i]]
        data.val_id = [i for i in data.val_id if not is_mistake[i]]
        data.test_id = [i for i in data.test_id if not is_mistake[i]]

    data.train_mask = torch.tensor(
        [x in data.train_id for x in range(data.num_nodes)])
    data.val_mask = torch.tensor(
        [x in data.val_id for x in range(data.num_nodes)])
    data.test_mask = torch.tensor(
        [x in data.test_id for x in range(data.num_nodes)])

    return data, data_pubid


def parse_pubmed():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path1 = os.path.join(project_root, 'dataset', 'PubMed_orig', 'data','Pubmed-Diabetes.NODE.paper.tab')

    n_nodes = 19717
    n_features = 500

    data_X = np.zeros((n_nodes, n_features), dtype='float32')
    data_Y = [None] * n_nodes
    data_pubid = [None] * n_nodes
    data_edges = []

    paper_to_index = {}
    feature_to_index = {}

    # parse nodes
    with open(path1, 'r') as node_file:
        # first two lines are headers
        node_file.readline()
        node_file.readline()

        k = 0

        for i, line in enumerate(node_file.readlines()):
            items = line.strip().split('\t')

            paper_id = items[0]
            data_pubid[i] = paper_id
            paper_to_index[paper_id] = i

            # label=[1,2,3]
            label = int(items[1].split('=')[-1]) - \
                1  # subtract 1 to zero-count
            data_Y[i] = label

            # f1=val1 \t f2=val2 \t ... \t fn=valn summary=...
            features = items[2:-1]
            for feature in features:
                parts = feature.split('=')
                fname = parts[0]
                fvalue = float(parts[1])

                if fname not in feature_to_index:
                    feature_to_index[fname] = k
                    k += 1

                data_X[i, feature_to_index[fname]] = fvalue

    # parse graph
    data_A = np.zeros((n_nodes, n_nodes), dtype='float32')

    path2 = os.path.join(project_root, 'dataset', 'PubMed_orig', 'data','Pubmed-Diabetes.DIRECTED.cites.tab')
    with open(path2, 'r') as edge_file:
        # first two lines are headers
        edge_file.readline()
        edge_file.readline()

        for i, line in enumerate(edge_file.readlines()):

            # edge_id \t paper:tail \t | \t paper:head
            items = line.strip().split('\t')

            edge_id = items[0]

            tail = items[1].split(':')[-1]
            head = items[3].split(':')[-1]

            data_A[paper_to_index[tail], paper_to_index[head]] = 1.0
            data_A[paper_to_index[head], paper_to_index[tail]] = 1.0
            if head != tail:
                data_edges.append(
                    (paper_to_index[head], paper_to_index[tail]))
                data_edges.append(
                    (paper_to_index[tail], paper_to_index[head]))

    return data_A, data_X, data_Y, data_pubid, np.unique(data_edges, axis=0).transpose()

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

def get_raw_text_pubmed(feature_type='orig', use_text=False, seed=0):
    data, data_pubid = get_pubmed_casestudy(SEED=seed)
    if not use_text:
        return data, None

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(project_root, 'dataset', 'PubMed_orig', 'pubmed.json')

    f = open(path)
    pubmed = json.load(f)
    df_pubmed = pd.DataFrame.from_dict(pubmed)

    AB = df_pubmed['AB'].fillna("")
    TI = df_pubmed['TI'].fillna("")
    text = []
    for ti, ab in zip(TI, AB):
        t = 'Title: ' + ti + '\n'+'Abstract: ' + ab
        text.append(t)
    if feature_type == 'orig':
        return data, text
    if feature_type == 'enhanced':
        # Example usage
        csv_path = os.path.join(project_root, 'enhanced_texts','enhanced_pubmed.csv')
        text = read_enhanced_text_column(csv_path)
        return data, text
    if feature_type == 'enhanced_impartial':
        csv_path = os.path.join(project_root, 'enhanced_texts','enhanced_pubmed.csv')
        text = read_enhanced_text_column_1(csv_path)
        return data, text
    if feature_type == 'enhanced_smoothing':
        csv_path = os.path.join(project_root, 'enhanced_texts','enhanced_pubmed.csv')
        text = read_enhanced_text_column_2(csv_path)
        return data, text


if __name__ == "__main__":
    # Get dataset (without using text features)
    data, text = get_raw_text_pubmed("orig", False)

    # Define CSV file path
    file_path = r"C:\Users\Wang Shaozhao\Downloads\pubmed.csv"

    try:
        # Read predicted values from CSV file (first column, no column index)
        predictions = []
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                # Ensure each row has at least one element
                if row:
                    # Assume predicted values are integers
                    predictions.append(int(row[0]))

        # Verify number of predictions matches number of samples
        if len(predictions) != len(data.y):
            raise ValueError(f"Number of predictions ({len(predictions)}) doesn't match number of samples ({len(data.y)})")

        # Calculate accuracy
        correct = 0
        total = len(data.y)
        for pred, true in zip(predictions, data.y.numpy()):
            if pred == true:
                correct += 1

        accuracy = correct / total
        print(f"Prediction accuracy: {accuracy:.4f} ({correct}/{total})")

    except FileNotFoundError:
        print(f"Error: File not found {file_path}")
    except Exception as e:
        print(f"Error calculating accuracy: {str(e)}")
