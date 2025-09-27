import csv
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import pandas as pd
import json
import numpy as np
import os
import time
from core.utils import time_logger

FILE = 'dataset/ogbn_products_orig/ogbn-products.csv'


@time_logger
def _process():
    if os.path.isfile(FILE):
        return

    print("Processing raw text...")

    data = []
    files = ['dataset/ogbn_products/Amazon-3M.raw/trn.json',
             'dataset/ogbn_products/Amazon-3M.raw/tst.json']
    for file in files:
        with open(file) as f:
            for line in f:
                data.append(json.loads(line))

    df = pd.DataFrame(data)
    df.set_index('uid', inplace=True)

    nodeidx2asin = pd.read_csv(
        'dataset/ogbn_products/mapping/nodeidx2asin.csv.gz', compression='gzip')

    dataset = PygNodePropPredDataset(
        name='ogbn-products', transform=T.ToSparseTensor())
    graph = dataset[0]
    graph.n_id = np.arange(graph.num_nodes)
    graph.n_asin = nodeidx2asin.loc[graph.n_id]['asin'].values

    graph_df = df.loc[graph.n_asin]
    graph_df['nid'] = graph.n_id
    graph_df.reset_index(inplace=True)

    if not os.path.isdir('dataset/ogbn_products_orig'):
        os.mkdir('dataset/ogbn_products_orig')
    pd.DataFrame.to_csv(graph_df, FILE,
                        index=False, columns=['uid', 'nid', 'title', 'content'])

def read_enhanced_text_column(csv_file_path):
    enhanced_texts = []
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            enhanced_texts.append(row['enhanced_text'])
    return enhanced_texts

def read_enhanced_text_column_1(csv_file_path):
    """
    Read enhanced text from CSV file and extract "Product...Description..." part
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
                title_end = full_text.find('Product:')
                if title_end == -1:
                    raise ValueError(f"Row {idx} missing 'Product:' marker")
                # Extract up to Abstract end (assuming Keywords come after Product)
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

                # Extract Product to Keywords part
                title_end = full_text.find('Description:')
                keywords_start = full_text.find('Keywords:', title_end if title_end != -1 else 0)

                if keywords_start == -1:
                    raise ValueError(f"Row {idx} missing 'Keywords:' marker")

                # Skip Description part, connect Product and Keywords
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

def get_raw_text_products(feature_type='orig', use_text=False, seed=0):
    # Get project root directory (load_products.py is under core/data_utils/)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file_path1 = os.path.join(project_root, 'dataset', 'ogbn_products', 'ogbn-products_subset.pt')
    data = torch.load(file_path1)
    file_path2 = os.path.join(project_root, 'dataset', 'ogbn_products_orig', 'ogbn-products_subset.csv')
    text = pd.read_csv(file_path2)
    text = [f'Product:{ti}; Description: {cont}\n'for ti,
            cont in zip(text['title'], text['content'])]

    data.edge_index = data.adj_t.to_symmetric()

    if not use_text:
        return data, None

    if feature_type=='orig':
        return data, text
    if feature_type == 'enhanced':
        # Example usage
        csv_path = os.path.join(project_root, 'enhanced_texts','enhanced_products.csv')
        text = read_enhanced_text_column(csv_path)
        return data, text
    if feature_type == 'enhanced_impartial':
        csv_path = os.path.join(project_root, 'enhanced_texts','enhanced_products.csv')
        text = read_enhanced_text_column_1(csv_path)
        return data, text
    if feature_type == 'enhanced_smoothing':
        csv_path = os.path.join(project_root, 'enhanced_texts','enhanced_products.csv')
        text = read_enhanced_text_column_2(csv_path)
        return data, text


if __name__ == '__main__':
    data, text = get_raw_text_products('orig',True)
    print(len(data.y))
    print(text[0])

def read_enhanced_text_column_1(csv_file_path):
    """
    Read enhanced text from CSV file and extract "Product...Description..." part
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
                title_end = full_text.find('Product:')
                if title_end == -1:
                    raise ValueError(f"Row {idx} missing 'Product:' marker")
                # Extract up to Abstract end (assuming Keywords come after Product)
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

                # Extract Product to Keywords part
                title_end = full_text.find('Description:')
                keywords_start = full_text.find('Keywords:', title_end if title_end != -1 else 0)

                if keywords_start == -1:
                    raise ValueError(f"Row {idx} missing 'Keywords:' marker")

                # Skip Description part, connect Product and Keywords
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
