import csv

from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import pandas as pd
import os

def read_enhanced_text_column(csv_file_path):
    enhanced_texts = []
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            enhanced_texts.append(row['enhanced_text'])
    return enhanced_texts

def read_enhanced_text_column_1(csv_file_path):
    """
    Read enhanced text from CSV file, extract the "Title...Abstract..." part
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
                # Truncate to end of Abstract (assuming Keywords come after Abstract)
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
    Read enhanced text from CSV file, extract the "Title...Keywords..." part
    """
    enhanced_texts = []

    # Verify file exists
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"File does not exist: {csv_file_path}")

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

                # Extract from Title to Keywords
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
                    # If no Abstract, take all the way to end of file
                    extracted = full_text.strip()

                enhanced_texts.append(extracted)
            except Exception as e:
                raise ValueError(f"Error processing row {idx}: {str(e)}") from e
    return enhanced_texts

def get_raw_text_arxiv(feature_type='orig', use_text=False, seed=0):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataset_root = os.path.join(root_dir, "dataset")  # dataset folder in root directory

    dataset = PygNodePropPredDataset(
        name='ogbn-arxiv',
        root=dataset_root,
        transform=T.ToSparseTensor()
    )
    data = dataset[0]

    # Process dataset splits (train/validation/test sets)
    idx_splits = dataset.get_idx_split()
    train_mask = torch.zeros(data.num_nodes).bool()
    val_mask = torch.zeros(data.num_nodes).bool()
    test_mask = torch.zeros(data.num_nodes).bool()
    train_mask[idx_splits['train']] = True
    val_mask[idx_splits['valid']] = True
    test_mask[idx_splits['test']] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    data.edge_index = data.adj_t.to_symmetric()
    if not use_text:
        return data, None

    # Read local nodeidx2paperid mapping file (located in dataset/ogbn_arxiv/)
    nodeidx2paperid_path = os.path.join(
        dataset_root, "ogbn_arxiv", "mapping", "nodeidx2paperid.csv.gz"
    )
    nodeidx2paperid = pd.read_csv(
        nodeidx2paperid_path,
        compression='gzip'
    )

    # Read local titleabs.tsv file (located in dataset/ogbn_arxiv_orig/)
    titleabs_path = os.path.join(
        dataset_root, "ogbn_arxiv_orig", "titleabs.tsv"
    )
    raw_text = pd.read_csv(
        titleabs_path,
        sep='\t',
        header=None,
        names=['paper id', 'title', 'abs'],
        encoding='utf-8'
    )

    df = pd.merge(nodeidx2paperid, raw_text, on='paper id')
    text = []
    for ti, ab in zip(df['title'], df['abs']):
        t = f'Title: {ti}\nAbstract: {ab}'
        text.append(t)
        
    if feature_type == 'orig':
        return data, text
    if feature_type == 'enhanced':
        # Usage example
        csv_path = os.path.join(root_dir, 'enhanced_texts', 'enhanced_ogbn-arxiv.csv')
        text = read_enhanced_text_column(csv_path)
        return data, text
    if feature_type == 'enhanced_impartial':
        csv_path = os.path.join(root_dir, 'enhanced_texts', 'enhanced_ogbn-arxiv.csv')
        text = read_enhanced_text_column_1(csv_path)
        return data, text
    if feature_type == 'enhanced_smoothing':
        csv_path = os.path.join(root_dir, 'enhanced_texts', 'enhanced_ogbn-arxiv.csv')
        text = read_enhanced_text_column_2(csv_path)
        return data, text


if __name__ == "__main__":
    # Get dataset
    data, text = get_raw_text_arxiv("enhanced", True)
