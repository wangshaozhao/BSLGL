import torch
import pandas as pd
from torch_geometric.data import Data
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

def get_raw_text_arxiv_2023(feature_type='orig', use_text=False, seed=0):
    # Load graph structure data (assuming graph.pt contains edge_index and y)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    graph_data = torch.load(os.path.join(project_root,'dataset','arxiv_2023','graph.pt'))
    edge_index = graph_data['edge_index']
    y = graph_data['y']
    num_nodes = y.shape[0]

    # Split train/val/test sets
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Simple split example (adjust according to actual data)
    train_mask[:int(0.6 * num_nodes)] = True
    val_mask[int(0.6 * num_nodes):int(0.8 * num_nodes)] = True
    test_mask[int(0.8 * num_nodes):] = True

    data = Data(
        edge_index=edge_index,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask
    )

    if not use_text:
        return data, None

    # Load original text (title and abstract)
    df = pd.read_csv(os.path.join(project_root, 'dataset', 'arxiv_2023_orig', 'paper_info.csv'))
    text = []
    for ti, ab in zip(df['title'], df['abstract']):
        text.append(f'Title: {ti}\nAbstract: {ab}')
        # text.append((ti, ab))
    if feature_type == 'orig':
        return data, text
    if feature_type == 'enhanced':
        # Example usage
        csv_path = os.path.join(project_root, 'enhanced_texts','enhanced_arxiv_2023.csv')
        text = read_enhanced_text_column(csv_path)
        return data, text
    if feature_type == 'enhanced_impartial':
        csv_path = os.path.join(project_root, 'enhanced_texts','enhanced_arxiv_2023.csv')
        text = read_enhanced_text_column_1(csv_path)
        return data, text
    if feature_type == 'enhanced_smoothing':
        csv_path = os.path.join(project_root, 'enhanced_texts','enhanced_arxiv_2023.csv')
        text = read_enhanced_text_column_2(csv_path)
        return data, text

if __name__ == "__main__":
    data, text = get_raw_text_arxiv_2023('enhanced', True)
    print(data)
    print(len(data.y))
    print(len(text))
    print(text[3256])
