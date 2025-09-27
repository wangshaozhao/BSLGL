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
    从CSV文件中读取增强文本，提取"Title...Abstract..."部分
    """
    enhanced_texts = []
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        # 校验必要字段是否存在
        required_fields = {'enhanced_text'}
        if not required_fields.issubset(reader.fieldnames):
            missing = required_fields - set(reader.fieldnames)
            raise ValueError(f"CSV文件缺少必要字段: {missing}")
        for idx, row in enumerate(reader):
            try:
                full_text = row['enhanced_text']
                # 提取Title和Abstract部分
                title_end = full_text.find('Abstract:')
                if title_end == -1:
                    raise ValueError(f"第{idx}行缺少'Abstract:'标记")
                # 截取到Abstract结束（假设Keywords在Abstract之后）
                keywords_start = full_text.find('Keywords:', title_end)
                if keywords_start != -1:
                    extracted = full_text[:keywords_start].strip()
                else:
                    extracted = full_text.strip()

                enhanced_texts.append(extracted)
            except Exception as e:
                raise ValueError(f"处理第{idx}行时出错: {str(e)}") from e

    return enhanced_texts


def read_enhanced_text_column_2(csv_file_path):
    """
    从CSV文件中读取增强文本，提取"Title...Keywords..."部分
    """
    enhanced_texts = []

    # 校验文件是否存在
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"文件不存在: {csv_file_path}")

    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        # 校验必要字段是否存在
        required_fields = {'enhanced_text'}
        if not required_fields.issubset(reader.fieldnames):
            missing = required_fields - set(reader.fieldnames)
            raise ValueError(f"CSV文件缺少必要字段: {missing}")

        for idx, row in enumerate(reader):
            try:
                full_text = row['enhanced_text']

                # 提取Title到Keywords部分
                title_end = full_text.find('Abstract:')
                keywords_start = full_text.find('Keywords:', title_end if title_end != -1 else 0)

                if keywords_start == -1:
                    raise ValueError(f"第{idx}行缺少'Keywords:'标记")

                # 跳过Abstract部分，连接Title和Keywords
                if title_end != -1:
                    title_part = full_text[:title_end].strip()
                    keywords_part = full_text[keywords_start:].strip()
                    extracted = f"{title_part}\n{keywords_part}"
                else:
                    # 如果没有Abstract，直接取到文件末尾
                    extracted = full_text.strip()

                enhanced_texts.append(extracted)
            except Exception as e:
                raise ValueError(f"处理第{idx}行时出错: {str(e)}") from e
    return enhanced_texts
def get_raw_text_arxiv(feature_type='orig',use_text=False, seed=0):

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataset_root = os.path.join(root_dir, "dataset")  # 根目录下的dataset文件夹

    dataset = PygNodePropPredDataset(
        name='ogbn-arxiv',
        root=dataset_root,
        transform=T.ToSparseTensor()
    )
    data = dataset[0]

    # 处理数据集分割（训练/验证/测试集）
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

    # 读取本地nodeidx2paperid映射文件（位于dataset/ogbn_arxiv/下）
    nodeidx2paperid_path = os.path.join(
        dataset_root, "ogbn_arxiv", "mapping", "nodeidx2paperid.csv.gz"
    )
    nodeidx2paperid = pd.read_csv(
        nodeidx2paperid_path,
        compression='gzip'
    )

    # 读取本地titleabs.tsv文件（位于dataset/ogbn_arxiv_orig/下）
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
        # 使用示例
        csv_path = os.path.join(root_dir, 'enhanced_texts','enhanced_ogbn-arxiv.csv')
        text = read_enhanced_text_column(csv_path)
        return data, text
    if feature_type == 'enhanced_impartial':
        csv_path = os.path.join(root_dir, 'enhanced_texts','enhanced_ogbn-arxiv.csv')
        text = read_enhanced_text_column_1(csv_path)
        return data, text
    if feature_type == 'enhanced_smoothing':
        csv_path = os.path.join(root_dir, 'enhanced_texts','enhanced_ogbn-arxiv.csv')
        text = read_enhanced_text_column_2(csv_path)
        return data, text


if __name__ == "__main__":
    # 获取数据集
    data, text = get_raw_text_arxiv("enhanced", True)

