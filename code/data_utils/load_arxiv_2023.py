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
def get_raw_text_arxiv_2023(feature_type='orig',use_text=False, seed=0):
    # 加载图结构数据 (假设graph.pt包含edge_index和y)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    graph_data = torch.load(os.path.join(project_root,'dataset','arxiv_2023','graph.pt'))
    edge_index = graph_data['edge_index']
    y = graph_data['y']
    num_nodes = y.shape[0]

    # 划分训练/验证/测试划分
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # 简单划分示例（实际需根据数据调整）
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

    # 加载原始文本（title和abstract）
    df = pd.read_csv(os.path.join(project_root, 'dataset', 'arxiv_2023_orig', 'paper_info.csv'))
    text = []
    for ti, ab in zip(df['title'], df['abstract']):
        text.append(f'Title: {ti}\nAbstract: {ab}')
        # text.append((ti, ab))
    if feature_type == 'orig':
        return data, text
    if feature_type == 'enhanced':
        # 使用示例
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
    data,text=get_raw_text_arxiv_2023('enhanced',True)
    print(data)
    print(len(data.y))
    print(len(text))
    print(text[3256])
