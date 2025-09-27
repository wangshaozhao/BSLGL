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
        # 使用示例
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
    # 获取数据集（不使用文本特征）
    data, text = get_raw_text_pubmed("orig", False)

    # 定义CSV文件路径
    file_path = r"C:\Users\王绍召\Downloads\pubmed.csv"

    try:
        # 读取CSV文件中的预测值（第一列，无列索引）
        predictions = []
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                # 确保每行至少有一个元素
                if row:
                    # 假设预测值是整数类型
                    predictions.append(int(row[0]))

        # 验证预测值数量与样本数量是否一致
        if len(predictions) != len(data.y):
            raise ValueError(f"预测值数量({len(predictions)})与样本数量({len(data.y)})不匹配")

        # 计算准确率
        correct = 0
        total = len(data.y)
        for pred, true in zip(predictions, data.y.numpy()):
            if pred == true:
                correct += 1

        accuracy = correct / total
        print(f"预测准确率: {accuracy:.4f} ({correct}/{total})")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {file_path}")
    except Exception as e:
        print(f"计算准确率时出错: {str(e)}")