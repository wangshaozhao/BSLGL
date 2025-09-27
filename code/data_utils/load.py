import torch

def load_data(dataset, feature_type, use_text=False, seed=0):
    """统一数据加载接口，支持多数据集文本属性加载"""
    if dataset == 'cora':
        from .load_cora import get_raw_text_cora as get_raw_text
        num_classes = 7
    elif dataset == 'pubmed':
        from .load_pubmed import get_raw_text_pubmed as get_raw_text
        num_classes = 3
    elif dataset == 'ogbn-arxiv':
        from .load_arxiv import get_raw_text_arxiv as get_raw_text
        num_classes = 40
    elif dataset == 'ogbn-products':
        from .load_products import get_raw_text_products as get_raw_text
        num_classes = 47
    elif dataset == 'arxiv_2023':
        from .load_arxiv_2023 import get_raw_text_arxiv_2023 as get_raw_text
        num_classes = 40
    else:
        raise ValueError(f"不支持的数据集: {dataset}")

    # 加载数据（含文本或不含文本）
    if not use_text:
        data, _ = get_raw_text(feature_type, use_text=False, seed=seed)
        return data, num_classes
    else:
        data, text = get_raw_text(feature_type,use_text=True, seed=seed)
        return data, num_classes, text