from yacs.config import CfgNode as CN
import argparse


def set_cfg(cfg):
    # 基础配置
    cfg.dataset = 'ogbn-arxiv'  # 'cora' 'arxiv_2023' 'ogbn-products' 'pubmed' 'obgn-arxiv'
    cfg.device = 'cpu'  # CPU运行
    cfg.seed = 42  # 0-1-10-24-42-100-111-520-1024-2025
    cfg.runs = 4

    # GNN配置
    cfg.gnn = CN()
    cfg.gnn.model = CN()
    cfg.gnn.model.name = 'GCN'  # 默认GCN 'MLP' 'SAGE'
    cfg.gnn.model.num_layers = 3
    cfg.gnn.model.hidden_dim = 64

    # GNN训练配置
    cfg.gnn.train = CN()
    cfg.gnn.train.weight_decay = 0  #0.00003/4/5
    cfg.gnn.train.epochs = 400
    cfg.gnn.train.feature_type = 'orig'  # 使用原始文本属性 'enhanced' 'enhanced_impartial' 'enhanced_smoothing'
    cfg.gnn.train.early_stop_patience = 30
    cfg.gnn.train.lr = 0.02        #0.01
    cfg.gnn.train.dropout = 0.2    #0.1

    # Sentence-BERT配置
    cfg.sbert = CN()
    cfg.sbert.model_name = 'all-mpnet-base-v2'

    return cfg


def parse_args():
    parser = argparse.ArgumentParser()
    # 基础配置
    parser.add_argument('--dataset', type=str, help='数据集名称')
    parser.add_argument('--device', type=str, help='运行设备 (cpu/cuda)')
    parser.add_argument('--seed', type=int, help='随机种子')
    parser.add_argument('--runs', type=int, help='运行次数')

    # GNN模型配置
    parser.add_argument('--gnn_model_name', type=str, help='GNN模型名称 (GCN/SAGE/MLP)')
    parser.add_argument('--gnn_model_num_layers', type=int, help='GNN层数')
    parser.add_argument('--gnn_model_hidden_dim', type=int, help='隐藏层维度')

    # 训练配置
    parser.add_argument('--gnn_train_weight_decay', type=float, help='权重衰减')
    parser.add_argument('--gnn_train_epochs', type=int, help='训练轮数')
    parser.add_argument('--gnn_train_feature_type', type=str, help='特征类型')
    parser.add_argument('--gnn_train_early_stop_patience', type=int, help='早停耐心值')
    parser.add_argument('--gnn_train_lr', type=float, help='学习率')
    parser.add_argument('--gnn_train_dropout', type=float, help='Dropout率')

    # SBERT配置
    parser.add_argument('--sbert_model_name', type=str, help='SBERT模型名称')

    return parser.parse_args()


def update_cfg(cfg):
    args = parse_args()

    # 更新配置
    if args.dataset is not None:
        cfg.dataset = args.dataset
    if args.device is not None:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed
    if args.runs is not None:
        cfg.runs = args.runs

    # GNN模型配置
    if args.gnn_model_name is not None:
        cfg.gnn.model.name = args.gnn_model_name
    if args.gnn_model_num_layers is not None:
        cfg.gnn.model.num_layers = args.gnn_model_num_layers
    if args.gnn_model_hidden_dim is not None:
        cfg.gnn.model.hidden_dim = args.gnn_model_hidden_dim

    # 训练配置
    if args.gnn_train_weight_decay is not None:
        cfg.gnn.train.weight_decay = args.gnn_train_weight_decay
    if args.gnn_train_epochs is not None:
        cfg.gnn.train.epochs = args.gnn_train_epochs
    if args.gnn_train_feature_type is not None:
        cfg.gnn.train.feature_type = args.gnn_train_feature_type
    if args.gnn_train_early_stop_patience is not None:
        cfg.gnn.train.early_stop_patience = args.gnn_train_early_stop_patience
    if args.gnn_train_lr is not None:
        cfg.gnn.train.lr = args.gnn_train_lr
    if args.gnn_train_dropout is not None:
        cfg.gnn.train.dropout = args.gnn_train_dropout

    # SBERT配置
    if args.sbert_model_name is not None:
        cfg.sbert.model_name = args.sbert_model_name

    return cfg


cfg = CN()
cfg = set_cfg(cfg)
