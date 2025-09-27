from yacs.config import CfgNode as CN
import argparse


def set_cfg(cfg):

    cfg.dataset = 'ogbn-arxiv'  # 'cora' 'arxiv_2023' 'ogbn-products' 'pubmed' 'obgn-arxiv'
    cfg.device = 'cpu'  # CPU
    cfg.seed = 42  # 0-1-10-24-42-100-111-520-1024-2025
    cfg.runs = 4

    
    cfg.gnn = CN()
    cfg.gnn.model = CN()
    cfg.gnn.model.name = 'GCN'  # GCN 'MLP' 'SAGE'
    cfg.gnn.model.num_layers = 3
    cfg.gnn.model.hidden_dim = 64


    cfg.gnn.train = CN()
    cfg.gnn.train.weight_decay = 0  #0.00003/4/5
    cfg.gnn.train.epochs = 400
    cfg.gnn.train.feature_type = 'orig'  #  'enhanced' 'enhanced_impartial' 'enhanced_smoothing'
    cfg.gnn.train.early_stop_patience = 30
    cfg.gnn.train.lr = 0.02        #0.01
    cfg.gnn.train.dropout = 0.2    #0.1


    cfg.sbert = CN()
    cfg.sbert.model_name = 'all-mpnet-base-v2'

    return cfg


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str)
    parser.add_argument('--device', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--runs', type=int)

    parser.add_argument('--gnn_model_name', type=str, help='(GCN/SAGE/MLP)')
    parser.add_argument('--gnn_model_num_layers', type=int)
    parser.add_argument('--gnn_model_hidden_dim', type=int)

    parser.add_argument('--gnn_train_weight_decay', type=float)
    parser.add_argument('--gnn_train_epochs', type=int)
    parser.add_argument('--gnn_train_feature_type', type=str)
    parser.add_argument('--gnn_train_early_stop_patience', type=int)
    parser.add_argument('--gnn_train_lr', type=float)
    parser.add_argument('--gnn_train_dropout', type=float, help='Dropout rate')

    # SBERT配置
    parser.add_argument('--sbert_model_name', type=str, help='SBERT model name')

    return parser.parse_args()


def update_cfg(cfg):
    args = parse_args()

    # update
    if args.dataset is not None:
        cfg.dataset = args.dataset
    if args.device is not None:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed
    if args.runs is not None:
        cfg.runs = args.runs


    if args.gnn_model_name is not None:
        cfg.gnn.model.name = args.gnn_model_name
    if args.gnn_model_num_layers is not None:
        cfg.gnn.model.num_layers = args.gnn_model_num_layers
    if args.gnn_model_hidden_dim is not None:
        cfg.gnn.model.hidden_dim = args.gnn_model_hidden_dim


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


    if args.sbert_model_name is not None:
        cfg.sbert.model_name = args.sbert_model_name

    return cfg


cfg = CN()
cfg = set_cfg(cfg)
