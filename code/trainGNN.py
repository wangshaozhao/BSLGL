import torch
import os

from code.GNNs.gnn_utils import Evaluator, EarlyStopping

from code.config import  update_cfg,cfg
from code.data_utils.load import load_data
from code.sentence_bert.encoder import SBERTEncoder
from code.utils import generate_bow_embeddings_cora


class GNNTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dataset_name = cfg.dataset
        self.gnn_name = cfg.gnn.model.name
        self.feature_type = cfg.gnn.train.feature_type

    
        self.data, self.num_classes = load_data(
            self.dataset_name, self.feature_type, use_text=False, seed=cfg.seed)
        self.data = self.data.to(self.device)

        if self.data.y.dim() == 2 and self.data.y.size(1) == 1:
            self.data.y = self.data.y.squeeze(1)  

        print(self.feature_type)
        if self.feature_type == "orig":

            _, _, texts = load_data(
                self.dataset_name,self.feature_type, use_text=True, seed=cfg.seed)
            sbert = SBERTEncoder(cfg.sbert.model_name)
            emb_path = f'embeddings/{self.dataset_name}_orig.npy'
            if not os.path.exists(emb_path):
                os.makedirs('embeddings', exist_ok=True)
                embeddings = sbert.encode(texts)
                sbert.save_embeddings(embeddings, emb_path)
            self.features = torch.from_numpy(sbert.load_embeddings(emb_path)).float().to(self.device)
        elif self.feature_type == "enhanced":
            _, _, texts = load_data(
                self.dataset_name,self.feature_type, use_text=True, seed=cfg.seed)
            sbert = SBERTEncoder(cfg.sbert.model_name)
            emb_path = f'embeddings/{self.dataset_name}_enhanced.npy'
            if not os.path.exists(emb_path):
                os.makedirs('embeddings', exist_ok=True)
                embeddings = sbert.encode(texts)
                sbert.save_embeddings(embeddings, emb_path)
            self.features = torch.from_numpy(sbert.load_embeddings(emb_path)).float().to(self.device)
        elif self.feature_type == "enhanced_smoothing":
            _, _, texts = load_data(
                self.dataset_name,self.feature_type, use_text=True, seed=cfg.seed)
            sbert = SBERTEncoder(cfg.sbert.model_name)
            emb_path = f'embeddings/{self.dataset_name}_enhanced_smoothing.npy'
            if not os.path.exists(emb_path):
                os.makedirs('embeddings', exist_ok=True)
                embeddings = sbert.encode(texts)
                sbert.save_embeddings(embeddings, emb_path)
            self.features = torch.from_numpy(sbert.load_embeddings(emb_path)).float().to(self.device)
        elif self.feature_type == "enhanced_impartial":
            _, _, texts = load_data(
                self.dataset_name,self.feature_type, use_text=True, seed=cfg.seed)
            sbert = SBERTEncoder(cfg.sbert.model_name)
            emb_path = f'embeddings/{self.dataset_name}_enhanced_impartial.npy'
            if not os.path.exists(emb_path):
                os.makedirs('embeddings', exist_ok=True)
                embeddings = sbert.encode(texts)
                sbert.save_embeddings(embeddings, emb_path)
            self.features = torch.from_numpy(sbert.load_embeddings(emb_path)).float().to(self.device)
        elif self.feature_type == "enhanced_shallow":
            _, _, texts = load_data(
                self.dataset_name, "enhanced", use_text=True, seed=cfg.seed)
            emb_path = f'embeddings/{self.dataset_name}_enhanced_shallow.npy'

            if self.dataset_name == "cora":
                from .utils import generate_bow_embeddings_cora
                embeddings = generate_bow_embeddings_cora(texts, emb_path, max_features=1433,min_df=10)
                self.features = torch.from_numpy(embeddings).float().to(self.device)
            elif self.dataset_name == "ogbn-products":
                from .utils import generate_bow_embeddings_products
                embeddings = generate_bow_embeddings_products(texts, emb_path, max_features=1433,min_df=10)
                self.features = torch.from_numpy(embeddings).float().to(self.device)
            elif self.dataset_name == "pubmed":
                from .utils import generate_embeddings_pubmed
                embeddings = generate_embeddings_pubmed(texts, emb_path, max_features=500)
                self.features = torch.from_numpy(embeddings).float().to(self.device)
            elif self.dataset_name == "ogbn-arxiv":
                from .utils import generate_embeddings_ogbn_arxiv
                embeddings = generate_embeddings_ogbn_arxiv(texts, emb_path, embedding_dim=128)
                self.features = torch.from_numpy(embeddings).float().to(self.device)
            elif self.dataset_name == "arxiv_2023":
                from .utils import generate_embeddings_arxiv_2023
                embeddings = generate_embeddings_arxiv_2023(texts, emb_path, embedding_dim=300)
                self.features = torch.from_numpy(embeddings).float().to(self.device)

        elif self.feature_type == "orig_shallow":
            _, _, texts = load_data(
                self.dataset_name, "orig", use_text=True, seed=cfg.seed)
            emb_path = f'embeddings/{self.dataset_name}_orig_shallow.npy'
            if self.dataset_name == "cora":
                from .utils import generate_bow_embeddings_cora
                embeddings = generate_bow_embeddings_cora(texts, emb_path, max_features=1433,min_df=10)
                self.features = torch.from_numpy(embeddings).float().to(self.device)
            elif self.dataset_name == "ogbn-products":
                from .utils import generate_bow_embeddings_products
                embeddings = generate_bow_embeddings_products(texts, emb_path, max_features=1433,min_df=10)
                self.features = torch.from_numpy(embeddings).float().to(self.device)
            elif self.dataset_name == "pubmed":
                from .utils import generate_embeddings_pubmed
                embeddings = generate_embeddings_pubmed(texts, emb_path, max_features=500)
                self.features = torch.from_numpy(embeddings).float().to(self.device)
            elif self.dataset_name == "ogbn-arxiv":
                from .utils import generate_embeddings_ogbn_arxiv
                embeddings = generate_embeddings_ogbn_arxiv(texts, emb_path, embedding_dim=128)
                self.features = torch.from_numpy(embeddings).float().to(self.device)
            elif self.dataset_name == "arxiv_2023":
                from .utils import generate_embeddings_arxiv_2023
                embeddings = generate_embeddings_arxiv_2023(texts, emb_path, embedding_dim=300)
                self.features = torch.from_numpy(embeddings).float().to(self.device)

        else:
            raise ValueError(f"not support：{self.feature_type}")

        self._init_model()

        self.early_stopping = EarlyStopping(
            patience=cfg.gnn.train.early_stop_patience,
            path=f'checkpoints/{self.dataset_name}_{self.gnn_name}_best.pt'
        )
        os.makedirs('checkpoints', exist_ok=True)

    def _init_model(self):
        print(self.gnn_name)
        print(self.dataset_name)
        if self.gnn_name == 'GCN':
            from code.GNNs.GCN.model import GCN as GNN
        elif self.gnn_name == 'SAGE':
            from code.GNNs.SAGE.model import SAGE as GNN
        elif self.gnn_name == 'MLP':
            from code.GNNs.MLP.model import MLP as GNN
        else:
            raise ValueError(f"not support: {self.gnn_name}")

        self.model = GNN(
            in_channels=self.features.shape[1],
            hidden_channels=self.cfg.gnn.model.hidden_dim,
            out_channels=self.num_classes,
            num_layers=self.cfg.gnn.model.num_layers,
            dropout=self.cfg.gnn.train.dropout
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.gnn.train.lr,
            weight_decay=self.cfg.gnn.train.weight_decay
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.evaluator = Evaluator(self.dataset_name)

    def train(self):
        best_val_acc = 0
        test_acc = 0  

        for epoch in range(self.cfg.gnn.train.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(self.features, self.data.edge_index)
            loss = self.criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            self.optimizer.step()

            val_acc = self._evaluate(out, self.data.val_mask)

            early_stop, msg = self.early_stopping.step(val_acc, self.model, epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = self._evaluate(out, self.data.test_mask)

            if epoch % 10 == 0:
                train_acc = self._evaluate(out, self.data.train_mask)
                print(f"Epoch {epoch}: Loss={loss.item():.4f}, TrainAcc={train_acc:.4f}, ValAcc={val_acc:.4f}, {msg}")

            if early_stop:
                print(f"early: epoch {epoch}")
                break

        self.model.load_state_dict(torch.load(self.early_stopping.path))

        final_out = self.model(self.features, self.data.edge_index)
        test_acc = self._evaluate(final_out, self.data.test_mask)

        print(f"Best Val Acc: {self.early_stopping.best_score:.4f}, Test Acc: {test_acc:.4f}")

    def _evaluate(self, out, mask):
        self.model.eval()
        with torch.no_grad():
            pred = out[mask].argmax(dim=1)
            return self.evaluator.eval({
                'y_pred': pred.unsqueeze(1),
                'y_true': self.data.y[mask].unsqueeze(1)
            })['acc']


if __name__ == '__main__':
    cfg = update_cfg(cfg)
    trainer = GNNTrainer(cfg)
    trainer.train()
