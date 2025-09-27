import torch
import os

class EarlyStopping:
    def __init__(self, patience=30, path='es_checkpoint.pt'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path
        self.best_epoch = 0

    def step(self, score, model, epoch):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
            self.best_epoch = epoch
            return False, f"Best (epoch {epoch})"
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True, "Early stop"
            return False, f"Counter {self.counter}/{self.patience}"
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.best_epoch = epoch
            self.counter = 0
            return False, f"Update (epoch {epoch})"

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

class Evaluator:
    def __init__(self, name):
        self.name = name

    def eval(self, results):
        y_pred = results['y_pred']
        y_true = results['y_true']
        correct = (y_pred == y_true).sum().item()
        acc = correct / y_true.shape[0]
        return {'acc': acc}