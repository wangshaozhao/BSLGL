import torch
import numpy as np
from sentence_transformers import SentenceTransformer

class SBERTEncoder:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
        self.device = torch.device('cpu')  

    def encode(self, texts, batch_size=32):

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=self.device
        )
        return embeddings.cpu().numpy()

    def save_embeddings(self, embeddings, save_path):

        np.save(save_path, embeddings)
        print(f"Embeddings saved to {save_path}")

    def load_embeddings(self, load_path):

        return np.load(load_path)
