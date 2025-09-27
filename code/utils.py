
import os
import numpy as np
import time
import datetime
import pytz

def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


def time2str(t):
    if t > 86400:
        return '{:.2f}day'.format(t / 86400)
    if t > 3600:
        return '{:.2f}h'.format(t / 3600)
    elif t > 60:
        return '{:.2f}min'.format(t / 60)
    else:
        return '{:.2f}s'.format(t)

def time_logger(func):
    def wrapper(*args, **kw):
        start_time = time.time()
        print(f'Start running {func.__name__} at {get_cur_time()}')
        ret = func(*args, **kw)
        print(
            f'Finished running {func.__name__} at {get_cur_time()}, running time = {time2str(time.time() - start_time)}.')
        return ret

    return wrapper

import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def generate_bow_embeddings_cora(texts, emb_path, max_features=1000, min_df=1):
    if not os.path.exists(emb_path):
        os.makedirs(os.path.dirname(emb_path), exist_ok=True)
        vectorizer = CountVectorizer(max_features=max_features, min_df=min_df)
        embeddings = vectorizer.fit_transform(texts).toarray()
        np.save(emb_path, embeddings)
    return np.load(emb_path)
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA

def generate_bow_embeddings_products(texts, emb_path, max_features=1433, min_df=10):

    if not os.path.exists(emb_path):
        os.makedirs(os.path.dirname(emb_path), exist_ok=True)
        vectorizer = CountVectorizer(max_features=max_features, min_df=min_df)
        bow_embeddings = vectorizer.fit_transform(texts).toarray()
        pca = PCA(n_components=100)
        pca_embeddings = pca.fit_transform(bow_embeddings)
        np.save(emb_path, pca_embeddings)
    return np.load(emb_path)

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def generate_embeddings_pubmed(texts, emb_path, max_features=500):

    if not os.path.exists(emb_path):
        os.makedirs(os.path.dirname(emb_path), exist_ok=True)
        vectorizer = TfidfVectorizer(max_features=max_features)
        tfidf_embeddings = vectorizer.fit_transform(texts).toarray()
        np.save(emb_path, tfidf_embeddings)
    return np.load(emb_path)

import os
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')  

def generate_embeddings_ogbn_arxiv(texts, emb_path, embedding_dim=128):

    if not os.path.exists(emb_path):
        os.makedirs(os.path.dirname(emb_path), exist_ok=True)

        tokenized_texts = [word_tokenize(text.lower()) for text in texts]

        model = Word2Vec(sentences=tokenized_texts, vector_size=embedding_dim, sg=1, window=5, min_count=1, workers=4)

        text_embeddings = []
        for tokens in tokenized_texts:
            embeddings = [model.wv[token] for token in tokens if token in model.wv]
            if embeddings:
                text_embeddings.append(np.mean(embeddings, axis=0))
            else:
                text_embeddings.append(np.zeros(embedding_dim))
        text_embeddings = np.array(text_embeddings)
        np.save(emb_path, text_embeddings)
    return np.load(emb_path)

import os
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize


def generate_embeddings_arxiv_2023(texts, emb_path, embedding_dim=300):

    if not os.path.exists(emb_path):
        os.makedirs(os.path.dirname(emb_path), exist_ok=True)
        tokenized_texts = [word_tokenize(text.lower()) for text in texts]

        model = Word2Vec(sentences=tokenized_texts, vector_size=embedding_dim, window=5, min_count=1, workers=4)
        text_embeddings = []
        for tokens in tokenized_texts:
            embeddings = [model.wv[token] for token in tokens if token in model.wv]
            if embeddings:
                text_embeddings.append(np.mean(embeddings, axis=0))
            else:
                text_embeddings.append(np.zeros(embedding_dim))
        text_embeddings = np.array(text_embeddings)
        np.save(emb_path, text_embeddings)
    return np.load(emb_path)
