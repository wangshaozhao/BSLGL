
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
    """生成并保存 BoW 嵌入"""
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
    """
    为 ogbn-products 生成 BoW 嵌入，先提取 BoW 特征，再用 PCA 降维到 100
    """
    if not os.path.exists(emb_path):
        os.makedirs(os.path.dirname(emb_path), exist_ok=True)
        # 生成 BoW 特征
        vectorizer = CountVectorizer(max_features=max_features, min_df=min_df)
        bow_embeddings = vectorizer.fit_transform(texts).toarray()
        # PCA 降维到 100 维
        pca = PCA(n_components=100)
        pca_embeddings = pca.fit_transform(bow_embeddings)
        np.save(emb_path, pca_embeddings)
    return np.load(emb_path)

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def generate_embeddings_pubmed(texts, emb_path, max_features=500):
    """
    为 PubMed 生成 TF-IDF 嵌入，词典大小为 500
    """
    if not os.path.exists(emb_path):
        os.makedirs(os.path.dirname(emb_path), exist_ok=True)
        # 生成 TF-IDF 特征
        vectorizer = TfidfVectorizer(max_features=max_features)
        tfidf_embeddings = vectorizer.fit_transform(texts).toarray()
        np.save(emb_path, tfidf_embeddings)
    return np.load(emb_path)

import os
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')  # 下载分词所需的资源

def generate_embeddings_ogbn_arxiv(texts, emb_path, embedding_dim=128):
    """
    为 ogbn-arxiv 生成 skip - gram 词嵌入，维度为 128
    """
    if not os.path.exists(emb_path):
        os.makedirs(os.path.dirname(emb_path), exist_ok=True)
        # 对文本进行分词
        tokenized_texts = [word_tokenize(text.lower()) for text in texts]
        # 训练 skip - gram 模型
        model = Word2Vec(sentences=tokenized_texts, vector_size=embedding_dim, sg=1, window=5, min_count=1, workers=4)
        # 获取每个文本的嵌入（这里简单地将文本中所有词的嵌入取平均，可根据需求调整）
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
    """
    为 tape-arxiv23 生成 Word2Vec 词嵌入，维度为 300
    """
    if not os.path.exists(emb_path):
        os.makedirs(os.path.dirname(emb_path), exist_ok=True)
        # 对文本进行分词
        tokenized_texts = [word_tokenize(text.lower()) for text in texts]
        # 训练 Word2Vec 模型（默认使用 CBOW 模式，若要指定 skip-gram 可设置 sg=1）
        model = Word2Vec(sentences=tokenized_texts, vector_size=embedding_dim, window=5, min_count=1, workers=4)
        # 获取每个文本的嵌入（这里简单地将文本中所有词的嵌入取平均，可根据需求调整）
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
