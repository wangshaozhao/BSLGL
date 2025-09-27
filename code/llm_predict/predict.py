from core.data_utils.load_cora import get_raw_text_cora
from core.data_utils.load_pubmed import get_raw_text_pubmed
from core.data_utils.load_arxiv_2023 import get_raw_text_arxiv_2023

from api import LLMClassifier
import os

if __name__ == "__main__":
    # 配置API密钥和输出路径
    api_key = "sk-yssamqbdjvmyflrhgmjzzqycdrjlbfkzvlkmsubpzuyxdsme"
    dataset = 'cora'
    # 定义输出文件路径
    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "llm_predict_result")
    output_dir = os.path.normpath(output_dir)

    # 根据数据集配置参数
    if dataset == "cora":
        output_file = os.path.join(output_dir, "predict_cora_enhanced.csv")
        label_map = {
            0: 'Case Based',
            1: 'Genetic Algorithms',
            2: 'Neural Networks',
            3: 'Probabilistic Methods',
            4: 'Reinforcement Learning',
            5: 'Rule Learning',
            6: 'Theory'
        }
        data, texts = get_raw_text_cora('enhanced', True)
        true_labels = data.y.tolist()
    elif dataset == "pubmed":
        output_file = os.path.join(output_dir, "predict_pubmed_orig.csv")
        label_map = {
            0: 'Experimentally induced diabetes',
            1: 'Type 1 diabetes',
            2: 'Type 2 diabetes'
        }
        data, texts = get_raw_text_pubmed('orig', True)
        true_labels = data.y.tolist()
    elif dataset == "arxiv-2023":
        label_map = {
            1: "Human-Computer Interaction",
            2: "Logic in Computer Science",
            3: "Computers and Society",
            4: "Cryptography and Security",
            5: "Distributed, Parallel, and Cluster Computing",
            6: "Human-Computer Interaction",
            7: "Computational Engineering, Finance, and Science",
            8: "Networking and Internet Architecture",
            9: "Computational Complexity",
            10: "Artificial Intelligence",
            11: "Multiagent Systems",
            12: "General Literature",
            13: "Neural and Evolutionary Computing",
            14: "Symbolic Computation",
            15: "Hardware Architecture",
            16: "Computer Vision and Pattern Recognition",
            17: "Graphics",
            18: "Emerging Technologies",
            20: "Computational Geometry",
            21: "Operating Systems",
            22: "Programming Languages",
            23: "Software Engineering",
            24: "Machine Learning",
            25: "Sound",
            26: "Social and Information Networks",
            27: "Robotics",
            28: "Information Theory",
            29: "Programming Languages",
            30: "Computation and Language",
            31: "Information Retrieval",
            32: "Mathematical Software",
            33: "Formal Languages and Automata Theory",
            34: "Data Structures and Algorithms",
            35: "Operating Systems",
            36: "Computer Science and Game Theory",
            37: "Databases",
            38: "Digital Libraries",
            39: "Discrete Mathematics"
        }
        output_file = os.path.join(output_dir, "predict_arxiv-2023_orig.csv")
        data, texts = get_raw_text_arxiv_2023('orig', True)
        true_labels = data.y.tolist()
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # 初始化分类器
    classifier = LLMClassifier(api_key, output_file)
    classifier.load_texts(texts)
    classifier.set_labels(true_labels, label_map)
    # 执行分类
    classifier.process_classification(start_index=0, end_index=10)