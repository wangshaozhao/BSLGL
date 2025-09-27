import os
import csv
import numpy as np
from pathlib import Path
import chardet  # 用于自动检测文件编码


def calculate_accuracy(csv_file_path):
    """
    计算LLM分类器的预测准确率（适配实际CSV列名）

    Args:
        csv_file_path (str): CSV文件路径

    Returns:
        float: 准确率 (0-1之间)
        dict: 详细统计信息
    """
    # 检查文件是否存在
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"文件不存在: {csv_file_path}")

    # 自动检测文件编码
    with open(csv_file_path, 'rb') as f:
        raw_data = f.read(10000)  # 读取部分数据用于检测编码
        detection_result = chardet.detect(raw_data)
        encoding = detection_result['encoding'] or 'utf-8'  # 默认为utf-8
        print(f"检测到文件编码: {encoding}")

    # 读取CSV文件
    correct_count = 0
    total_count = 0
    error_count = 0
    retry_exceeded_count = 0

    # 按类别统计
    class_stats = {}

    with open(csv_file_path, 'r', encoding=encoding, errors='replace') as f:
        reader = csv.DictReader(f)

        # 检查必要的列是否存在
        required_columns = ['predicted_num', 'true_num', 'predicted_text', 'true_text']
        missing_columns = [col for col in required_columns if col not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"CSV文件缺少必要的列：{missing_columns}，现有列：{reader.fieldnames}")

        for row in reader:
            total_count += 1

            try:
                # 从CSV列中获取预测标签和真实标签（根据实际列名）
                pred_num = int(row['predicted_num'])
                true_num = int(row['true_num'])  # 关键修正：确保这一列确实是数字ID
            except ValueError as e:
                # 处理无法转换为整数的情况
                error_count += 1
                print(f"警告：第{total_count}行数据格式错误 - {str(e)}，已跳过")
                continue

            # 处理重试超限的情况（假设用-1表示）
            if pred_num == -1:
                if row['predicted_text'] == 'MAX_RETRY_EXCEEDED':
                    retry_exceeded_count += 1
                else:
                    error_count += 1
                continue

            # 初始化类别统计（使用真实标签名称）
            true_text = row['true_text'].strip()
            if true_num not in class_stats:
                class_stats[true_num] = {
                    'true_count': 0,
                    'correct_count': 0,
                    'class_name': true_text  # 修正：使用true_text作为类别名称
                }

            class_stats[true_num]['true_count'] += 1

            # 检查预测是否正确
            if pred_num == true_num:
                correct_count += 1
                class_stats[true_num]['correct_count'] += 1

    # 计算总体准确率
    valid_count = total_count - error_count - retry_exceeded_count
    if valid_count > 0:
        overall_accuracy = correct_count / valid_count
    else:
        overall_accuracy = 0.0

    # 计算每个类别的准确率
    for class_id, stats in class_stats.items():
        if stats['true_count'] > 0:
            stats['accuracy'] = stats['correct_count'] / stats['true_count']
        else:
            stats['accuracy'] = 0.0

    # 准备返回结果
    result = {
        'overall_accuracy': overall_accuracy,
        'total_samples': total_count,
        'correct_predictions': correct_count,
        'error_samples': error_count,
        'retry_exceeded_samples': retry_exceeded_count,
        'valid_samples': valid_count,
        'class_accuracy': class_stats
    }

    return overall_accuracy, result


def print_detailed_results(result):
    """打印详细的准确率统计结果"""
    print("=" * 60)
    print("LLM分类器性能评估")
    print("=" * 60)
    print(f"总样本数: {result['total_samples']}")
    print(f"有效预测样本数: {result['valid_samples']}")
    print(f"错误样本数: {result['error_samples']}")
    print(f"重试超限样本数: {result['retry_exceeded_samples']}")
    print(f"正确预测数: {result['correct_predictions']}")
    print(f"总体准确率: {result['overall_accuracy']:.4f} ({result['overall_accuracy'] * 100:.2f}%)")
    print("-" * 60)

    # 打印每个类别的准确率
    print("各类别准确率:")
    for class_id, stats in result['class_accuracy'].items():
        print(f"  类别 {class_id} ({stats['class_name']}): "
              f"{stats['accuracy']:.4f} ({stats['correct_count']}/{stats['true_count']})")
    print("=" * 60)


if __name__ == "__main__":
    dataset = 'ogbn-arxiv'  # 可切换为 'cora' 'pubmed' 'arxiv-2023''ogbn-products' 'ogbn-arxiv'
    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "llm_predict_result")
    target_dir = os.path.normpath(output_dir)

    # 构建完整的CSV文件路径（根据数据集选择）
    if dataset == "cora":
        csv_file = os.path.join(target_dir, "predict_cora_enhanced.csv")
    elif dataset == "pubmed":
        csv_file = os.path.join(target_dir, "predict_pubmed_orig.csv")
    elif dataset =="arxiv-2023":
        csv_file = os.path.join(target_dir, "predict_arxiv-2023_enhanced.csv")
    elif dataset == 'ogbn-products':
        csv_file = os.path.join(target_dir, "predict_products_enhanced.csv")
    elif dataset == 'ogbn-arxiv':
        csv_file = os.path.join(target_dir, "predict_ogbn-arxiv_orig.csv")
    else:
        raise ValueError(f"不支持的数据集: {dataset}")

    # 计算准确率
    try:
        accuracy, detailed_results = calculate_accuracy(csv_file)
        # 打印结果
        print_detailed_results(detailed_results)
    except Exception as e:
        print(f"执行失败: {str(e)}")
