import json
import os
import re
from collections import defaultdict


def load_pubmed_results(json_path):
    """加载之前保存的PubMed Top5 TF-IDF结果JSON文件"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"结果文件不存在：{json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)
    print(f"成功加载结果文件，共{len(results)}个样本")
    return results


def get_label_keywords(label_map):
    """从标签映射生成标签关键词列表（用于文本匹配）
    例如：将标签名拆分为核心词汇，如'Type 1 diabetes'拆为['Type 1 diabetes', 'Type 1', 'diabetes']
    提高标签匹配的召回率，避免因文本表述差异漏检
    """
    label_keywords = defaultdict(list)
    for label_code, label_name in label_map.items():
        # 完整标签名（最高优先级匹配）
        label_keywords[label_code].append(label_name)
        # 拆分标签名中的核心词组（如'Type 1 diabetes'拆为'Type 1'和'diabetes'）
        words = label_name.split()
        if len(words) >= 2:
            # 保留2个词以上的组合（避免单个词如'diabetes'过度匹配）
            for i in range(2, len(words) + 1):
                phrase = " ".join(words[:i])  # 如'Type 1'、'Experimentally induced'
                label_keywords[label_code].append(phrase)
    return label_keywords


def is_text_contains_label(text, target_label_code, label_keywords):
    """判断文本是否包含目标标签的关键词（不区分大小写，避免大小写差异影响）
    Args:
        text: 待检测文本（原始文本/相似文本/增强文本）
        target_label_code: 当前样本的真实标签编码（0/1/2）
        label_keywords: 标签关键词字典（由get_label_keywords生成）
    Returns:
        bool: 文本是否包含目标标签关键词
    """
    if not text or not isinstance(text, str):
        return False  # 空文本或非字符串直接判定为无标签

    # 统一转为小写，避免大小写敏感问题
    text_lower = text.lower()
    # 检查当前样本标签对应的所有关键词
    for keyword in label_keywords[target_label_code]:
        if keyword.lower() in text_lower:
            return True  # 匹配到任一关键词即判定为包含标签
    return False


def filter_target_samples(results, label_map, save_dir="pubmed_filtered_results"):
    """筛选目标样本：
    1. 原始文本不包含当前样本的标签信息
    2. Top5相似文本中至少4个包含当前样本的标签信息
    3. 增强文本包含当前样本的标签信息
    """
    # 1. 初始化标签关键词字典
    label_keywords = get_label_keywords(label_map)
    print("\n=== 标签关键词映射（用于文本匹配）===")
    for code, keywords in label_keywords.items():
        print(f"标签{code}（{label_map[code]}）：{keywords}")

    # 2. 遍历所有样本筛选目标索引
    target_indices = []  # 存储符合条件的样本索引
    match_logs = []  # 存储每个样本的匹配详情（便于调试和分析）

    print(f"\n开始筛选目标样本（共{len(results)}个样本）...")
    for idx, sample in enumerate(results):
        # 获取当前样本的核心信息
        sample_idx = sample["索引"]
        true_label_code = sample["真实标签"]
        true_label_name = sample["真实标签名称"]
        orig_text = sample["原始文本"]
        enhanced_text = sample["增强文本"]
        # 获取Top5相似文本（按顺序）
        top5_texts = [
            sample["原始文本1（Top1相似）"],
            sample["原始文本2（Top2相似）"],
            sample["原始文本3（Top3相似）"],
            sample["原始文本4（Top4相似）"],
            sample["原始文本5（Top5相似）"]
        ]

        # 3. 逐项检查条件
        # 条件1：原始文本不包含当前标签
        orig_has_label = is_text_contains_label(orig_text, true_label_code, label_keywords)
        # 条件2：Top5相似文本中至少4个包含当前标签
        top5_has_label_count = sum(
            is_text_contains_label(txt, true_label_code, label_keywords)
            for txt in top5_texts
        )
        top5_meet = top5_has_label_count >= 4
        # 条件3：增强文本包含当前标签
        enhanced_has_label = is_text_contains_label(enhanced_text, true_label_code, label_keywords)

        # 记录匹配日志
        log = {
            "样本索引": sample_idx,
            "真实标签": f"{true_label_code}（{true_label_name}）",
            "原始文本含标签": orig_has_label,
            "Top5相似文本含标签数量": top5_has_label_count,
            "Top5满足条件（≥4）": top5_meet,
            "增强文本含标签": enhanced_has_label,
            "是否符合所有条件": orig_has_label is False and top5_meet and enhanced_has_label
        }
        match_logs.append(log)

        # 4. 收集符合所有条件的样本索引
        if log["是否符合所有条件"]:
            target_indices.append(sample_idx)
            # 打印符合条件的样本（实时反馈）
            print(f"✓ 样本索引{sample_idx}符合条件（标签：{true_label_name}，Top5含标签数：{top5_has_label_count}）")

    # 5. 保存结果（筛选出的索引 + 完整匹配日志）
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存目标索引（便于直接使用）
    target_idx_path = os.path.join(save_dir, "target_sample_indices.json")
    with open(target_idx_path, "w", encoding="utf-8") as f:
        json.dump({
            "筛选条件": [
                "1. 原始文本不包含当前样本的标签信息",
                "2. Top5相似文本中至少4个包含当前样本的标签信息",
                "3. 增强文本包含当前样本的标签信息"
            ],
            "符合条件的样本数量": len(target_indices),
            "符合条件的样本索引": target_indices
        }, f, indent=2, ensure_ascii=False)

    # 保存完整匹配日志（便于分析未匹配样本的原因）
    log_path = os.path.join(save_dir, "sample_match_logs.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(match_logs, f, indent=2, ensure_ascii=False)

    # 6. 输出筛选统计结果
    print(f"\n=== 筛选结果统计 ===")
    print(f"总样本数：{len(results)}")
    print(f"符合条件的样本数：{len(target_indices)}")
    print(f"符合条件的样本索引已保存至：{os.path.abspath(target_idx_path)}")
    print(f"完整匹配日志已保存至：{os.path.abspath(log_path)}")

    if target_indices:
        print(f"\n符合条件的样本索引前10个：{target_indices[:10]}（共{len(target_indices)}个）")
    else:
        print("\n⚠️  未找到符合所有条件的样本，建议检查标签关键词匹配规则或原始数据")

    return target_indices, match_logs


if __name__ == "__main__":
    # -------------------------- 配置参数 --------------------------
    # 1. 之前保存的Top5 TF-IDF结果文件路径（请根据实际路径修改）
    RESULTS_JSON_PATH = r"D:\TAPE_enhancer\core\data_utils\pubmed_similar_results\pubmed_top5_tfidf_results.json"
    # 2. 标签映射（需与main.py完全一致）
    LABEL_MAP = {
        0: 'Experimentally induced diabetes',
        1: 'Type 1 diabetes',
        2: 'Type 2 diabetes'
    }

    # -------------------------- 执行筛选 --------------------------
    try:
        # 1. 加载结果数据
        results = load_pubmed_results(RESULTS_JSON_PATH)
        # 2. 筛选目标样本
        target_indices, _ = filter_target_samples(results, LABEL_MAP)
    except Exception as e:
        print(f"执行出错：{str(e)}")