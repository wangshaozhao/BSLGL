import os
import csv
import time
import requests
import re


class LLMClassifier:
    def __init__(self, api_key, output_file):
        # 初始化输出文件设置
        self.output_file = output_file
        self.output_dir = os.path.dirname(self.output_file)

        # 大模型API配置
        self.api_url = "https://api.siliconflow.cn/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.model = 'deepseek-ai/DeepSeek-V3.1' #"deepseek-ai/DeepSeek-R1"

        # 数据存储
        self.texts = None
        self.true_labels = None
        self.label_map = None

        self._create_output_dir()

    def _create_output_dir(self):
        """创建输出目录"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")

    def load_texts(self, texts):
        """加载待分类文本"""
        if not isinstance(texts, list) or len(texts) == 0:
            raise ValueError("Input texts must be a non-empty list")
        self.texts = texts
        print(f"Successfully loaded {len(self.texts)} text records")
        return True

    # def set_labels(self, true_labels, label_map):
    #     """设置真实标签和标签映射"""
    #     self.true_labels = true_labels
    #     self.label_map = label_map
    def set_labels(self, true_labels, label_map):
        """设置真实标签和标签映射（新增解嵌套逻辑）"""

        # 解嵌套逻辑：如果元素是列表，取第一个元素；否则保留原元素
        def _unpack_label(label):
            if isinstance(label, list):
                # 处理空列表的极端情况
                return label[0] if label else -1  # 空列表返回无效标签-1
            return label

        # 对所有标签执行解嵌套
        self.true_labels = [_unpack_label(label) for label in true_labels]
        self.label_map = label_map

        # 额外校验：确保处理后的标签是可转整数的类型
        for i, label in enumerate(self.true_labels[:5]):  # 仅校验前5个，避免输出过长
            if not isinstance(label, (int, str, float)):
                print(f"警告：索引 {i} 的标签类型仍异常，值={label}, 类型={type(label)}")

    def call_llm_api(self, prompt):
        """调用LLM API"""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 10
        }
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers=self.headers,
                timeout=200
            )
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip()
            print(f"API call failed: {response.status_code} - {response.text}")
            return None
        except Exception as e:
            print(f"API call exception: {str(e)}")
            return None

    def is_valid_label(self, label_num):
        """验证标签是否有效"""
        return label_num is not None and label_num in self.label_map

    def extract_class_label(self, llm_response):
        """从响应中提取数字标签"""
        if not llm_response:
            return None
        numbers = re.findall(r'\d+', llm_response)
        return int(numbers[0]) if numbers else None

    def generate_classification_prompt(self, text, label_map):
        """Generate classification prompt with strict output constraints"""
        label_desc = "\n".join([f"{k}: {v}" for k, v in label_map.items()])
        return f"""Analyze the following academic paper/product name and description, determine its category:

    Paper/Product Content:
    {text}

    Available Categories (return ONLY the corresponding number):
    {label_desc}

    STRICT INSTRUCTIONS:
    1. Output must contain ONLY a single number (0,1,2,etc)
    2. No additional text, explanations, or punctuation
    3. Do not include category names or descriptions
    4. If uncertain, choose the closest matching number

    Your response MUST be in this exact format:
    [number]"""

    def get_llm_prediction_with_retry(self, text, max_attempts=50):
        """获取预测结果（带重试机制）"""
        prompt = self.generate_classification_prompt(text, self.label_map)

        for attempt in range(1, max_attempts + 1):
            llm_response = self.call_llm_api(prompt)

            if not llm_response:
                print(f"API调用失败，重试中... ({attempt}/{max_attempts})")
                time.sleep(2)
                continue

            label_num = self.extract_class_label(llm_response)

            if self.is_valid_label(label_num):
                return label_num, self.label_map[label_num], llm_response

            print(f"无效响应: '{llm_response}'，提取数字: {label_num}，重试中... ({attempt}/{max_attempts})")
            time.sleep(1)

        print(f"⚠️ 达到最大重试次数 {max_attempts} 次")
        return -1, "MAX_RETRY_EXCEEDED", None

    def process_classification(self, start_index=0, end_index=None):
        """执行分类流程"""
        if not all([self.texts, self.true_labels, self.label_map]):
            raise ValueError("Please load texts and labels first")

        total = len(self.texts)
        start = max(0, min(start_index, total))
        end = min(end_index if end_index else total, total)

        if start >= end:
            print(f"Invalid range: start={start}, end={end}")
            return None

        print(f"Processing texts {start}-{end - 1} of {total}")

        # 准备输出文件
        columns = ['index', 'predicted_num', 'predicted_text',
                   'true_num', 'true_text', 'enhanced_text']
        file_exists = os.path.exists(self.output_file)

        with open(self.output_file, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            if not file_exists:
                writer.writeheader()

            for i in range(start, end):
                try:
                    text = self.texts[i]
                    true_num = int(self.true_labels[i])

                    true_text = self.label_map.get(true_num, "UNKNOWN")

                    print(f"Processing {i + 1}/{total} (True: {true_text})")

                    # 获取预测结果（带重试）
                    pred_num, pred_text, _ = self.get_llm_prediction_with_retry(text)

                    writer.writerow({
                        'index': i,
                        'predicted_num': pred_num,
                        'predicted_text': pred_text,
                        'true_num': true_num,
                        'true_text': true_text,
                        'enhanced_text': text
                    })

                    print(f"Result: Predicted {pred_num} ({pred_text})")
                    time.sleep(1)

                except Exception as e:
                    print(f"Error processing {i}: {str(e)}")
                    writer.writerow({
                        'index': i,
                        'predicted_num': -1,
                        'predicted_text': f"ERROR: {str(e)}",
                        'true_num': self.true_labels[i] if i < len(self.true_labels) else -1,
                        'true_text': self.label_map.get(int(self.true_labels[i]), "UNKNOWN") if i < len(
                            self.true_labels) else "UNKNOWN",
                        'enhanced_text': text if i < len(self.texts) else "MISSING"
                    })
                    continue

        print(f"Classification completed. Results saved to: {self.output_file}")
        return self.output_file