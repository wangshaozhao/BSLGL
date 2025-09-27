import os
import csv
import time
import requests
import re


class LLMClassifier:
    def __init__(self, api_key, output_file):
        # Initialize output file settings
        self.output_file = output_file
        self.output_dir = os.path.dirname(self.output_file)

        # LLM API configuration
        self.api_url = "https://api.siliconflow.cn/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.model = 'deepseek-ai/DeepSeek-R1' #"deepseek-ai/DeepSeek-R1"

        # Data storage
        self.texts = None
        self.true_labels = None
        self.label_map = None

        self._create_output_dir()

    def _create_output_dir(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")

    def load_texts(self, texts):
        """Load texts to be classified"""
        if not isinstance(texts, list) or len(texts) == 0:
            raise ValueError("Input texts must be a non-empty list")
        self.texts = texts
        print(f"Successfully loaded {len(self.texts)} text records")
        return True

    def set_labels(self, true_labels, label_map):
        """Set true labels and label mapping (with nested list handling)"""

        # Unpack nested labels: if element is list, take first element; otherwise keep original
        def _unpack_label(label):
            if isinstance(label, list):
                # Handle empty list edge case
                return label[0] if label else -1  # Return invalid label -1 for empty list
            return label

        # Apply unpacking to all labels
        self.true_labels = [_unpack_label(label) for label in true_labels]
        self.label_map = label_map

        # Additional validation: ensure processed labels can be converted to integers
        for i, label in enumerate(self.true_labels[:5]):  # Only check first 5 to avoid long output
            if not isinstance(label, (int, str, float)):
                print(f"Warning: Label at index {i} has abnormal type, value={label}, type={type(label)}")

    def call_llm_api(self, prompt):
        """Call LLM API"""
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
        """Validate if label is valid"""
        return label_num is not None and label_num in self.label_map

    def extract_class_label(self, llm_response):
        """Extract numeric label from response"""
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
        """Get prediction with retry mechanism"""
        prompt = self.generate_classification_prompt(text, self.label_map)

        for attempt in range(1, max_attempts + 1):
            llm_response = self.call_llm_api(prompt)

            if not llm_response:
                print(f"API call failed, retrying... ({attempt}/{max_attempts})")
                time.sleep(2)
                continue

            label_num = self.extract_class_label(llm_response)

            if self.is_valid_label(label_num):
                return label_num, self.label_map[label_num], llm_response

            print(f"Invalid response: '{llm_response}', extracted number: {label_num}, retrying... ({attempt}/{max_attempts})")
            time.sleep(1)

        print(f"⚠️ Reached maximum retry attempts {max_attempts}")
        return -1, "MAX_RETRY_EXCEEDED", None

    def process_classification(self, start_index=0, end_index=None):
        """Execute classification process"""
        if not all([self.texts, self.true_labels, self.label_map]):
            raise ValueError("Please load texts and labels first")

        total = len(self.texts)
        start = max(0, min(start_index, total))
        end = min(end_index if end_index else total, total)

        if start >= end:
            print(f"Invalid range: start={start}, end={end}")
            return None

        print(f"Processing texts {start}-{end - 1} of {total}")

        # Prepare output file
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

                    # Get prediction (with retry)
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
