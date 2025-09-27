import os
import csv
import numpy as np
from pathlib import Path
import chardet  # For automatic file encoding detection


def calculate_accuracy(csv_file_path):
    """
    Calculate prediction accuracy of LLM classifier (adapted to actual CSV column names)

    Args:
        csv_file_path (str): Path to CSV file

    Returns:
        float: Accuracy (between 0-1)
        dict: Detailed statistics
    """
    # Check if file exists
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"File not found: {csv_file_path}")

    # Auto-detect file encoding
    with open(csv_file_path, 'rb') as f:
        raw_data = f.read(10000)  # Read partial data for encoding detection
        detection_result = chardet.detect(raw_data)
        encoding = detection_result['encoding'] or 'utf-8'  # Default to utf-8
        print(f"Detected file encoding: {encoding}")

    # Read CSV file
    correct_count = 0
    total_count = 0
    error_count = 0
    retry_exceeded_count = 0

    # Statistics by class
    class_stats = {}

    with open(csv_file_path, 'r', encoding=encoding, errors='replace') as f:
        reader = csv.DictReader(f)

        # Check for required columns
        required_columns = ['predicted_num', 'true_num', 'predicted_text', 'true_text']
        missing_columns = [col for col in required_columns if col not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"CSV file missing required columns: {missing_columns}, existing columns: {reader.fieldnames}")

        for row in reader:
            total_count += 1

            try:
                # Get predicted and true labels from CSV columns (using actual column names)
                pred_num = int(row['predicted_num'])
                true_num = int(row['true_num'])  # Critical fix: ensure this column is numeric ID
            except ValueError as e:
                # Handle cases that can't be converted to integers
                error_count += 1
                print(f"Warning: Row {total_count} has invalid data format - {str(e)}, skipped")
                continue

            # Handle retry exceeded cases (assuming -1 indicates this)
            if pred_num == -1:
                if row['predicted_text'] == 'MAX_RETRY_EXCEEDED':
                    retry_exceeded_count += 1
                else:
                    error_count += 1
                continue

            # Initialize class statistics (using true label name)
            true_text = row['true_text'].strip()
            if true_num not in class_stats:
                class_stats[true_num] = {
                    'true_count': 0,
                    'correct_count': 0,
                    'class_name': true_text  # Fix: use true_text as class name
                }

            class_stats[true_num]['true_count'] += 1

            # Check if prediction is correct
            if pred_num == true_num:
                correct_count += 1
                class_stats[true_num]['correct_count'] += 1

    # Calculate overall accuracy
    valid_count = total_count - error_count - retry_exceeded_count
    if valid_count > 0:
        overall_accuracy = correct_count / valid_count
    else:
        overall_accuracy = 0.0

    # Calculate accuracy for each class
    for class_id, stats in class_stats.items():
        if stats['true_count'] > 0:
            stats['accuracy'] = stats['correct_count'] / stats['true_count']
        else:
            stats['accuracy'] = 0.0

    # Prepare return results
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
    """Print detailed accuracy statistics"""
    print("=" * 60)
    print("LLM Classifier Performance Evaluation")
    print("=" * 60)
    print(f"Total samples: {result['total_samples']}")
    print(f"Valid predictions: {result['valid_samples']}")
    print(f"Error samples: {result['error_samples']}")
    print(f"Retry exceeded samples: {result['retry_exceeded_samples']}")
    print(f"Correct predictions: {result['correct_predictions']}")
    print(f"Overall accuracy: {result['overall_accuracy']:.4f} ({result['overall_accuracy'] * 100:.2f}%)")
    print("-" * 60)

    # Print accuracy for each class
    print("Accuracy by class:")
    for class_id, stats in result['class_accuracy'].items():
        print(f"  Class {class_id} ({stats['class_name']}): "
              f"{stats['accuracy']:.4f} ({stats['correct_count']}/{stats['true_count']})")
    print("=" * 60)


if __name__ == "__main__":
    dataset = 'ogbn-arxiv'  # Can be switched to 'cora' 'pubmed' 'arxiv-2023''ogbn-products' 'ogbn-arxiv'
    output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "llm_predict_result")
    target_dir = os.path.normpath(output_dir)

    # Build full CSV file path (based on dataset selection)
    if dataset == "cora":
        csv_file = os.path.join(target_dir, "predict_cora_enhanced.csv")
    elif dataset == "pubmed":
        csv_file = os.path.join(target_dir, "predict_pubmed_orig.csv")
    elif dataset == "arxiv-2023":
        csv_file = os.path.join(target_dir, "predict_arxiv-2023_enhanced.csv")
    elif dataset == 'ogbn-products':
        csv_file = os.path.join(target_dir, "predict_products_enhanced.csv")
    elif dataset == 'ogbn-arxiv':
        csv_file = os.path.join(target_dir, "predict_ogbn-arxiv_orig.csv")
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Calculate accuracy
    try:
        accuracy, detailed_results = calculate_accuracy(csv_file)
        # Print results
        print_detailed_results(detailed_results)
    except Exception as e:
        print(f"Execution failed: {str(e)}")
