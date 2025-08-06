import json
import argparse
from typing import Dict, List, Tuple, Any


def get_top_k_docs(doc_scores: Dict[str, float], k: int) -> List[str]:
    """Get the top-k docs based on scores."""

    # Sort docs by scores in descending order
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in sorted_docs[:k]]

def compute_recall(samples: List[Dict[str, Any]], scores: Dict[str, Dict[str, float]], k: int) -> Tuple[float, float]:
    """Compute recall_any and recall_all at k."""
    recall = 0
    recall_any_count = 0
    recall_all_count = 0
    valid_samples = 0
    
    for sample in samples:
        qid = str(sample.get('idx'))
        gt_docs = sample.get('gt_docs', [])
            
        # Convert gt_docs to strings to match scores keys
        gt_docs = [str(ch) for ch in gt_docs]
        
        # Get top-k predicted docs
        doc_scores = scores[qid]
        top_k_docs = get_top_k_docs(doc_scores, k)
        
        # Check if any or all ground truth docs are in top-k
        any_match = any(ch in top_k_docs for ch in gt_docs)
        all_match = all(ch in top_k_docs for ch in gt_docs)

        count_matches = len([ch for ch in top_k_docs if ch in gt_docs])
        
        recall += count_matches / len(gt_docs) if gt_docs else 0
        recall_any_count += int(any_match)
        recall_all_count += int(all_match)
        valid_samples += 1
    
    # Compute recall metrics
    recall = recall / valid_samples if valid_samples > 0 else 0
    recall_any = recall_any_count / valid_samples if valid_samples > 0 else 0
    recall_all = recall_all_count / valid_samples if valid_samples > 0 else 0
    
    return recall, recall_any, recall_all

def main():

    parser = argparse.ArgumentParser(description='Compute recall metrics for doc retrieval')
    parser.add_argument('--retrieval_result_file', required=True, help='Path to the retrieval scores JSON file')
    parser.add_argument('--data_file', required=True, help='Path to the ground truth data JSON file')

    args = parser.parse_args()

    # Load data
    with open(args.retrieval_result_file, 'r') as f:
        scores = json.load(f)
    with open(args.data_file, 'r') as f:
        samples = json.load(f)
    
    print(f"Loaded {len(scores)} score entries and {len(samples)} ground truth samples")
    
    # Compute and display recall metrics for each k
    for k in [3, 5, 10]:
        recall, recall_any, recall_all = compute_recall(samples, scores, k)
        print(f"Recall@{k}:")
        print(f"  Recall: {recall:.4f}")
        print(f"  Recall-all: {recall_all:.4f}")
        print(f"  Recall-any: {recall_any:.4f}")

if __name__ == "__main__":
    main()