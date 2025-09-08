import argparse
from tqdm import tqdm
from pyserini.search import get_qrels
from beir.retrieval.evaluation import EvaluateRetrieval
from itertools import product
import json
import random
from qrretriever.attn_retriever import FullHeadRetriever


def beir_eval(retrieval_results, task: str):
    """
    retrieval_results: a dict of qid -> {doc_id -> score}, retrieval results from a specific head
    """
    ks = [5, 10]
    qrel_name = 'beir-v1.0.0-{}-test'.format(task)
    _qrels = get_qrels(qrel_name)
    evaluator = EvaluateRetrieval()
    qrels = {}
    for qid in retrieval_results:
        assert isinstance(qid, str)
        try:
            __qrels = _qrels[qid]
        except:
            try:
                __qrels = _qrels[int(qid)]
            except:
                print('Error in qrels for query id: ', qid)
                continue
        
        # make sure the qrels are in the right format
        qrels[qid] = {}
        for doc_id in __qrels:
            qrels[qid][str(doc_id)] = __qrels[doc_id]
            
        doc_keys = list(qrels[qid].keys())
        for key in doc_keys:
            if not isinstance(qrels[qid][key], int):
                qrels[qid][key] = int(qrels[qid][key]) # make sure the relevance is integer
            if qrels[qid][key] == 0:
                qrels[qid].pop(key)

    ndcg, _, recall, precision = evaluator.evaluate(qrels, retrieval_results, ks)
    return ndcg



def get_doc_scores_per_head(full_head_retriever, data_instances, truncate_by_space=0):
    """
    data_instances: a list of dicts, each dict represents an instance
    """
    random.seed(42)

    doc_scores_per_head = {} # qid -> {doc_id -> score tensor with shape (n_layers, n_heads)}
    for i, data in enumerate(tqdm(data_instances)):

        query = data["question"]
        docs = data["paragraphs"]

        random.shuffle(docs)  # shuffle docs for each instance to detect QRHead for BEIR
        
        for p in docs:

            paragraph_text = p['paragraph_text'].strip()

            if truncate_by_space > 0:
                # Truncate each paragraph by space.
                if len(paragraph_text.split(' ')) > truncate_by_space:
                    print('number of words being truncated: ', len(paragraph_text.split(' ')) - truncate_by_space, flush=True)

                p['paragraph_text'] = ' '.join(paragraph_text.split(' ')[:truncate_by_space])

            else:
                p['paragraph_text'] = paragraph_text

        retrieval_scores = full_head_retriever.score_docs_per_head_for_detection(query, docs) # doc_id -> score tensor with shape (n_layers, n_heads)
        doc_scores_per_head[data['idx']] = retrieval_scores

    return doc_scores_per_head



def score_heads(doc_scores_per_head, task='nq'):
    """
    doc_scores_per_head: a dict of dicts, outer dict key is question idx, inner dict key is doc idx, value is a (n_layers, n_heads) tensor
    """

    # pick first qid
    first_qid = next(iter(doc_scores_per_head))
    first_doc_id = next(iter(doc_scores_per_head[first_qid]))
    example_tensor = doc_scores_per_head[first_qid][first_doc_id]
    num_layers, num_heads = example_tensor.shape

    # score by head
    layer_head = product(range(num_layers), range(num_heads))
    head_scores = {}

    for layer, head in tqdm(layer_head, total=num_layers * num_heads):
        retrieval_results = {} # get new retrieval results for this head, qid -> {doc_id -> score}

        for qid, per_doc_score_tensors in doc_scores_per_head.items():
            # per_doc_score_tensors: a dict of doc_id -> (n_layers, n_heads) tensor
            doc_id2score = {}
            for doc_id, score_tensor in per_doc_score_tensors.items():
                score = score_tensor[layer][head]
                doc_id2score[doc_id] = score.item()

            retrieval_results[qid] = doc_id2score
            
        head_ncdg = beir_eval(retrieval_results, task)
        head_scores[(layer, head)] = head_ncdg["NDCG@10"]

    # replace key with layer-head
    head_scores_list = [(f"{layer}-{head}", score) for (layer, head), score in head_scores.items()]
    # sort heads by scores
    head_scores_list.sort(key=lambda x: x[1], reverse=True)

    return head_scores_list # a list of tuples (head, score)





if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file to find QRHead.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file to save scores for each head.")

    parser.add_argument("--truncate_by_space", type=int, default=0, help="Truncate paragraphs by number of words. Default is 0 (no truncation).")

    parser.add_argument("--config_or_config_path", type=str, default=None, help="Path to the configuration file or a configuration string. If not provided, defaults will be used.")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Path to the model directory or model name.")

    args = parser.parse_args()

    full_head_retriever = FullHeadRetriever(
        config_or_config_path=args.config_or_config_path,
        model_name_or_path=args.model_name_or_path,
    )

    # read input file
    print(f"Reading input file: {args.input_file}", flush=True)
    with open(args.input_file, "r") as f:
        data_instances = json.load(f)

    doc_scores_per_head = get_doc_scores_per_head(full_head_retriever, data_instances, truncate_by_space=args.truncate_by_space)
    head_scores_list = score_heads(doc_scores_per_head, task='nq')

    with open(args.output_file, "w") as f:
        json.dump(head_scores_list, f, indent=4)