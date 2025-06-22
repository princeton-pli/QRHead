import os
import json
from pyserini.search import get_qrels
from beir.retrieval.evaluation import EvaluateRetrieval
import argparse


def beir_eval(retrieval_results, data: str):
    ks = [5, 10]
    qrel_name = 'beir-v1.0.0-{}-test'.format(data)
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


def collect_results(args):
    tasks = ["trec-covid", "nfcorpus", "scifact", "robust04", "dbpedia-entity", "fiqa", "trec-news", "scidocs", "fever", "climate-fever", "nq"]
    task_results = {}

    for task in tasks:
        fname = args.retrieval_result_dir + f"/beir_{task}.json"
        if os.path.exists(fname):
            with open(fname, "r") as f:
                retrieval_results = json.load(f)
                result = beir_eval(retrieval_results, task)
        else:
            result = {}
        task_results[task] = result

    print("All Task: " + ", ".join(tasks))

    k = 10
    metric = "NDCG@{k}".format(k=k)
    row = []
    for task, result in task_results.items():
        val = result.get(metric, 0)
        val = f"{val * 100:.1f}"
        row.append(val)

        print(f"Task: {task}, {metric}: {val}")

    print(metric + ": " + ", ".join([str(x) for x in row]))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_result_dir", required=True, type=str, help="Path to the directory containing BEIR retrieval results.")
    args = parser.parse_args()

    collect_results(args)