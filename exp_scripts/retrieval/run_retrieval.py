import argparse
import json
from tqdm import tqdm
from qrretriever.attn_retriever import FullHeadRetriever, QRRetriever


def main():

    parser = argparse.ArgumentParser(description="Run FullHeadRetriever on input file and write to output file.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSON file.")

    parser.add_argument("--data_type", type=str, choices=["beir", "lme", "clipper"], required=True, help="Type of data. Default is 'lme'.")
    parser.add_argument("--truncate_by_space", type=int, default=0, help="Truncate paragraphs by number of words. Default is 0 (no truncation).")

    parser.add_argument("--retriever_type", type=str, choices=["full_head", "qr_head"], default="full_head", help="Type of retriever to use. Default is 'full_head'.")

    parser.add_argument("--config_or_config_path", type=str, default=None, help="Path to the configuration file or a configuration string. If not provided, defaults will be used.")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Path to the model directory or model name.")

    args = parser.parse_args()

    if args.retriever_type == "full_head":
        retriever = FullHeadRetriever(
            config_or_config_path=args.config_or_config_path,
            model_name_or_path=args.model_name_or_path,
        )
    elif args.retriever_type == "qr_head":
        retriever = QRRetriever(
            config_or_config_path=args.config_or_config_path,
            model_name_or_path=args.model_name_or_path,
        )

    # read input file
    with open(args.input_file, "r") as f:
        data_instances = json.load(f)


    results = {}

    for i, data in enumerate(tqdm(data_instances)):

        query = data["question"]
        docs = data["paragraphs"]
        
        for p in docs:

            paragraph_text = p['paragraph_text'].strip()

            if args.truncate_by_space > 0:
                # Truncate each paragraph by space.
                if len(paragraph_text.split(' ')) > args.truncate_by_space:
                    print('number of words being truncated: ', len(paragraph_text.split(' ')) - args.truncate_by_space, flush=True)

                p['paragraph_text'] = ' '.join(paragraph_text.split(' ')[:args.truncate_by_space])

            else:
                p['paragraph_text'] = paragraph_text

        if args.data_type == "beir":
            # Reverse the order of docs for BEIR, to match the document order used in ICR baseline.
            docs = docs[::-1]

        retrieval_scores = retriever.score_docs(query, docs) # doc_id -> score        
        results[data['idx']] = retrieval_scores

    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()