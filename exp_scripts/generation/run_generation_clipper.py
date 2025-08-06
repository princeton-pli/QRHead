import torch
from vllm import LLM, SamplingParams
from tqdm import tqdm
import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import traceback
import json
from typing import Dict, List



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True) # Json file
    parser.add_argument('--output_file', type=str, required=True) # Json file
        
    parser.add_argument('--model_path', type=str, required=True)

    parser.add_argument('--retrieval_method', type=str, required=True, choices=['oracle', 'baseline', 'retriever'], help='oracle, baseline, retriever')
    parser.add_argument('--retrieval_result_file', type=str, required=True) # 'none' if not using retrieval

    parser.add_argument('--topk_doc', type=int, required=True)
    return parser.parse_args()


def get_topk_doc_idx(doc_scores: Dict[str, float], k: int) -> List[str]:
    """Get the top-k docs based on scores."""

    # Sort docs by scores in descending order
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_idx for doc_idx, _ in sorted_docs[:k]]







def prepare_prompt_for_generation(docs: List[Dict], topk_doc_idx: List[str], gt_doc_idx: List[str], statement: str):
    """Prepare the prompt for one sample generation.
    
    Args:
        docs (List[Dict]): List of dictionaries, each with 'idx' and 'paragraph_text'.
        topk_doc_idx (List[str]): List of doc ids. The order indicates the rank of the doc, output from get_topk_doc_idx().
        gt_doc_idx (List[str]): List of ground truth doc ids.
        statement (str): The statement to evaluate.
    
    Returns:
        str: The formatted prompt for the model.
    """

    generation_instruction = "You are provided with a context and a statement. Your task is to carefully read the context and then determine whether the statement is true or false.  \n\nAnswer TRUE if the statement is true in its entirety based on the context provided.\nAnswer FALSE if any part of the statement is false based on the context provided.\n\n<context>{}</context>\n\n\n<statement>{}</statement>\n\n<question>Based on the context provided, is the above statement TRUE or FALSE?</question>\n\nFirst provide an explanation of your decision-making process, and then provide your final answer. Use the following format:\n\n<explanation>YOUR EXPLANATION</explanation>\n<answer>YOUR ANSWER</answer>"

    id2text = {}
    for doc in docs:
        id = str(doc['idx'])
        text = doc['paragraph_text']
        id2text[id] = text

    if topk_doc_idx is None:
        if gt_doc_idx is None: # baseline
            # If no rerank result and not ground truth, use all docs
            result_doc_idx = [doc['idx'] for doc in docs]
        else: # oracle
            result_doc_idx = gt_doc_idx
    else:
        # Get the top-k docs
        result_doc_idx = topk_doc_idx
        assert len(set(result_doc_idx)) == len(result_doc_idx), "Duplicate doc idx found in the top-k context"
        assert len(result_doc_idx) == args.topk_doc, f"Top-k context length should be {args.topk_doc}, but got {len(result_doc_idx)}"

    # Sort result_doc_idx by increasing order, because id means chapter number
    result_doc_idx = sorted(result_doc_idx, key=lambda x: int(x))

    result_docs = []
    for id in result_doc_idx:
        text = id2text[str(id)]
        result_docs.append(text)
    
    # Concatenate the context
    context = "\n\n\n".join(result_docs)

    # Prepare the prompt
    prompt = generation_instruction.format(context, statement)
    return prompt






def vllm_eval(qids, prompts, model_path):
    """
    Batch inference from vllm
    Given a dataframe of claims, generate inference for each claim
    Replaced the claim only, not the book text
    """

    llm = LLM(model=model_path, max_model_len=131072, tensor_parallel_size=torch.cuda.device_count())    # TODO: max_model_len=131072

    try:
        tokenizer = llm.get_tokenizer()
        print("Tokenizer:", tokenizer.chat_template)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("Using custom tokenizer")
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1000,
        stop_token_ids=[
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ],
    )

    inputs = []
    for prompt in prompts:
        try:
            messages = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except:
            traceback.print_exc()

        inputs.append(messages)

    outputs = llm.generate(inputs, sampling_params)

    results = []
    for i, out in enumerate(outputs):

        assert type(out.outputs[0].text) == str
        hypothesis = out.outputs[0].text
        qid = qids[i]

        results.append({
            "idx": qid,
            "hypothesis": hypothesis,
        })

    return results
        
        





def main(args):

    if args.retrieval_method == 'retriever':
        assert args.retrieval_result_file != 'none', "retrieval_result_file should not be 'none' when retrieval_method is 'retriever'"
    elif args.retrieval_method in ['baseline', 'oracle']:
        assert args.retrieval_result_file == 'none', "retrieval_result_file should be 'none' when retrieval_method is 'baseline' or 'oracle'"
    else:
        raise ValueError("retrieval_method should be one of ['retriever', 'baseline', 'oracle']")

    print('#############################')
    print('input_file: ', args.input_file)
    print('model_path: ', args.model_path)
    print('retrieval_method: ', args.retrieval_method)
    print('retrieval_result_file: ', args.retrieval_result_file)
    print('topk_doc: ', args.topk_doc)
    print('##############################')


    with open(args.input_file, "r", encoding="utf-8") as f: # Json file
        in_data = json.load(f)

    if args.retrieval_result_file != 'none':
        with open(args.retrieval_result_file, "r", encoding="utf-8") as f:
            all_doc_scores = json.load(f)     



    qids = []
    prompts = []

    for entry in tqdm(in_data): # 'idx', 'question', 'num_gold_docs', 'gt_docs', 'paragraphs'

        if args.retrieval_method == 'retriever':
            topk_dox_idx = get_topk_doc_idx(all_doc_scores[entry['idx']], args.topk_doc)
            gt_doc_idx = None
        elif args.retrieval_method == 'baseline':
            topk_dox_idx = None
            gt_doc_idx = None
        elif args.retrieval_method == 'oracle':
            topk_dox_idx = None
            gt_doc_idx = entry['gt_docs']

        prompt = prepare_prompt_for_generation(
            docs=entry['paragraphs'],
            topk_doc_idx=topk_dox_idx,
            gt_doc_idx=gt_doc_idx,
            statement=entry['question']
        )

        qids.append(entry['idx'])
        prompts.append(prompt)




    results = vllm_eval(
        qids,
        prompts,
        args.model_path,
    )

    # Save the results to Jsonl file
    out_f = open(args.output_file, 'w')

    for result in results:
        qid = result['idx']
        hypothesis = result['hypothesis']

        out_f.write(json.dumps({
            "idx": qid,
            "hypothesis": hypothesis
        }) + '\n')

    out_f.close()




if __name__ == '__main__':
    args = parse_args()
    main(args)