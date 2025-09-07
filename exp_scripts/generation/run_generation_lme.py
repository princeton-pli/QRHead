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
    parser.add_argument('--input_file', type=str, required=True) # Json file: data
    parser.add_argument('--output_file', type=str, required=True) # Json file
        
    parser.add_argument('--model_path', type=str, required=True)

    parser.add_argument('--retrieval_method', type=str, required=True, choices=['oracle', 'baseline', 'retriever'], help='oracle, baseline, retriever')
    parser.add_argument('--retrieval_result_file', type=str, required=True) # 'none' if not using retriever as retrieval_method

    parser.add_argument('--topk_doc', type=int, required=True)
    return parser.parse_args()


def get_topk_doc_idx(doc_scores: Dict[str, float], k: int) -> List[str]:
    """Get the top-k docs based on scores."""

    # Sort docs by scores in descending order
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_idx for doc_idx, _ in sorted_docs[:k]]





# Find rounds based on the retrieval_result

# haystack_data: a list of dictionaries, each containing:
# - question_id
# - question_type
# - question
# - answer
# - question_date
# - haystack_dates: a list of dates
# - haystack_session_ids: a list of session ids
# - haystack_sessions: a list of sessions, each session is a list of turns
# - answer_session_ids

def extract_relevant_rounds(haystack_example: List[Dict], topk_round_idx: List[str], gt_round_idx: List[str]):

    # Create a new example object with the same fields
    new_example = {
        "question_id": haystack_example["question_id"],
        "question_type": haystack_example["question_type"],
        "question": haystack_example["question"],
        "answer": haystack_example["answer"],
        "question_date": haystack_example["question_date"],
        "answer_session_ids": haystack_example["answer_session_ids"],
    }

    if topk_round_idx is None and gt_round_idx is None: # baseline
        new_example["haystack_sessions"] = haystack_example["haystack_sessions"]
        new_example["haystack_session_ids"] = haystack_example["haystack_session_ids"]
        new_example["haystack_dates"] = haystack_example["haystack_dates"]
        return new_example

    elif topk_round_idx is None and gt_round_idx is not None: # oracle
        result_round_idx = gt_round_idx

    elif topk_round_idx is not None and gt_round_idx is None: # retriever
        # Get the top-k docs
        result_round_idx = topk_round_idx # example: [gpt4_2655b836__answer_4be1b6b4_3__2, ...]
        assert len(set(result_round_idx)) == len(result_round_idx), "Duplicate doc idx found in the top-k context"
        assert len(result_round_idx) == args.topk_doc, f"Top-k context length should be {args.topk_doc}, but got {len(result_round_idx)}"

    else:
        raise ValueError("Both rerank_example and gt_doc_ids are provided. Please provide only one or none of them.")

    all_sessions = haystack_example["haystack_sessions"]
    all_session_ids = haystack_example["haystack_session_ids"]
    all_dates = haystack_example["haystack_dates"]

    filtered_sessions = []
    filtered_session_ids = []
    filtered_dates = []

    for curr_session, curr_session_id, curr_date in zip(all_sessions, all_session_ids, all_dates):
        # Check if the round ids in the current session match any of the result_round_idx
        # If so, add the rounds to the new session
        new_session = []
        
        for round in curr_session:
            round_id = round.get("round_id")
            if round_id in result_round_idx:
                new_session.append(round)

        if len(new_session) > 0:
            filtered_sessions.append(new_session)
            filtered_session_ids.append(curr_session_id)
            filtered_dates.append(curr_date)
    
    # Add filtered data to the new example
    new_example["haystack_sessions"] = filtered_sessions
    new_example["haystack_session_ids"] = filtered_session_ids
    new_example["haystack_dates"] = filtered_dates
    return new_example






def generate_prompt(curr_haystack_dates, curr_haystack_sessions, curr_question, curr_question_date):

        answer_prompt_template = 'I will give you several history chats between you and a user. Please answer the question based on the relevant chat history.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer:'

        # clean up
        retrieved_chunks_cleaned = []
        for session_date, session_entry in zip(curr_haystack_dates, curr_haystack_sessions):
            for turn_entry in session_entry:
                if type(turn_entry) == dict and 'has_answer' in turn_entry:
                    turn_entry.pop('has_answer')
                if type(turn_entry) == dict and 'round_id' in turn_entry:
                    turn_entry.pop('round_id')
            retrieved_chunks_cleaned.append((session_date, session_entry))

        # sort sessions by their dates
        retrieved_chunks_cleaned.sort(key=lambda x: x[0])

        history_string = ""
        for i, cur_item in enumerate(retrieved_chunks_cleaned):
            chunk_date, chunk_entry = cur_item

            sess_string = '\n' + json.dumps(chunk_entry)
            history_string += '\n### Session {}:\nSession Date: {}\nSession Content:\n{}\n'.format(i+1, chunk_date, sess_string)
        
        prompt = answer_prompt_template.format(history_string, curr_question_date, curr_question)
        return prompt
    




def vllm_eval(prompts, model_dir):

    # max_model_len=131072 for Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct, Llama-3.2-3B-Instruct
    #######################################################################
    # TODO: Change max_model_len if using other models
    #######################################################################
    llm = LLM(model=model_dir, max_model_len=131072, tensor_parallel_size=torch.cuda.device_count())

    try:
        tokenizer = llm.get_tokenizer()
        print("Tokenizer:", tokenizer.chat_template)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        print("Using custom tokenizer")

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=500,
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
    return outputs







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
    
    with open(args.input_file, "r") as f: # Json file
        in_data = json.load(f)

    if args.retrieval_result_file != 'none':
        with open(args.retrieval_result_file, "r") as f:
            all_doc_scores = json.load(f)     

        


    qids = []
    prompts = []

    for i, haystack_example in tqdm(enumerate(in_data)):

        if args.retrieval_method == 'retriever':
            assert haystack_example['question_id'] in all_doc_scores, f"Question ID {haystack_example['question_id']} not found in rerank data"
            topk_doc_idx = get_topk_doc_idx(all_doc_scores[haystack_example['question_id']], args.topk_doc)
            gt_doc_idx = None
        elif args.retrieval_method == 'baseline':
            topk_doc_idx = None
            gt_doc_idx = None
        elif args.retrieval_method == 'oracle':
            topk_doc_idx = None
            gt_doc_idx = haystack_example['ground_truth_rounds']

        new_sample = extract_relevant_rounds(haystack_example, topk_doc_idx, gt_doc_idx)

        prompt = generate_prompt(
            curr_haystack_dates=new_sample['haystack_dates'],
            curr_haystack_sessions=new_sample['haystack_sessions'],
            curr_question=new_sample['question'],
            curr_question_date=new_sample['question_date'],
        )

        qids.append(new_sample['question_id'])
        prompts.append(prompt)




    outputs = vllm_eval(
        prompts,
        args.model_path,
    )

    # Save the results to Jsonl file
    out_f = open(args.output_file, 'w')

    for i, out in enumerate(outputs):

        assert type(out.outputs[0].text) == str
        hypothesis = out.outputs[0].text
        qid = qids[i]

        # Save the results to the output file
        out_f.write(json.dumps({
            "idx": qid,
            "hypothesis": hypothesis
        }) + '\n')
    out_f.close()




if __name__ == '__main__':
    args = parse_args()
    main(args)