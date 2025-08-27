import json
import re
import argparse


def parse_file(file_path):
    if file_path is None:
        return None
    
    results = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            idx = data.get("idx")
            hypothesis = data.get("hypothesis")

            try:
                tag = "answer" 
                pattern = re.compile(rf"<{tag}>(.*?)</{tag}>", re.DOTALL)
                match = pattern.findall(hypothesis)[0].strip().lower()
            except:
                match = None

            results.append({
                "idx": idx,
                "answer": match
            })
    return results



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--TRUE_generation_result_file", type=str, default=None, help="Path to the TRUE generation result file")
    parser.add_argument("--FALSE_generation_result_file", type=str, default=None, help="Path to the FALSE generation result file")
    args = parser.parse_args()

    TRUE_generation_result_file = args.TRUE_generation_result_file
    FALSE_generation_result_file = args.FALSE_generation_result_file

    TRUE_answers = parse_file(TRUE_generation_result_file)
    FALSE_answers = parse_file(FALSE_generation_result_file)

    # Handle cases where one or both files are None
    if TRUE_answers is None and FALSE_answers is None:
        print("Both file paths are None. Nothing to evaluate.")
    elif TRUE_answers is None:
        # Only check if answers for FALSE_generation_path are 'false'
        correct_answers = sum(1 for ans in FALSE_answers if ans['answer'] == "false")
        accuracy = correct_answers / len(FALSE_answers)
        print(f"Evaluating only FALSE file. Accuracy (answers = 'false'): {accuracy:.4f}")
    elif FALSE_answers is None:
        # Only check if answers for TRUE_generation_path are 'true'
        correct_answers = sum(1 for ans in TRUE_answers if ans['answer'] == "true")
        accuracy = correct_answers / len(TRUE_answers)
        print(f"Evaluating only TRUE file. Accuracy (answers = 'true'): {accuracy:.4f}")
    else:
        # Both files exist, accuracy stores binary values, 1 when TRUE_answer is true and FALSE_answer is false, 0 otherwise
        accuracy_results = []
        count_none = 0
        for TRUE_ans, FALSE_ans in zip(TRUE_answers, FALSE_answers):
            if TRUE_ans['answer'] == "true" and FALSE_ans['answer'] == "false":
                accuracy_results.append(1)
            else:
                accuracy_results.append(0)

            if TRUE_ans['answer'] is None or FALSE_ans['answer'] is None:
                count_none += 1

        print(f"Number of entries with None answers: {count_none}")

        # Calculate the accuracy
        accuracy = sum(accuracy_results) / len(accuracy_results)
        print(f"Accuracy (TRUE = 'true' and FALSE = 'false'): {accuracy:.4f}")
