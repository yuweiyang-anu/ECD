import re
import json
import argparse
from openai import OpenAI

##############################################
Answer_Judge_Prompt = """Your task is to rigorously evaluate whether the VLM's prediction aligns with the expected answer for a question regarding a chart (note: the chart image itself is not provided here). The evaluation should focus solely on factual alignment between the prediction and the ground truth. Minor differences in wording or phrasing are acceptable as long as the core meaning remains consistent. 

For numerical answers, **a margin of error up to 5% is acceptable** unless explicitly stated otherwise in the question. However, partial correctness or incomplete responses should not be considered correct.

- Question: {question}
- Expected Answer: {answer}
- Prediction: {prediction}

Please respond using the following format:
Correctness: (Yes or No)
"""
##############################################

MAX_RETRIES = 3


def post_process_model_response(response):
    # Regular expression to match the rewritten answer with case insensitivity
    match = re.search(r"Correctness:\s*(.*)", response,
                      re.IGNORECASE | re.DOTALL)

    if match:
        answer_string = match.group(1).strip()
        if "yes" in answer_string.lower():
            return 1
        elif "no" in answer_string.lower():
            return 0
        else:
            return -1
    else:
        print("Failed to extract the correctness")
        return -1


def create_chat_response_by_messages(
    model="gpt-4o",
    client=None,
    messages=None,
    headers=None,
    max_tokens=256,
    temperature=0,
    top_p=0.95,
):
    retries = 0
    while retries < MAX_RETRIES:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                extra_headers=headers,
            )
            return response.choices[0].message.content

        except (TypeError, ValueError) as e:
            print(
                f"Error occurred: {e}. Retrying {retries + 1}/{MAX_RETRIES}..."
            )
            retries += 1

    print("Max retries reached. Returning None.")
    return None


def llm_evaluation(model, client, infer_data_path, output_file):
    # 1. load data
    with open(infer_data_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    eval_results = []
    # 2. start evaluation
    for index, item in enumerate(data):
        print('Generating rating for Image:', index)
        eval_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role":
                "user",
                "content":
                Answer_Judge_Prompt.format(
                    question=item["question"],
                    answer=item["gt_answer"],
                    prediction=item["pred_answer"],
                ),
            },
        ]

        ### Call LLM
        print("\nCalling OpenAI for Judge...")
        output = create_chat_response_by_messages(
            model=model,
            client=client,
            messages=eval_messages,
            max_tokens=4096,
            temperature=0,
            top_p=0.95,
        )
        item['output'] = output
        item["correctness"] = post_process_model_response(output)
        print('correctness:', item["correctness"])
        eval_results.append(item)

    # 3. post process
    correct_count = 0
    split_correct_counts = {
    }  # Dictionary to store correct counts for each Split
    split_total_counts = {}  # Dictionary to store total counts for each Split

    for item in eval_results:
        # Get the split name for the current record
        correctness = item["correctness"]
        split = item["split"]

        # Update counts for the current split
        if split not in split_correct_counts:
            split_correct_counts[split] = 0
            split_total_counts[split] = 0

        if correctness == 1:
            split_correct_counts[split] += 1
        split_total_counts[split] += 1

        # Update the overall correct count
        correct_count += 1 if correctness == 1 else 0

    # 4. calculate total score
    total_count = len(data)
    accuracy = 100 * correct_count / total_count if total_count > 0 else 0

    # Print total accuracy
    print(f"total: {total_count}\tacc: {accuracy:.2f}")

    # Print accuracy for each split
    for split_name, total in split_total_counts.items():
        split_accuracy = 100 * split_correct_counts[
            split_name] / total if total > 0 else 0
        print(f"{split_name}: {total}\tacc: {split_accuracy:.2f}")

    # 5. save results
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {output_file}")


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--infer_data_path", type=str, default="")
    parser.add_argument("--output_file", type=str, default="")

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    print(args)

    openai_client = OpenAI(api_key="your_own_openai_key")

    llm_evaluation(model=args.model_name,
                   client=openai_client,
                   infer_data_path=args.infer_data_path,
                   output_file=args.output_file)
