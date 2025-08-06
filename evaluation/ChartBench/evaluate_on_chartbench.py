import re
import json
import logging
import argparse
from openai import OpenAI
from langchain import PromptTemplate
from langchain import FewShotPromptTemplate


def eval_gpt_acc(question, answer_gt, answer_pred):
    client = OpenAI(api_key="your_openai_key")

    examples = [
        {
            "query":
            "<question> What was the incremental increase in revenue from 2020 to 2021? <groundtruth answer> 5 million $ <answer> 20\n</s>",
            "answer": "False"
        },
        {
            "query":
            "<question> What percentage of government spending was allocated to infrastructure in 2020? <groundtruth answer> 10% <answer> 14-4=10\n</s>",
            "answer": "True"
        },
        {
            "query":
            "<question> What is the total production of Wind Energy in the four months from January to April 2021? <groundtruth answer> 2300 MW <answer> The total production of Wind Energy in the four months from January to April 2021 is 2450 MW.",
            "answer": "True"
        },
        {
            "query":
            "<question> What is the total of manufactured goods for UK and Germany combined? <groundtruth answer> 5 <answer> Five",
            "answer": "True"
        },
    ]

    # create a example template
    example_template = """
    User: {query}
    AI: {answer}
    """

    # create a prompt example from above template
    example_prompt = PromptTemplate(input_variables=["query", "answer"],
                                    template=example_template)

    # instruction
    prefix = f"""Given multiple question-answer pairs and the corresponding predictions, evaluate the correctness of predictions. The output should be only "True" or "False". Note that if the groundtruth answer is a numeric value with/without the unit, impose 5% error tolerance to the answer, e.g., the answer of 95 is marked as correct when groundtruth value is 100 million."""
    # and the suffix our user input and output indicator
    suffix = """
    User: {query}
    AI: """

    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
        example_separator="\n\n")

    query = f"<question> {question} <groundtruth answer> {answer_gt} <answer> {answer_pred}"

    iteration = 0
    while iteration < 10:
        try:
            completion = client.chat.completions.create(
                messages=[{
                    "role":
                    "user",
                    "content":
                    few_shot_prompt_template.format(query=query)
                }],
                model="gpt-4o",
                n=1,
                max_tokens=512,
                temperature=0,
                top_p=1,
                seed=42,
            ).choices[0].message.content
            break
        except Exception as e:
            iteration += 1
            print(f"===== Retry: {iteration} =====")
            print(f"Error occurs when calling API: {e}")
            continue

    # data_gen = completion.choices[0].message['content']
    data_gen = completion
    if 'True' in data_gen:
        acc = 1
    if 'False' in data_gen:
        acc = 0

    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_data_path",
                        required=True,
                        help="Path to the inference data")
    parser.add_argument("--output_file", type=str, default="")
    args = parser.parse_args()
    infer_result = args.infer_data_path

    len_sum = 0

    log_file = args.output_file
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logging.info('infer_result:' + infer_result)

    qa_score_set_total = []

    with open(infer_result, encoding='utf-8') as json_file:
        data = json.load(json_file)

    # Initialize counters
    qa_count = 0
    correct_predictions = 0
    correct_list = []

    # Iterate through each entry in the data
    for ind, entry in enumerate(data):
        # Check if QA is Acc+
        if entry['type']['QA'] == 'Acc+':
            qa_count += 1

            # Use regular expression to check if 'yes' or 'no' appears in pred_answer
            pred_answer_match = re.search(r'\b(yes|no)\b',
                                          entry['pred_answer'], re.IGNORECASE)

            # If match found, extract the answer
            if pred_answer_match:
                pred_answer = pred_answer_match.group(0).lower()
                gt_answer = entry['gt_answer'].lower()

                if pred_answer == gt_answer:
                    correct_predictions += 1
                    correct_list.append(1)
                else:
                    correct_list.append(0)
            else:
                correct_list.append(0)

    print(f'Binary Accuracy:', correct_predictions / qa_count)

    qa_score_set_total = []
    for ind, entry in enumerate(data):
        # Check if QA is Acc+
        if entry['type']['QA'] == 'GPT-acc':
            qa_count += 1

            print(f'======Evaluating: {ind+1}======')
            # Check if the answer is 'yes' or 'no' using regex (case-insensitive)
            pred_answer = entry['pred_answer'].strip()
            gt_answer = entry['gt_answer'].strip()
            question = entry['question'].strip()

            qa_score = eval_gpt_acc(question, gt_answer, pred_answer)
            logging.info(f'***Score of {ind}***:' + str(qa_score))
            qa_score_set_total.append(qa_score)

    qa_score_total = sum(qa_score_set_total) / len(qa_score_set_total)
    print('NQA score:', qa_score_total)
    logging.info('*************** Performance *****************')
    logging.info('average')
    logging.info('%.4f' % qa_score_total)
