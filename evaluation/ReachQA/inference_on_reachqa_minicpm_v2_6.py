import os
import json
import torch
from PIL import Image
import argparse
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # input/output
    parser.add_argument('--model_type', type=str, default='ori', required=True)
    parser.add_argument('--save_name', type=str, default='ori', required=True)
    parser.add_argument('--model_path', type=str)
    args = parser.parse_args()

    if args.model_type == 'finetuning':
        model = AutoModel.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16)  # sdpa or flash_attention_2, no eager
        model = model.eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                                  trust_remote_code=True)

    else:  # 'ori'
        model = AutoModel.from_pretrained(
            'openbmb/MiniCPM-V-2_6',
            trust_remote_code=True,
            attn_implementation='sdpa',
            torch_dtype=torch.bfloat16)  # sdpa or flash_attention_2, no eager
        model = model.eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6',
                                                  trust_remote_code=True)

    model_name = 'minicpm-v2_6'
    temperature = 0
    result_list = []
    output_dir = f'./infer_results/{args.save_name}'

    input_file = 'ECD/public_benchmarks/ReachQA/test_data/test_data.json'
    print(f"Reading {input_file}...")
    with open(input_file) as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir,
                               f'gen-{model_name}-{args.save_name}.json')

    for k in tqdm(range(len(data))):
        json_object = {}
        prompt = data[k]['question']
        image_path = 'ECD/public_benchmarks/ReachQA/test_data/images/' + data[
            k]["image"]
        image = Image.open(image_path).convert('RGB')
        json_object["question"] = prompt
        json_object["gt_answer"] = data[k]['answer']
        json_object["split"] = data[k]["split"]
        json_object["image"] = data[k]["image"]

        msgs = [{'role': 'user', 'content': [image, prompt]}]

        # Preparation for inference
        response = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)
        print('response:', response)
        json_object["pred_answer"] = response

        result_list.append(json_object)

        print(f"Saving results to {output_file}...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w+") as f:
            json.dump(result_list, f, indent=4)
        print(f"Results saved.")
