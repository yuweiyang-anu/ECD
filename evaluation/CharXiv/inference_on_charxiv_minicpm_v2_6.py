import os
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from transformers import AutoModel, AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # input/output
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
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
    output_dir = f'./infer_results/{args.save_name}'
    image_dir = 'ECD/public_benchmarks/charXiv/images/'

    input_file = os.path.join('ECD/public_benchmarks/charXiv/data/',
                              f"{args.mode}_{args.split}.json")
    print(f"Reading {input_file}...")
    with open(input_file) as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir, f'gen-{model_name}-{args.mode}_{args.split}.json')

    if args.mode == 'descriptive':
        from eval_utils.descriptive_utils import build_descriptive_quries
        queries = build_descriptive_quries(data, image_dir)
    elif args.mode == 'reasoning':
        from eval_utils.reasoning_utils import build_reasoning_queries
        queries = build_reasoning_queries(data, image_dir)
    else:
        raise ValueError("Mode not supported")

    print("Number of test problems to run:", len(queries))
    print("Evaluation mode:", args.mode)
    print("Output file:", output_file)

    for k in tqdm(queries):
        prompt = queries[k]['question']
        image_path = queries[k]["figure_path"]
        image = Image.open(image_path).convert('RGB')

        msgs = [{'role': 'user', 'content': [image, prompt]}]

        # Preparation for inference
        response = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)
        print('response:', response)
        queries[k]['response'] = response

    for k in queries:
        queries[k].pop("figure_path", None)
        queries[k].pop("question", None)

    try:
        print(f"Saving results to {output_file}...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w+") as f:
            json.dump(queries, f, indent=4)
        print(f"Results saved.")
    except Exception as e:
        print(e)
        print(f"Error in saving {output_file}")
