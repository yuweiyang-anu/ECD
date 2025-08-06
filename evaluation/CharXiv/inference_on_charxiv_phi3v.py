import os
import json
import torch
import argparse
from tqdm import tqdm

from PIL import Image
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # input/output
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='ori', required=True)
    parser.add_argument('--save_name', type=str, default='ori', required=True)
    parser.add_argument('--model_path', type=str)
    args = parser.parse_args()

    model_id = "microsoft/Phi-3-vision-128k-instruct"

    kwargs = {}
    kwargs['torch_dtype'] = torch.bfloat16

    if args.model_type == 'finetuning':
        processor = AutoProcessor.from_pretrained(model_id,
                                                  trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, trust_remote_code=True,
            torch_dtype="auto").cuda()
    else:
        processor = AutoProcessor.from_pretrained(model_id,
                                                  trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype="auto").cuda()

    model_name = 'phi-3-vision'

    temperature = 0
    result_list = []
    output_dir = f'./infer_results/{args.save_name}'
    image_dir = 'ECD/public_benchmarks/CharXiv/images/'
    input_file = os.path.join('ECD/public_benchmarks/charXiv/data/',
                              f"{args.mode}_{args.split}.json")

    print(f"Reading {input_file}...")
    with open(input_file) as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir, f'gen-{model_name}-{args.mode}_{args.split}.json')
    print("Output file:", output_file)

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
        query = queries[k]['question']
        image_path = queries[k]["figure_path"]
        image = Image.open(image_path).convert('RGB')

        messages = [{'role': 'user', 'content': f"<|image_1|>\n{query}"}]
        prompt = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        print('query:', query)
        inputs = processor(prompt, [image], return_tensors="pt").to("cuda:0")
        generation_args = {
            "max_new_tokens": 500,
            "temperature": 0.0,
            "do_sample": False,
        }
        generate_ids = model.generate(
            **inputs,
            eos_token_id=processor.tokenizer.eos_token_id,
            **generation_args)
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        result = processor.batch_decode(generate_ids,
                                        skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)[0]
        print('response:', result)
        queries[k]['response'] = result

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
