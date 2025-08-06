import os
import json
import torch
import argparse
from tqdm import tqdm

from PIL import Image
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # input/output
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
    else:  # 'ori'
        processor = AutoProcessor.from_pretrained(model_id,
                                                  trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype="auto").cuda()

    user_prompt = '<|user|>\n'
    assistant_prompt = '<|assistant|>\n'
    prompt_suffix = "<|end|>\n"

    model_name = 'phi-3-vision'

    temperature = 0
    result_list = []
    output_dir = f'./infer_results/{args.save_name}'

    input_file = 'ECD/public_benchmarks/ChartBench/test_data.json'
    print(f"Reading {input_file}...")
    with open(input_file) as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir,
                               f'gen-{model_name}-{args.save_name}.json')
    print("Output file:", output_file)

    for k in tqdm(range(len(data))):
        json_object = {}
        prompt = data[k]['query']
        print('===question:===', prompt)
        image_path = 'ECD/public_benchmarks/ChartBench/' + data[k]["image"]
        image = Image.open(image_path).convert('RGB')
        json_object["id"] = data[k]['id']
        json_object["type"] = data[k]['type']
        json_object["question"] = prompt
        json_object["gt_answer"] = data[k]['label']
        json_object["image"] = data[k]["image"]

        query = f"{user_prompt}<|image_1|>\n{prompt}{prompt_suffix}{assistant_prompt}"
        inputs = processor(query, image, return_tensors="pt").to("cuda:0")

        print('===query:===', query)
        generate_ids = model.generate(
            **inputs,
            temperature=0,
            max_new_tokens=1000,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]

        response = processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        json_object["pred_answer"] = response
        result_list.append(json_object)

        print(f"Saving results to {output_file}...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w+") as f:
            json.dump(result_list, f, indent=4)
        print(f"Results saved.")
