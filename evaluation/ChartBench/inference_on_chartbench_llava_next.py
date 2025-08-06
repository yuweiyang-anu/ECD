import os
import re
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # input/output
    parser.add_argument('--model_type', type=str, default='ori', required=True)
    parser.add_argument('--save_name', type=str, default='ori', required=True)
    parser.add_argument('--model_path', type=str)
    args = parser.parse_args()

    if args.model_type == 'finetuning':
        processor = LlavaNextProcessor.from_pretrained(args.model_path)
        model = LlavaNextForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float16, device_map="auto")
    else:
        processor = LlavaNextProcessor.from_pretrained(
            "llava-hf/llama3-llava-next-8b-hf")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llama3-llava-next-8b-hf",
            torch_dtype=torch.float16,
            device_map="auto")

    model_name = 'llama3-llava-next-8b-hf'

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

    for k in tqdm(range(0, len(data))):
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

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image"
                    },
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation,
                                               add_generation_prompt=True)

        inputs = processor(images=image, text=prompt,
                           return_tensors="pt").to(model.device)

        # autoregressively complete prompt
        output = model.generate(**inputs, max_new_tokens=100)
        text_output = processor.decode(output[0], skip_special_tokens=True)
        response = re.search(r'(?<=assistant)(.*)', text_output,
                             re.DOTALL).group(1).strip()
        print('===response:===', response)

        json_object["pred_answer"] = response

        result_list.append(json_object)

        print(f"Saving results to {output_file}...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w+") as f:
            json.dump(result_list, f, indent=4)
        print(f"Results saved.")
