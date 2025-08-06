import os
import re
import json
import torch
import argparse
from tqdm import tqdm

from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import requests

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
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float16, device_map="auto")
    else:
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype="auto",
            device_map="auto")

    model_name = 'qwen2.5-vl-7b-instruct'

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
        # image = Image.open(image_path).convert('RGB')

        messages = [{
            "role":
            "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": prompt
                },
            ],
        }]

        text = processor.apply_chat_template(messages,
                                             tokenize=False,
                                             add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = processor.batch_decode(generated_ids_trimmed,
                                          skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)

        print('===response:===', response[0])

        queries[k]['response'] = response[0]

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
