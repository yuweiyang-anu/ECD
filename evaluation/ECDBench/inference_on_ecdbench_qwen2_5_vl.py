import os
import re
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # input/output
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
    result_list = []
    output_dir = f'./infer_results/{args.save_name}'

    input_file = 'ECD/public_benchmarks/ECDBench/ECD_Bench_All.json'
    print(f"Reading {input_file}...")
    with open(input_file) as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir,
                               f'gen-{model_name}-{args.save_name}.json')

    for k in tqdm(range(len(data))):
        json_object = {}
        prompt = data[k]['question']
        image_path = 'ECD/public_benchmarks/ECDBench/rendered_images/' + data[
            k]["image_id"]
        image = Image.open(image_path).convert('RGB')
        json_object["question"] = prompt
        json_object["gt_answer"] = data[k]['answer']
        json_object["split"] = data[k]["split"]

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
        response = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]

        print('===response:===', response)

        json_object["pred_answer"] = response
        result_list.append(json_object)

        print(f"Saving results to {output_file}...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w+") as f:
            json.dump(result_list, f, indent=4)
        print(f"Results saved.")