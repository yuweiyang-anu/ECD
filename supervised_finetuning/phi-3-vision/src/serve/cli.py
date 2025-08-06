from transformers import TextStreamer
from PIL import Image
import torch
import requests
from io import BytesIO
from decord import VideoReader, cpu
import argparse
import warnings
from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init

warnings.filterwarnings("ignore")

def is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def encode_video(video_path, max_num_frames=10):
    def uniform_sample(l, n):
        gap = len(l) / n
        idxs = [int(i * gap + gap / 2) for i in range(n)]
        return [l[i] for i in idxs]

    vr = VideoReader(video_path, ctx=cpu(0))
    sample_fps = round(vr.get_avg_fps() / 1)  # FPS
    frame_idx = [i for i in range(0, len(vr), sample_fps)]
    if len(frame_idx) > max_num_frames:
        frame_idx = uniform_sample(frame_idx, max_num_frames)
    frames = vr.get_batch(frame_idx).asnumpy()
    frames = [Image.fromarray(v.astype('uint8')) for v in frames]
    return frames

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def main(args):

    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    
    use_flash_attn = True

    if args.disable_flash_attention:
        use_flash_attn = False

    processor, model = load_pretrained_model(model_path = args.model_path, model_base=args.model_base, 
                                             model_name=model_name, device_map=args.device, 
                                             load_4bit=args.load_4bit, load_8bit=args.load_8bit,
                                             device=args.device, use_flash_attn=use_flash_attn
    )

    messages = []

    image_list = []
    if is_video_file(args.image_file):
        image_list = encode_video(args.image_file, max_frames=args.max_frames)
    else:
        if ',' in args.image_file:
            image_files = args.image_file.split(',')
            for img_file in image_files:
                image_list.append(load_image(img_file.strip()))
        else:
            image_list.append(load_image(args.image_file))

    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": True if args.temperature > 0 else False,
        "repetition_penalty": args.repetition_penalty,
    }
    
    placeholder = ""

    while True:
        try:
            inp = input(f"User: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"Assistant: ", end="")

        if image_list is not None and len(messages) < 2:
            # only putting the image token in the first turn of user.
            for i in range(len(image_list)):
                placeholder += f"<|image_{i+1}|>\n"
            
            inp = placeholder + inp

        messages.append({"role": "user", "content": inp})

        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(prompt, image_list, return_tensors="pt").to(args.device)
        
        streamer = TextStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            generate_ids = model.generate(
                **inputs, 
                streamer=streamer,
                **generation_args,
                use_cache=True,
                eos_token_id=processor.tokenizer.eos_token_id
            )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        outputs = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        messages.append({"role":"assistant", "content": outputs})

        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default="microsoft/Phi-3-vision-128k-instruct")
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--max-frames", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--disable_flash_attention", action="store_true")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)