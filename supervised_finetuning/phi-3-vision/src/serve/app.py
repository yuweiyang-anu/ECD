import argparse
from threading import Thread
import gradio as gr
from PIL import Image
from src.utils import load_pretrained_model, get_model_name_from_path, disable_torch_init
from transformers import TextIteratorStreamer
from functools import partial
import warnings
from decord import VideoReader, cpu
import requests
from io import BytesIO

warnings.filterwarnings("ignore")

def is_video_file(filename):
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.mpeg']
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

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

def bot_streaming(message, history, generation_args, max_num_frames=10):
    images = []
    videos = []
    image_counter = 1
    if message["files"]:
        for file_item in message["files"]:
            if isinstance(file_item, dict):
                file_path = file_item["path"]
            else:
                file_path = file_item
            if is_video_file(file_path):
                videos.append(file_path)
            else:
                images.append(file_path)

    conversation = []
    input_img = []
    has_image = False
    for user_turn, assistant_turn in history:
        if isinstance(user_turn, tuple):
            placeholder = ""
            file_paths = user_turn[0]
            has_image = True
            if not isinstance(file_paths, list):
                file_paths = [file_paths]
            for idx, file_path in enumerate(file_paths):
                if is_video_file(file_path):
                    frames = encode_video(file_path, max_num_frames=max_num_frames)
                    for i in range(len(frames)):
                        placeholder += f"<|image_{image_counter}|>\n"
                        image_counter += 1
                    input_img.extend(frames)
                else:
                    placeholder += f"<|image_{image_counter}|>\n"
                    image_counter += 1
                    input_img.append(load_image(file_path))
        else:
            if has_image:
                user_content = placeholder + user_turn
                has_image = False
            else:
                user_content = user_turn
        
            conversation.append({"role": "user", "content": user_content})

        if assistant_turn is not None:
            assistant_content = assistant_turn
            conversation.append({"role": "assistant", "content": assistant_content})

    for image in images:
        input_img.append(load_image(image))
    for video in videos:
        frames = encode_video(video, max_num_frames=max_num_frames)
        input_img.extend(frames)
    user_text = message['text']
    placeholder = ""
    if message["files"]:
        for _ in message["files"]:
            placeholder += f"<|image_{image_counter}|>\n"
            image_counter += 1
    user_content = placeholder + user_text
    conversation.append({"role": "user", "content": user_content})

    prompt = processor.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    
    if input_img:
        inputs = processor(prompt, images=input_img, return_tensors="pt").to(device)
    else:
        inputs = processor(prompt, return_tensors="pt").to(device)

    image_counter = 1

    streamer = TextIteratorStreamer(processor.tokenizer, **{"skip_special_tokens": True, "skip_prompt": True, 'clean_up_tokenization_spaces':False,}) 
    generation_kwargs = dict(inputs, streamer=streamer, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    buffer = ""
    for new_text in streamer:
        buffer += new_text
        yield buffer

def main(args):

    global processor, model, device

    device = args.device
    
    disable_torch_init()

    use_flash_attn = True
    
    model_name = get_model_name_from_path(args.model_path)
    
    if args.disable_flash_attention:
        use_flash_attn = False

    processor, model = load_pretrained_model(model_base = args.model_base, model_path = args.model_path, 
                                                device_map=args.device, model_name=model_name, 
                                                load_4bit=args.load_4bit, load_8bit=args.load_8bit,
                                                device=args.device, use_flash_attn=use_flash_attn
    )

    chatbot = gr.Chatbot(scale=2)
    chat_input = gr.MultimodalTextbox(interactive=True, file_types=["image"], placeholder="Enter message or upload file...",
                                  show_label=False)
    
    generation_args = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": True if args.temperature > 0 else False,
        "repetition_penalty": args.repetition_penalty,
    }
    
    bot_streaming_with_args = partial(bot_streaming, generation_args=generation_args, max_num_frames=args.max_frames)

    with gr.Blocks(fill_height=True) as demo:
        gr.ChatInterface(
            fn=bot_streaming_with_args,
            title="Phi-3-Vision",
            stop_btn="Stop Generation",
            multimodal=True,
            textbox=chat_input,
            chatbot=chatbot,
        )


    demo.queue(api_open=False)
    demo.launch(show_api=False, share=False, server_name='0.0.0.0')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default="microsoft/Phi-3-vision-128k-instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--max-frames", type=int, default=10)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--disable_flash_attention", action="store_true")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)