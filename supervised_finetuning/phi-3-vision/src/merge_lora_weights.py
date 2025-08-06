import argparse
from utils import get_model_name_from_path, load_pretrained_model, modify_config_file

def merge_lora(args):
    model_name = get_model_name_from_path(args.model_path)
    processor, model = load_pretrained_model(model_path=args.model_path, model_base=args.model_base,
                                             model_name=model_name, device_map='cpu',
                                             )

    if args.safe_serialization:
        state_dict = model.state_dict()
        state_dict = {k:v for k, v in state_dict.items() if "wte" not in k}
        model.save_pretrained(args.save_model_path, state_dict=state_dict, safe_serialization=True)
        processor.chat_template = processor.tokenizer.chat_template
        processor.save_pretrained(args.save_model_path)

    else:
        model.save_pretrained(args.save_model_path, safe_serialization=False)
        processor.chat_template = processor.tokenizer.chat_template
        processor.save_pretrained(args.save_model_path)

    modify_config_file(args.save_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, required=True)
    parser.add_argument("--save-model-path", type=str, required=True)
    parser.add_argument("--safe-serialization", action='store_true')

    args = parser.parse_args()

    merge_lora(args)