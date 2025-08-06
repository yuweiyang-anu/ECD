### You need to replace the 'your_trained_model_path' of the actual model path and activate the corresponding conda enironment for different model evaluation

## Evaluation scripts for llava_next
python inference_on_charxiv_llava-next.py --split 'val' --model_type 'base' --mode 'descriptive' --save_name 'original_llava_next' --model_path ''

python inference_on_charxiv_llava-next.py --split 'val' --model_type 'base' --mode 'reasoning' --save_name 'original_llava_next' --model_path ''

python inference_on_charxiv_llava-next.py --split 'val' --model_type 'finetuning' --mode 'descriptive' --save_name 'llava_next_on_ECD' --model_path 'your_trained_model_path'

python inference_on_charxiv_llava-next.py --split 'val' --model_type 'finetuning' --mode 'reasoning' --save_name 'llava_next_on_ECD' --model_path 'your_trained_model_path'

cd eval_utils
pip install openai 
python evaluate.py --model_name 'llama3-llava-next-8b-hf' --split val --mode descriptive --output_name 'original_llava_next'
python evaluate.py --model_name 'llama3-llava-next-8b-hf' --split val --mode reasoning --output_name 'original_llava_next'

python evaluate.py --model_name 'llama3-llava-next-8b-hf' --split val --mode descriptive --output_name 'llava_next_on_ECD'
python evaluate.py --model_name 'llama3-llava-next-8b-hf' --split val --mode reasoning --output_name 'llava_next_on_ECD'

python get_stats.py --model_name 'llama3-llava-next-8b-hf' --split val --output_name 'original_llava_next'
python get_stats.py --model_name 'llama3-llava-next-8b-hf' --split val --output_name 'llava_next_on_ECD'

## Evaluation scripts for minicpm_v2_6
python inference_on_charxiv_minicpm_v2_6.py --split 'val' --model_type 'base' --mode 'descriptive' --save_name 'original_minicpm_v2_6' --model_path ''

python inference_on_charxiv_minicpm_v2_6.py --split 'val' --model_type 'base' --mode 'reasoning' --save_name 'original_minicpm_v2_6' --model_path ''

python inference_on_charxiv_minicpm_v2_6.py --split 'val' --model_type 'finetuning' --mode 'descriptive' --save_name 'minicpm_v2_6_on_ECD' --model_path 'your_trained_model_path'

python inference_on_charxiv_minicpm_v2_6.py --split 'val' --model_type 'finetuning' --mode 'reasoning' --save_name 'minicpm_v2_6_on_ECD' --model_path 'your_trained_model_path'

cd eval_utils
pip install openai
python evaluate.py --model_name 'minicpm-v2_6' --split val --mode descriptive --output_name 'original_minicpm_v2_6'
python evaluate.py --model_name 'minicpm-v2_6' --split val --mode reasoning --output_name 'original_minicpm_v2_6'

python evaluate.py --model_name 'minicpm-v2_6' --split val --mode descriptive --output_name 'minicpm_v2_6_on_ECD'
python evaluate.py --model_name 'minicpm-v2_6' --split val --mode reasoning --output_name 'minicpm_v2_6_on_ECD'

python get_stats.py --model_name 'minicpm-v2_6' --split val --output_name 'original_minicpm_v2_6'
python get_stats.py --model_name 'minicpm-v2_6' --split val --output_name 'minicpm_v2_6_on_ECD'

## Evaluation scripts for phi3v
pip install transformers==4.47.0 
python inference_on_charxiv_phi3v.py --split 'val' --model_type 'base' --mode 'descriptive' --save_name 'original_phi3v' --model_path ''

python inference_on_charxiv_phi3v.py --split 'val' --model_type 'base' --mode 'reasoning' --save_name 'original_phi3v' --model_path ''

python inference_on_charxiv_phi3v.py --split 'val' --model_type 'finetuning' --mode 'descriptive' --save_name 'full_phi3v_on_ECD' --model_path 'your_trained_model_path'

python inference_on_charxiv_phi3v.py --split 'val' --model_type 'finetuning' --mode 'descriptive' --save_name 'full_phi3v_on_ECD' --model_path 'your_trained_model_path'

cd eval_utils
pip install openai
python evaluate.py --model_name 'phi-3-vision' --split val --mode descriptive --output_name 'original_phi3v'
python evaluate.py --model_name 'phi-3-vision' --split val --mode reasoning --output_name 'original_phi3v'

python evaluate.py --model_name 'phi-3-vision' --split val --mode descriptive --output_name 'full_phi3v_on_ECD'
python evaluate.py --model_name 'phi-3-vision' --split val --mode reasoning --output_name 'full_phi3v_on_ECD'

python get_stats.py --model_name 'phi-3-vision' --split val --output_name 'original_phi3v'
python get_stats.py --model_name 'phi-3-vision' --split val --output_name 'full_phi3v_on_ECD'

## Evaluation scripts for qwen2_5_vl
python inference_on_charxiv_qwen2_5_vl.py --split 'val' --model_type 'base' --mode 'descriptive' --save_name 'original_qwen2_5_vl' --model_path ''

python inference_on_charxiv_qwen2_5_vl.py --split 'val' --model_type 'base' --mode 'reasoning' --save_name 'original_qwen2_5_vl' --model_path ''

python inference_on_charxiv_qwen2_5_vl.py --split 'val' --model_type 'finetuning' --mode 'descriptive' --save_name 'lora_qwen2_5_vl_on_ECD' --model_path 'your_trained_model_path'

python inference_on_charxiv_qwen2_5_vl.py --split 'val' --model_type 'finetuning' --mode 'reasoning' --save_name 'lora_qwen2_5_vl_on_ECD' --model_path 'your_trained_model_path'

cd eval_utils
pip install openai
python evaluate.py --model_name 'qwen2.5-vl-7b-instruct' --split val --mode descriptive --output_name 'original_qwen2_5_vl'
python evaluate.py --model_name 'qwen2.5-vl-7b-instruct' --split val --mode reasoning --output_name 'original_qwen2_5_vl'
python evaluate.py --model_name 'qwen2.5-vl-7b-instruct' --split val --mode descriptive --output_name 'lora_qwen2_5_vl_on_ECD'
python evaluate.py --model_name 'qwen2.5-vl-7b-instruct' --split val --mode reasoning --output_name 'lora_qwen2_5_vl_on_ECD'

python get_stats.py --model_name 'qwen2.5-vl-7b-instruct' --split val --output_name 'original_qwen2_5_vl'
python get_stats.py --model_name 'qwen2.5-vl-7b-instruct' --split val --output_name 'lora_qwen2_5_vl_on_ECD'