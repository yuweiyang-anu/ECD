### You need to replace the 'your_trained_model_path' of the actual model path and activate the corresponding conda enironment for different model evaluation

pip install langchain
## Evaluation scripts for llava_next
python inference_on_chartx_llava_next.py --model_type 'base' --save_name 'original_llava_next' --model_path ''

python inference_on_chartx_llava_next.py --model_type 'finetuning' --save_name 'llava_next_on_ECD' --model_path 'your_trained_model_path'

python evaluate_on_chartx.py --infer_data_path './infer_results/original_llava_next/gen-llama3-llava-next-8b-hf-original_llava_next.json' --output_file './infer_results/original_llava_next/score-llama3-llava-next-8b-hf-original_llava_next.txt'

python evaluate_on_chartx.py --infer_data_path './infer_results/llava_next_on_ECD/gen-llama3-llava-next-8b-hf-llava_next_on_ECD.json' --output_file './infer_results/llava_next_on_ECD/score-llama3-llava-next-8b-hf-llava_next_on_ECD.txt'

## Evaluation scripts for minicpm_v2_6
python inference_on_chartx_minicpm_v2_6.py --model_type 'base' --save_name 'original_minicpm_v2_6' --model_path ''

python inference_on_chartx_minicpm_v2_6.py --model_type 'finetuning' --save_name 'minicpm_v2_6_on_ECD' --model_path 'your_trained_model_path'

python evaluate_on_chartx.py --infer_data_path './infer_results/original_minicpm_v2_6/gen-minicpm-v2_6-original_minicpm_v2_6.json' --output_file './infer_results/original_minicpm_v2_6/score-minicpm-v2_6-original_minicpm_v2_6.txt'

python evaluate_on_chartx.py --infer_data_path './infer_results/minicpm_v2_6_on_ECD/gen-minicpm-v2_6-minicpm_v2_6_on_ECD.json' --output_file './infer_results/minicpm_v2_6_on_ECD/score-minicpm-v2_6-minicpm_v2_6_on_ECD.txt'

## Evaluation scripts for phi3v
pip install transformers==4.47.0
python inference_on_chartx_phi3v.py --model_type 'base' --save_name 'original_phi3v' --model_path ''

python inference_on_chartx_phi3v.py --model_type 'finetuning' --save_name 'full_phi3v_on_ECD' --model_path 'your_trained_model_path'

python evaluate_on_chartx.py --infer_data_path './infer_results/original_phi3v/gen-phi-3-vision-original_phi3v.json' --output_file './infer_results/original_phi3v/score-phi-3-vision-original_phi3v.txt'

python evaluate_on_chartx.py --infer_data_path './infer_results/full_phi3v_on_ECD/gen-phi-3-vision-full_phi3v_on_ECD.json' --output_file './infer_results/full_phi3v_on_ECD/score-phi-3-vision-full_phi3v_on_ECD.txt'

## Evaluation scripts for qwen2_5_vl
python inference_on_chartx_qwen2_5_vl.py --model_type 'base' --save_name 'original_qwen2_5_vl' --model_path ''

python inference_on_chartx_qwen2_5_vl.py --model_type 'finetuning' --save_name 'lora_qwen2_5_vl_on_ECD' --model_path 'your_trained_model_path'

python evaluate_on_chartx.py --infer_data_path './infer_results/original_qwen2_5_vl/gen-qwen2.5-vl-7b-instruct-original_qwen2_5_vl.json' --output_file './infer_results/original_qwen2_5_vl/score-qwen2.5-vl-7b-instruct-original_qwen2_5_vl.txt'

python evaluate_on_chartx.py --infer_data_path './infer_results/lora_qwen2_5_vl_on_ECD/gen-qwen2.5-vl-7b-instruct-lora_qwen2_5_vl_on_ECD.json' --output_file './infer_results/lora_qwen2_5_vl_on_ECD/score-qwen2.5-vl-7b-instruct-lora_qwen2_5_vl_on_ECD.txt'