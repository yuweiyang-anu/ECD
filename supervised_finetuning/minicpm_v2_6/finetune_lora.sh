# huggingface-cli login
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="VLM-SFT-on-ECD"
export WANDB_API_KEY="your_wandb_api_key" # need more setup
export HUGGINGFACE_HUB_TOKEN="your_huggingface_api_key" # need more setup

llamafactory-cli train minicpm-v-v2_6.yaml
llamafactory-cli export minicpm-v-v2_6-lora_merge.yaml
