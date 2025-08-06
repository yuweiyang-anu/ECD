export WANDB_PROJECT="VLM-SFT-on-ECD"
export WANDB_API_KEY="your_wandb_api_key" # need more setup

llamafactory-cli train llama-llava-next-8b.yaml
llamafactory-cli export llama-llava-next-8b_lora_merge.yaml
