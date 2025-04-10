# Fine-tune JanusPro with DeepSpeed and LoRA

## Environment Setup
To set up the environment, run the following commands:

```bash
cd VisualSearch/finetune

conda create -n janus-pro-lora python=3.10 -y
conda init
source ~/.bashrc
conda activate janus-pro-lora

pip install -r requirements.txt
```

## Download the Model
Download the Janus-Pro-7B model using the `modelscope` CLI:

```bash
modelscope download --model deepseek-ai/Janus-Pro-7B --local_dir ./Janus-Pro-7B
```