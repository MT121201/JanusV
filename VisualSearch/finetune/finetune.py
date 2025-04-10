import os
import torch
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from transformers import Trainer, TrainingArguments, get_scheduler
from multimodal_trainer import EnhancedMultiModalTrainer
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--pretrained_model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--user_question', type=str, default="Please describe the content of this image.")
    parser.add_argument('--optimizer_name', type=str, default="AdamW")
    parser.add_argument('--lora_config', type=str, required=True)
    parser.add_argument('--training_args', type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    # Load LoRA configuration
    lora_config = eval(args.lora_config)

    # Load DeepSpeed configuration file
    deepspeed_config_path = args.deepspeed_config

    # Start the fine-tuning process with DeepSpeed
    trainer = EnhancedMultiModalTrainer(
        data_dir=args.data_dir,
        pretrained_model_path=args.pretrained_model_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr=args.lr,
        user_question=args.user_question,
        optimizer_name=args.optimizer_name,
        lora_config=lora_config,
        training_args=eval(args.training_args)
    )

    trainer.train()

if __name__ == "__main__":
    main()
