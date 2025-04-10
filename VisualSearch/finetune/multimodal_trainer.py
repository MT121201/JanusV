# multimodal_trainer.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Tuple, Dict, Any
import torch
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW, SGD
from transformers import Trainer, TrainingArguments, get_scheduler
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images


class EnhancedMultiModalModel(MultiModalityCausalLM):
    """
    This class extends the `MultiModalityCausalLM` to provide a custom forward logic 
    for handling multimodal inputs, specifically joint processing of text and image data.

    Methods
    -------
    forward(input_ids=None, attention_mask=None, pixel_values=None, image_token_masks=None, labels=None, **kwargs)
        Custom forward method to process multimodal inputs and pass them through the language model.

    _process_multimodal_inputs(input_ids: torch.LongTensor, pixel_values: torch.FloatTensor, image_token_masks: torch.BoolTensor) -> torch.Tensor
        Processes multimodal inputs by aligning image features with text embeddings.

    Attributes
    ----------
    language_model : nn.Module
        The underlying language model used for text processing.
    vision_model : nn.Module
        The vision model used for extracting features from image inputs.
    aligner : nn.Module
        A module for aligning image features with text embeddings.
    """

    def forward(self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        image_token_masks=None,
        labels=None,
        **kwargs,
    ):
        """
        Perform a forward pass through the multimodal trainer model.
        Args:
            input_ids (torch.Tensor, optional): Input token IDs for the text data.
            attention_mask (torch.Tensor, optional): Attention mask for the text input.
            pixel_values (torch.Tensor, optional): Pixel values for the image data.
            image_token_masks (torch.Tensor, optional): Masks for the image tokens.
            labels (torch.Tensor, optional): Labels for the supervised training task.
            **kwargs: Additional keyword arguments for the language model.
        Returns:
            transformers.modeling_outputs.Seq2SeqLMOutput: The output of the language model, 
            which includes logits, loss (if labels are provided), and other model-specific outputs.
        """
        
        if pixel_values is not None and image_token_masks is not None:
            inputs_embeds = self._process_multimodal_inputs(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_token_masks=image_token_masks,
            )
        else:
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        kwargs.pop("inputs_embeds", None)
        outputs = self.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        return outputs

    def _process_multimodal_inputs(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_token_masks: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        Processes multimodal inputs by aligning image features with text embeddings.
        Args:
            input_ids (torch.LongTensor): Tensor containing token IDs for the text input, 
                with shape (batch_size, sequence_length).
            pixel_values (torch.FloatTensor): Tensor containing pixel values for the images, 
                with shape (batch_size, num_images, channels, height, width).
            image_token_masks (torch.BoolTensor): Boolean tensor indicating which tokens 
                correspond to image tokens, with shape (batch_size, sequence_length).
        Returns:
            torch.Tensor: A tensor containing the combined text embeddings and aligned 
            image features, with shape (batch_size, sequence_length, embedding_dim).
        Raises:
            ValueError: If the number of image tokens indicated by `image_token_masks` 
                does not match the number of aligned image features for any sample in the batch.
        Notes:
            - The method extracts image features using a vision model, aligns them using 
              an alignment module, and integrates them into the text embeddings.
            - The alignment ensures that image features are properly flattened and reshaped 
              to match the expected token structure.
        """
        batch_size, num_images = pixel_values.shape[:2]
        images = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])
        image_features = self.vision_model(images)
        aligned_features = self.aligner(image_features)

        aligned_features = aligned_features.view(batch_size, num_images, *aligned_features.shape[1:])
        aligned_features = aligned_features.flatten(1, 2)

        text_embeds = self.language_model.get_input_embeddings()(input_ids)

        for i in range(batch_size):
            num_image_tokens = image_token_masks[i].sum().item()
            num_aligned_tokens = aligned_features[i].shape[0]
            if num_image_tokens != num_aligned_tokens:
                raise ValueError(
                    f"Mismatch at sample {i}: "
                    f"image_token_masks has {num_image_tokens} tokens, "
                    f"but aligned_features has {num_aligned_tokens} tokens!"
                )
            text_embeds[i][image_token_masks[i]] = aligned_features[i]

        return text_embeds


class EnhancedMultiModalTrainer:
    def __init__(self, 
                 data_dir: str, 
                 pretrained_model_path: str, 
                 output_dir: str, 
                 batch_size: int = 1, 
                 max_epochs: int = 10, 
                 lr: float = 3e-4, 
                 user_question: str = "What does this photo depict?",
                 optimizer_name: str = "AdamW",
                 lora_config: dict = None,
                 training_args: dict = None):
        """
        Initialize the enhanced multimodal trainer.
        """
        self.data_dir = data_dir
        self.pretrained_model_path = pretrained_model_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.user_question = user_question
        self.optimizer_name = optimizer_name
        self.lora_config = lora_config if lora_config else {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
        self.training_args = training_args if training_args else {
            "fp16": True,
            "max_grad_norm": 1.0,
            "save_strategy": "epoch",
            "evaluation_strategy": "no",
            "logging_steps": 50,
            "save_total_limit": 2,
            "remove_unused_columns": False,
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _load_data(self) -> List[Tuple[str, str]]:
        """Load image-text paired data."""
        pairs = []
        img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        for root, _, files in os.walk(self.data_dir):
            for fname in files:
                base, ext = os.path.splitext(fname)
                full_path = os.path.join(root, fname)
                if ext.lower() in img_exts:
                    txt_path = os.path.join(root, f"{base}.txt")
                    if os.path.exists(txt_path):
                        with open(txt_path, "r", encoding="utf-8") as f:
                            text_content = f.read().strip()
                        pairs.append((full_path, text_content))
        if not pairs:
            raise ValueError(f"No matching image-text pairs found in {self.data_dir}.")
        print(f"Loaded {len(pairs)} image-text pairs from {self.data_dir}")
        return pairs

    def _prepare_model(self):
        """Load the pretrained model and configure LoRA fine-tuning."""
        print(f"Loading pretrained model from {self.pretrained_model_path}")
        self.processor = VLChatProcessor.from_pretrained(
            self.pretrained_model_path,
            slow_image_processor_class="AutoImageProcessor"
        )
        self.model = EnhancedMultiModalModel.from_pretrained(
            self.pretrained_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Configure LoRA
        lora_config = LoraConfig(**self.lora_config)
        self.model = get_peft_model(self.model, lora_config)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable params: {trainable_params} / {total_params}")

    def _prepare_optimizer_and_scheduler(self, dataset_size: int):
        """
        Prepare the optimizer and learning rate scheduler for training.

        Args:
            dataset_size (int): The size of the dataset used for training.

        Raises:
            ValueError: If the specified optimizer name is not supported.
            ValueError: If the specified learning rate scheduler type is not supported.

        Attributes Set:
            self.optimizer: The optimizer instance configured based on the specified optimizer name.
            self.lr_scheduler: The learning rate scheduler instance configured based on the specified scheduler type.

        Notes:
            - Supported optimizers: "AdamW", "SGD".
            - Supported learning rate schedulers: "linear", "cosine", "cosine_with_restarts", 
              "polynomial", "constant", "constant_with_warmup".
            - The number of warmup steps is set to 10% of the total training steps.
        """
        if self.optimizer_name == "AdamW":
            optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        elif self.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        num_training_steps = dataset_size * self.max_epochs // self.batch_size
        num_warmup_steps = int(0.1 * num_training_steps)

        scheduler_name = self.training_args.get("lr_scheduler_type", "cosine")
        if scheduler_name not in ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]:
            raise ValueError(f"Unsupported learning rate scheduler: {scheduler_name}")

        lr_scheduler = get_scheduler(
            name=scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Custom data collation function."""
        conversations = []
        for item in batch:
            conversations.extend(self._generate_conversation(item["image_path"], item["text"]))
        
        pil_images_list = load_pil_images(conversations)
        encoded = self.processor(
            conversations=conversations,
            images=pil_images_list,
            return_tensors="pt",
            force_batchify=True,
        )
        encoded["labels"] = encoded["input_ids"].clone()

        image_placeholder_token_id = self.processor.tokenizer.convert_tokens_to_ids("<image_placeholder>")
        input_ids = encoded["input_ids"]
        image_token_masks = (input_ids == image_placeholder_token_id)

        if not image_token_masks.any():
            raise ValueError("No <image_placeholder> tokens found in the input!")

        encoded["image_token_masks"] = image_token_masks
        return dict(encoded)

    def _generate_conversation(self, image_path: str, assistant_text: str) -> List[Dict[str, Any]]:
        """Generate conversation templates."""
        return [
            {"role": "<|User|>", "content": f"<image_placeholder>\n{self.user_question}", "images": [image_path]},
            {"role": "<|Assistant|>", "content": assistant_text},
        ]

    def train(self):
        """Main training process."""
        pairs = self._load_data()
        dataset = [{"image_path": img, "text": txt} for img, txt in pairs]

        self._prepare_model()
        self._prepare_optimizer_and_scheduler(len(dataset))

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.max_epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.lr,
            **self.training_args,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=self._collate_fn,
            optimizers=(self.optimizer, self.lr_scheduler),
        )

        print("[Start!] Training started!")
        trainer.train()

        print("[Progress] Merging LoRA weights...")
        self.model = self.model.merge_and_unload()

        print("[Progress] Saving...")
        self.model.save_pretrained(self.output_dir)
        self.processor.save_pretrained(self.output_dir)
        print(f"[Done!] Fine-tuned model saved to {self.output_dir}")