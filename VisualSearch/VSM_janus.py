from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from VisualSearch.janus.models.modeling_vlm import (
	MultiModalityPreTrainedModel,
	MultiModalityCausalLM,
	MultiModalityConfig,
)

from .segment_anything.modeling import PromptEncoder, MaskDecoder, TwoWayTransformer
from .owlvit.owlvit import OwlViT


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float, scale=1000, eps=1e-6):
	"""
	Compute the DICE loss, similar to generalized IOU for masks.
	Args:
		inputs: A float tensor of arbitrary shape. The predictions for each example.
		targets: A float tensor with the same shape as inputs. Stores the binary
				 classification label for each element in inputs.
		num_masks: Number of masks in the batch.
		scale: Scaling factor for inputs and targets.
		eps: Small value to avoid division by zero.
	"""
	inputs = inputs.sigmoid().flatten(1, 2)
	targets = targets.flatten(1, 2)
	numerator = 2 * (inputs / scale * targets).sum(-1)
	denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
	loss = 1 - (numerator + eps) / (denominator + eps)
	return loss / (num_masks + 1e-8)


def sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor, num_masks: float):
	"""
	Compute the sigmoid cross-entropy loss.
	Args:
		inputs: A float tensor of arbitrary shape. The predictions for each example.
		targets: A float tensor with the same shape as inputs. Stores the binary
				 classification label for each element in inputs.
		num_masks: Number of masks in the batch.
	"""
	loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
	return loss.flatten(1, 2).mean(1) / (num_masks + 1e-8)


class VSMMetaModel:
	def __init__(self, config, **kwargs):
		super(VSMMetaModel, self).__init__(config)
		self.config = config

		if not hasattr(self.config, "train_mask_decoder"):
			self.config.train_mask_decoder = kwargs["train_mask_decoder"]
			self.config.out_dim = kwargs["out_dim"]
		else:
			is_eval = kwargs.get("is_eval", False)
			self.initialize_lisa_modules(self.config, is_eval)

	def initialize_lisa_modules(self, config, is_eval=False):
		# Initialize OWL-ViT
		self.owlvit = OwlViT(1, is_eval)
		self.owlvit.train()
		self._set_requires_grad(self.owlvit.parameters(), True)
		self._set_requires_grad(self.owlvit.vision_model.parameters(), False)
		self.owlvit.vision_model.eval()
		self._set_requires_grad(self.owlvit.box_head.parameters(), False)

		# Visual projection layer
		self.visual_projection = nn.Linear(self.owlvit.vision_model.config.hidden_size, 256, bias=False)
		self._set_requires_grad(self.visual_projection.parameters(), True)

		# Prompt Encoder and Mask Decoder
		self.prompt_encoder = PromptEncoder(
			embed_dim=256,
			image_embedding_size=(48, 48),
			input_image_size=(768, 768),
			mask_in_chans=16,
		)
		self.prompt_encoder.train()
		self._set_requires_grad(self.prompt_encoder.parameters(), True)

		self.mask_decoder = MaskDecoder(
			num_multimask_outputs=3,
			transformer=TwoWayTransformer(
				depth=2,
				embedding_dim=256,
				mlp_dim=2048,
				num_heads=8,
			),
			transformer_dim=256,
			iou_head_depth=3,
			iou_head_hidden_dim=256,
		)
		self.mask_decoder.train()
		self._set_requires_grad(self.mask_decoder.parameters(), True)

		# Text projection layers for detection and segmentation
		self.text_hidden_fcs_det = self._create_text_projection_layer(config.hidden_size, config.out_dim)
		self.text_hidden_fcs_seg = self._create_text_projection_layer(config.hidden_size, 256)

	def _set_requires_grad(self, parameters, requires_grad):
		for param in parameters:
			param.requires_grad = requires_grad

	def _create_text_projection_layer(self, in_dim, out_dim):
		layers = [
			nn.Linear(in_dim, in_dim),
			nn.ReLU(inplace=True),
			nn.Linear(in_dim, out_dim),
			nn.Dropout(0.0),
		]
		module = nn.ModuleList([nn.Sequential(*layers)])
		module.train()
		self._set_requires_grad(module.parameters(), True)
		return module




class VSMModel(VSMMetaModel, MultiModalityPreTrainedModel):
	def __init__(self, config, **kwargs):
		super().__init__(config, **kwargs)
		self._configure_language_model(config)
		self._configure_vision_model(config)

	def _configure_language_model(self, config):
		if hasattr(config, 'language_config'):
			config.language_config.use_cache = False

	def _configure_vision_model(self, config):
		if hasattr(config, 'vision_config') and hasattr(config.vision_config, 'select_feature'):
			config.vision_config.select_feature = "patch"


class VSMForCausalLM(MultiModalityCausalLM):
	def __init__(self, config: MultiModalityConfig, **kwargs):
		# Handle vision tower and other settings based on config structure
		self._initialize_config(config, **kwargs)

		# Location token index
		self.loc_token_idx = kwargs.pop("loc_token_idx")

		# Initialize the parent class (MultiModalityCausalLM)
		super().__init__(config)

		# Initialize the VSMModel with the config
		self.model = VSMModel(config, **kwargs)

		# Language model head
		hidden_size = getattr(config.language_config, 'hidden_size', config.hidden_size)
		self.lm_head = nn.Linear(hidden_size, config.vocab_size, bias=False)

		# Initialize weights and apply final processing
		self.post_init()

	def _initialize_config(self, config, **kwargs):
		if not hasattr(config, "train_mask_decoder"):
			if hasattr(config, 'vision_config'):
				config.vision_config.model_name = kwargs.get(
					"vision_tower", "siglip_large_patch16_384"
				)
			self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
			self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
			self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
			self.det_loss_weight = kwargs.pop("det_loss_weight", None)
		else:
			if hasattr(config, 'vision_config'):
				config.vision_config.model_name = config.vision_tower

	def get_visual_embs(self, pixel_values: torch.FloatTensor):
		with torch.no_grad():
			return self.model.owlvit.get_visual_embs(pixel_values)

	def forward(self, **kwargs):
		if "past_key_values" in kwargs:
			return super().forward(**kwargs)
		return self.model_forward(**kwargs)

	def model_forward(
		self,
		images: torch.FloatTensor,
		input_ids: torch.LongTensor,
		labels: torch.LongTensor,
		attention_masks: torch.LongTensor,
		offset: torch.LongTensor,
		masks_list: List[torch.FloatTensor],
		label_list: List[torch.Tensor],
		bboxes_labels_list: List[torch.FloatTensor],
		bboxes_valid_list: torch.Tensor,
		masks_valid_list: List[torch.Tensor],
		resize_list: List[tuple],
		inference: bool = False,
		**kwargs,
	):
		image_embeddings = self.get_visual_embs(images)
		batch_size = image_embeddings.shape[0]
		assert batch_size == len(offset) - 1

		loc_token_mask = self._create_loc_token_mask(input_ids)

		if inference:
			return self._inference_forward(
				images, attention_masks, input_ids, loc_token_mask, label_list, resize_list
			)

		return self._training_forward(
			images, attention_masks, input_ids, labels, loc_token_mask, masks_list,
			label_list, bboxes_labels_list, bboxes_valid_list, masks_valid_list, resize_list
		)

	def _create_loc_token_mask(self, input_ids):
		loc_token_mask = input_ids[:, 1:] == self.loc_token_idx
		loc_token_mask = torch.cat(
			[loc_token_mask, torch.zeros((loc_token_mask.shape[0], 1)).bool().cuda()],
			dim=1,
		)
		return torch.cat(
			[torch.zeros((loc_token_mask.shape[0], 255)).bool().cuda(), loc_token_mask],
			dim=1,
		)

	def _inference_forward(self, images, attention_masks, input_ids, loc_token_mask, label_list, resize_list):
		output = super().forward(
			images=images.expand(input_ids.shape[0], -1, -1, -1).contiguous(),
			attention_mask=attention_masks,
			input_ids=input_ids,
			output_hidden_states=True,
		)
		output_hidden_states = [output.hidden_states]
		pred_masks, pred_logits, pred_boxes = self._process_outputs(
			output_hidden_states, loc_token_mask, label_list, resize_list
		)
		return {
			"pred_masks": pred_masks,
			"pred_logits": pred_logits,
			"pred_boxes": pred_boxes,
		}

	def _training_forward(
		self, images, attention_masks, input_ids, labels, loc_token_mask, masks_list,
		label_list, bboxes_labels_list, bboxes_valid_list, masks_valid_list, resize_list
	):
		output = super().forward(
			images=images,
			attention_mask=attention_masks,
			input_ids=input_ids,
			labels=labels,
			output_hidden_states=True,
		)
		output_hidden_states = output.hidden_states

		pred_masks, pred_logits, pred_boxes = self._process_outputs(
			output_hidden_states, loc_token_mask, label_list, resize_list
		)

		loss = self._compute_loss(
			output, pred_masks, masks_list, pred_logits, pred_boxes,
			bboxes_labels_list, bboxes_valid_list, masks_valid_list
		)
		return loss

	def _process_outputs(self, output_hidden_states, loc_token_mask, label_list, resize_list):
		hidden_states_seg, hidden_states_det = self._process_hidden_states(output_hidden_states)
		pred_embeddings_seg, pred_embeddings_det = self._extract_embeddings(
			hidden_states_seg, hidden_states_det, loc_token_mask
		)
		pred_masks = self._generate_masks(pred_embeddings_seg, label_list, resize_list)
		pred_logits, pred_boxes = self._generate_detections(pred_embeddings_det)
		return pred_masks, pred_logits, pred_boxes

	def _process_hidden_states(self, output_hidden_states):
		hidden_states_seg = [self.model.text_hidden_fcs_seg[0](output_hidden_states[-1])]
		hidden_states_det = [self.model.text_hidden_fcs_det[0](output_hidden_states[-1])]
		return (
			torch.stack(hidden_states_seg, dim=-1).sum(dim=-1),
			torch.stack(hidden_states_det, dim=-1).sum(dim=-1),
		)

	def _extract_embeddings(self, hidden_states_seg, hidden_states_det, loc_token_mask):
		pred_embeddings_seg = hidden_states_seg[loc_token_mask]
		pred_embeddings_det = hidden_states_det[loc_token_mask]
		return pred_embeddings_seg, pred_embeddings_det

	def _generate_masks(self, pred_embeddings_seg, label_list, resize_list):
		pred_masks = []
		for i, pred_embedding in enumerate(pred_embeddings_seg):
			sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
				points=None, boxes=None, masks=None, text_embeds=pred_embedding.unsqueeze(1),
			)
			sparse_embeddings = sparse_embeddings.to(pred_embedding.dtype)
			low_res_masks, _ = self.model.mask_decoder(
				image_embeddings=self.model.visual_projection(self.get_visual_embs(label_list[i]).unsqueeze(0)).permute(0, 3, 1, 2),
				image_pe=self.model.prompt_encoder.get_dense_pe(),
				sparse_prompt_embeddings=sparse_embeddings,
				dense_prompt_embeddings=dense_embeddings,
				multimask_output=False,
			)
			pred_mask = F.interpolate(
				low_res_masks, label_list[i].shape, mode="bilinear", align_corners=False
			)
			pred_masks.append(pred_mask[:, 0])
		return pred_masks

	def _generate_detections(self, pred_embeddings_det):
		detection_result_batch = []
		for pred_embedding in pred_embeddings_det:
			bs = pred_embedding.shape[0]
			detection_result = self.model.owlvit(
				self.get_visual_embs(pred_embedding).unsqueeze(0).repeat(bs, 1, 1, 1),
				pred_embedding.unsqueeze(1)
			)
			detection_result_batch.append(detection_result)
		pred_logits = torch.cat([result['pred_logits'] for result in detection_result_batch], 0)
		pred_boxes = torch.cat([result['pred_boxes'] for result in detection_result_batch], 0)
		return pred_logits, pred_boxes

	def _compute_loss(
		self, output, pred_masks, masks_list, pred_logits, pred_boxes,
		bboxes_labels_list, bboxes_valid_list, masks_valid_list
	):
		ce_loss = output.loss * self.ce_loss_weight
		mask_loss = self._compute_mask_loss(pred_masks, masks_list, masks_valid_list)
		detection_loss = self._compute_detection_loss(pred_logits, pred_boxes, bboxes_labels_list, bboxes_valid_list)
		return {
			"loss": ce_loss + mask_loss + detection_loss,
			"ce_loss": ce_loss,
			"mask_loss": mask_loss,
			"detection_loss": detection_loss,
		}

	def _compute_mask_loss(self, pred_masks, masks_list, masks_valid_list):
		mask_bce_loss, mask_dice_loss, num_masks = 0, 0, 0
		for pred_mask, gt_mask, masks_valid in zip(pred_masks, masks_list, masks_valid_list):
			mask_bce_loss += (sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0] * masks_valid).sum()
			mask_dice_loss += (dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0]) * gt_mask.shape[0] * masks_valid).sum()
			num_masks += masks_valid.sum()
		mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
		mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
		return mask_bce_loss + mask_dice_loss

	def _compute_detection_loss(self, pred_logits, pred_boxes, bboxes_labels_list, bboxes_valid_list):
		num_boxes = sum(len(bboxes_label) for bboxes_label, bboxes_valid in zip(bboxes_labels_list, bboxes_valid_list) if bboxes_valid)
		num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=pred_logits.device)
		num_boxes = torch.clamp(num_boxes, min=1).item()

		detection_result_batch = {'pred_logits': pred_logits, 'pred_boxes': pred_boxes}
		target_det = [{"labels": torch.zeros(len(bboxes_label)).to(bboxes_label.device, torch.long), "boxes": bboxes_label}
					  for bboxes_label in bboxes_labels_list]
		loss_dict = self.model.owlvit.criterion(detection_result_batch, target_det, num_boxes)
		weight_dict = self.model.owlvit.criterion.weight_dict
		return sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict) * self.det_loss_weight
