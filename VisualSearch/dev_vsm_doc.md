
# VSM Note on Replacing LLaVA with Janus

This document explains how the VSM model has been refactored to replace the LLaVA base model with DeepSeek's **Janus** model. It includes detailed notes on architecture, refactoring changes, and component-specific behavior.

---

## VSMMetaModel

### 1. **Initialization**
The `__init__` method:
- Accepts a `config` object and additional keyword arguments (`kwargs`).

### 2. **Module Initialization**
The `initialize_lisa_modules` method sets up the following components:

#### a. **OWL-ViT (Object Detection Backbone)**
- **Purpose:** Handles object detection tasks.
- **Configuration:**
  - Initializes the `OwlViT` model with `n_classes=1` and evaluation mode (`is_eval`).
  - Freezes specific parts of the model:
    - Vision model
    - Box head
  - Keeps other parts trainable.

#### b. **Visual Projection Layer**
- Adds a learnable linear layer to project visual embeddings into a 256-dimensional space.
- Keeps this layer trainable.

#### c. **Prompt Encoder** (from Segment Anything)
- **Purpose:** Generates embeddings from input prompts (points, boxes, masks).
- **Configuration:**
  - Embedding dimension: `256`
  - Image embedding size: `(48, 48)`
  - Input image size: `(768, 768)`
  - Mask input channels: `16`
- Keeps all parameters trainable.

#### d. **Mask Decoder** (from Segment Anything)
- **Purpose:** Handles segmentation tasks.
- **Configuration:**
  - Transformer depth: `2`
  - Embedding dimension: `256`
  - MLP dimension: `2048`
  - Number of heads: `8`
  - IOU head depth: `3`
  - IOU head hidden dimension: `256`
- Keeps all parameters trainable.

#### e. **Text Feature Projection Layers**
- **Purpose:** Projects text features for detection and segmentation tasks.
- **Detection Task:**
  - Input: `config.hidden_size`
  - Output: `config.out_dim`
- **Segmentation Task:**
  - Output: `256`
- Architecture: `Linear → ReLU → Linear → Dropout`

---

### Trainable Parameters
The `VSMMetaModel` selectively enables or disables gradient updates for different components:
- **Trainable:** Visual projection layer, prompt encoder, mask decoder, and text feature projection layers.
- **Frozen:** Vision model and box head of the `OwlViT` model.

### Purpose
The `VSMMetaModel` acts as a base class for higher-level models, providing:
- Object detection via `OwlViT`
- Segmentation via Segment Anything
- Text feature projection for multimodal tasks

---

## VSMModel

The `VSMModel` class adapts the meta model into a Janus-compatible `MultiModalityPreTrainedModel`. It ensures correct application of:
- `language_config.use_cache = False`
- `vision_config.select_feature = "patch"`

This structure replaces flat configs used in LLaVA.

---

## 🧠 VSMForCausalLM Class

The `VSMForCausalLM` class is the **core multimodal model** that wraps and extends Janus's `MultiModalityCausalLM`. It integrates:
- **Language modeling** via Janus
- **Vision-language fusion** for VQA
- **Region-based visual grounding** using OWLViT + Segment Anything

### 🔧 Inheritance
```python
class VSMForCausalLM(MultiModalityCausalLM):
```

---

### 🔩 Constructor (`__init__`)
```python
def __init__(self, config: MultiModalityConfig, **kwargs):
```
- Loads loss weights and vision tower info from kwargs
- Initializes `VSMModel` as the visual backbone
- Builds `lm_head` and sets Janus config attributes
- Finalizes weights via `self.post_init()`

---

### ⚙️ Core Methods

#### 1. `forward(...)`
Delegates based on presence of cached keys (for fast decoding). Otherwise, calls `model_forward`.

#### 2. `model_forward(...)`
Main logic for both training and inference:
- Computes visual embeddings
- Builds `<loc>` token mask
- Dispatches to `_training_forward` or `_inference_forward`

#### 3. `_training_forward(...)`
- Computes LM outputs
- Projects embeddings for detection/segmentation
- Runs segmentation (`_generate_masks`)
- Runs detection (`_generate_detections`)
- Combines all losses via `_compute_loss`

#### 4. `_inference_forward(...)`
- Computes output embeddings
- Produces predictions only (no ground truth)
- Returns `pred_masks`, `pred_logits`, `pred_boxes`

#### 5. `_process_outputs(...)`
- Applies MLPs on LM hidden states
- Extracts region-level embeddings
- Generates masks and bounding boxes

#### 6. `_compute_loss(...)`
- Combines CE loss, BCE+Dice loss for segmentation, OWLViT detection loss

---

### 🧩 Helper Functions

| Function | Purpose |
|----------|---------|
| `_create_loc_token_mask` | Builds token alignment mask |
| `_process_hidden_states` | Applies task-specific heads |
| `_extract_embeddings` | Aligns token output |
| `_generate_masks` | Produces segmentation output |
| `_generate_detections` | Produces detection output |
| `_compute_mask_loss` | BCE + Dice loss |
| `_compute_detection_loss` | OWLViT loss |

---

### 📤 Output Format

| Mode | Output |
|------|--------|
| Training | Dictionary of total, CE, mask, and detection losses |
| Inference | Predicted `masks`, `boxes`, `logits` |

---

## Note on Changes to VSMModel for JanusPro Compatibility

### Language Model Settings
- Old: `self.config.use_cache = False`
- New: `config.language_config.use_cache = False`

### Vision Settings
- Old: `self.config.mm_vision_select_feature = "patch"`
- New: `config.vision_config.select_feature = "patch"`

---

### Omitted Settings
Settings like `image_aspect_ratio`, `tune_mm_mlp_adapter`, and `freeze_mm_mlp_adapter` were removed:
- Image processing is handled by `VLChatProcessor`
- Component freezing is done after model load

---

## Additional Guidance

- To specify a vision tower:
```python
config.vision_config.model_name = "siglip_large_patch16_384"
```
- Use `VLChatProcessor.from_pretrained()` during training to preprocess image/text inputs.

---

## Acknowledgement
This documentation was edited with assistance from ChatGPT.
