# VSM Note on replace LLava by Janus

## VSMMeta

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

#### c. **Prompt Encoder**
From Segment Anything
- **Purpose:** Generates embeddings from input prompts ( points, boxes, masks).
- **Configuration:**
  - Embedding dimension: `256`
  - Image embedding size: `(48, 48)`
  - Input image size: `(768, 768)`
  - Mask input channels: `16`
- Keeps all parameters trainable.

#### d. **Mask Decoder**
From Segment Anything
- **Purpose:** Handles segmentation tasks.
- **Configuration:**
  - Uses a transformer-based architecture (`TwoWayTransformer`).
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
  - Input dimension: `config.hidden_size`
  - Output dimension: `config.out_dim`
  - Architecture: Linear → ReLU → Linear → Dropout
- **Segmentation Task:**
  - Input dimension: `config.hidden_size`
  - Output dimension: `256`
  - Architecture: Linear → ReLU → Linear → Dropout
- Keeps all parameters trainable.

---

### Trainable Parameters
The `VSMMetaModel` selectively enables or disables gradient updates for different components:
- **Trainable:** Visual projection layer, prompt encoder, mask decoder, and text feature projection layers.
- **Frozen:** Vision model and box head of the `OwlViT` model.

---

### Purpose
The `VSMMetaModel` acts as a base class for higher-level models, providing:
- Object detection capabilities via `OwlViT`.
- Segmentation capabilities via the prompt encoder and mask decoder.
- Text feature projection for multimodal tasks.

---

## VSM Model

The `VSMModel` class was originally designed to work with LLaVA's `LlavaLlamaModel`, which uses a flat configuration structure. To make it compatible with JanusPro's `MultiModalityPreTrainedModel`, which employs a nested `MultiModalityConfig`, several updates were applied to the `__init__` method. These changes ensure that configuration settings are properly assigned within nested sub-configs, such as `language_config` and `vision_config`.

Absolutely! Here's a detailed and developer-focused version of the `## VSMForCausalLM` section for your dev document, designed to explain the logic, purpose, and internal workings of the class:

---

## 🧠 `VSMForCausalLM` Class

The `VSMForCausalLM` class is the **core multimodal model** that wraps and extends Janus's `MultiModalityCausalLM`. It integrates three key capabilities:
- **Language modeling** via Janus
- **Vision-language fusion** for VQA
- **Region-based visual grounding** for segmentation and detection using OWLViT + Segment Anything

### 🔧 Inheritance
```python
class VSMForCausalLM(MultiModalityCausalLM):
```
This class extends Janus’s base generative model while embedding custom vision modules. It coordinates the **language decoder**, **visual encoder**, and **task-specific heads**.

---

### 🔩 Constructor (`__init__`)

```python
def __init__(self, config: MultiModalityConfig, **kwargs):
```

- Initializes and configures Janus's components based on the passed config.
- Extracts loss weights and optional overrides (e.g., `vision_tower`, `loc_token_idx`).
- Builds a language model head: `self.lm_head = nn.Linear(...)`
- Initializes the `VSMModel`, which holds the OWLViT and segmentation modules.

---

### ⚙️ Core Methods

#### 1. `forward(...)`
Acts as the unified entry point. Delegates to:
- `super().forward(...)` when `past_key_values` are present (for fast decoding).
- `self.model_forward(...)` for custom training or inference flow.

---

#### 2. `model_forward(...)`
The main handler for **both training and inference**.

**Inputs**:
- `images`, `input_ids`, `labels`, etc.
- Visual and textual features
- Metadata like bounding boxes and segmentation masks

**Behavior**:
- Computes visual embeddings using OWLViT.
- Builds the location token mask (`<loc>` tokens for spatial grounding).
- Routes to `_inference_forward` or `_training_forward` depending on the mode.

---

#### 3. `_training_forward(...)`
Performs the full multimodal computation during training:

1. Gets LM output and hidden states from Janus
2. Extracts token-aligned hidden states
3. Projects hidden states for:
   - Segmentation (`text_hidden_fcs_seg`)
   - Detection (`text_hidden_fcs_det`)
4. Calls:
   - `_generate_masks()` → segmentation prediction
   - `_generate_detections()` → bounding box prediction
5. Computes total loss via `_compute_loss(...)`

---

#### 4. `_inference_forward(...)`
Used for **VQA / inference** without ground truth labels.
- Expands images to match beam search
- Retrieves hidden states
- Runs segmentation and detection heads
- Returns predictions: `pred_masks`, `pred_logits`, `pred_boxes`

---

#### 5. `_process_outputs(...)`
Modular logic to:
- Transform LM hidden states → visual token embeddings
- Generate masks and detections
- Used by both training and inference paths

---

#### 6. `_compute_loss(...)`
Computes:
- CE loss from Janus
- Mask loss: BCE + Dice
- Detection loss: OWLViT criterion
All combined with their respective weights.

---

### 📤 Output Format
Returns a dictionary with:
- `loss`: Total combined loss
- `ce_loss`: Language modeling loss
- `mask_loss`: Segmentation loss
- `detection_loss`: Bounding box loss

In inference, it returns predicted:
- Masks
- Logits
- Bounding boxes

---

### 🧩 Modular Breakdown

| Function                        | Purpose                                  |
|--------------------------------|------------------------------------------|
| `_create_loc_token_mask`       | Builds mask to find `<loc>` tokens       |
| `_process_hidden_states`       | Applies MLP heads for task projection    |
| `_extract_embeddings`          | Retrieves token-aligned task embeddings  |
| `_generate_masks`              | Runs SAM-style segmentation decoder      |
| `_generate_detections`         | Runs OWLViT for open-vocabulary detection|
| `_compute_mask_loss`           | Combines BCE and Dice losses             |
| `_compute_detection_loss`      | Uses OWLViT’s detection loss pipeline    |

---

### ✅ Design Goals

- Separate VQA and structured outputs (masks, boxes)
- Modular, reusable task heads
- Compatible with Janus multimodal config
- Support both autoregressive generation and region-level alignment



## Note on Changes to VSMModel for JanusPro Compatibility

### Language Model Settings
- **Original**: 
  ```python
  self.config.use_cache = False
  ```
- **Modified**: 
  ```python
  config.language_config.use_cache = False
  ```
- **Reason**: In JanusPro, language model settings are grouped under `language_config` rather than the top-level `config`.

### Vision Settings
- **Original**:
  ```python
  self.config.mm_vision_select_feature = "patch"
  ```
- **Modified**:
  ```python
  config.vision_config.select_feature = "patch"  # if applicable
  ```
- **Reason**: Vision-related settings are now nested under `vision_config`. A `hasattr` check is used to ensure compatibility if the attribute isn’t available.

### Omitted Settings
Settings such as `image_aspect_ratio`, `tune_mm_mlp_adapter`, and `freeze_mm_mlp_adapter` were removed from the config.

- **Reason**: These are handled elsewhere in JanusPro:
  - Image preprocessing (e.g., `image_aspect_ratio`) is managed by `VLChatProcessor`.
  - Module freezing or tuning (e.g., adapters) occurs post-initialization, not via the config.

## Additional Guidance

- To specify a vision tower, set it explicitly:
  ```python
  config.vision_config.model_name = "siglip_large_patch16_384"
  ```
  
- Freezing or tuning specific components (e.g., the aligner) should be done after model initialization, outside the config.

## Acknowledge
This note is edited my ChatGPT helper