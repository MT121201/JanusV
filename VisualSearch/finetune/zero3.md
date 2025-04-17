# 🧾 DeepSpeed ZeRO-3 Configuration (Markdown Version)
This configuration is designed for fine-tuning large-scale models using DeepSpeed's ZeRO Stage 3 optimization. It includes settings for mixed precision training, batch size management, and advanced memory optimization techniques

## ⚙️ Mixed Precision Settings

```json
"fp16": {
  "enabled": "auto",             // Automatically enable FP16 if supported by hardware
  "loss_scale": 0,               // Use dynamic loss scaling
  "loss_scale_window": 1000,     // Window size for adjusting loss scale
  "initial_scale_power": 16,     // Initial loss scale: 2^16
  "hysteresis": 2,               // Delay in decreasing loss scale after overflow
  "min_loss_scale": 1            // Minimum loss scale value
},
"bf16": {
  "enabled": "auto"              // Automatically enable BF16 if supported by hardware
}
```

## 📦 Batch Size and Gradient Accumulation

```json
"train_micro_batch_size_per_gpu": "auto",  // Automatically determine micro-batch size per GPU
"train_batch_size": "auto",                // Automatically determine total training batch size
"gradient_accumulation_steps": "auto"      // Automatically determine gradient accumulation steps
```

## 🧠 ZeRO Optimization Setting


```json
"zero_optimization": {
  "stage": 3,                               // Enable ZeRO Stage 3: partition optimizer states, gradients, and parameters
  "overlap_comm": true,                     // Overlap communication with computation for efficiency
  "contiguous_gradients": true,             // Allocate gradients contiguously in memory to reduce fragmentation
  "sub_group_size": 1e9,                    // Size of parameter subgroups for partitioning
  "reduce_bucket_size": "auto",             // Automatically determine bucket size for gradient reduction
  "stage3_prefetch_bucket_size": "auto",    // Automatically determine prefetch bucket size
  "stage3_param_persistence_threshold": "auto", // Automatically determine threshold for parameter persistence in memory
  "stage3_max_live_parameters": 1e9,        // Maximum number of live parameters in memory
  "stage3_max_reuse_distance": 1e9,         // Maximum reuse distance for parameters
  "stage3_gather_16bit_weights_on_model_save": true // Gather 16-bit weights from all partitions when saving the model
}
```
