# Development Notes on JanusV

## Status Tracking

### JanusPro-7B
- [✅] Clone and test the model
- [] Add `<loc>` token
- [] Prepare fine-tuning datasets
- [] Fine-tune with LoRA/QLoRA

---

## Conda Environments

### Dev1
- All three repositories' requirements installed

### Janus - Vstart - VSA
- Only named repository settings installed

---

## Bugs

### JanusPro-7B Issue
- **Issue**: Inference of JanusPro-7B in the `dev1` environment leads to the following error:
  
  ```bash
  TypeError: Object of type AlignerConfig is not JSON serializable
  ```

- **Debugging**:
  - Download JanusPro-7B offline
  - Use the Conda Janus environment to perform inference

---

### FlashAtt Installation Error
- **Issue**: An error occurred during the installation of `flash-att`.
  
- **Debugging**:
  - Manually install by downloading from the official repository
