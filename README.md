# ComfyUI SAM3 Integration

ComfyUI custom nodes for Meta's **SAM 3 (Segment Anything with Concepts)**.

SAM 3 enables **open-vocabulary text-based segmentation** - segment objects using natural language prompts like "person", "red car", "shoe" without manual clicking or drawing.

---

## Features

- 🎯 **Text-based segmentation**: Use natural language to describe what to segment
- 🔀 **Multi-class support**: Comma-separated prompts like `"person,car,dog"`
- 🎭 **Instance merging**: All detected instances automatically merged
- 🖼️ **Dual outputs**: Mask + masked image
- ⚡ **GPU acceleration**: bf16/fp16/fp32 precision support
- 📦 **Auto-download**: Model downloads from Hugging Face automatically

---

## Installation

### Via ComfyUI Manager (Recommended)

Search for `ComfyUI-segment-anything-3` in ComfyUI Manager and install.

### Manual Installation

```bash
cd ComfyUI/custom_nodes/
git clone --recursive https://github.com/wzyfromhust/ComfyUI-segment-anything-3.git
cd ComfyUI-segment-anything-3
```

**Note**: Use `--recursive` to include SAM2 and SAM3 submodules.

---

## Nodes

### 1. DownloadAndLoadSAM3Model

Loads the SAM3 model.

**Parameters**:
- `device`: cuda/cpu
- `precision`: bf16 (default), fp16, fp32
- `confidence_threshold`: 0.0-1.0 (default: 0.5)

### 2. Sam3Segmentation

Segments objects using text prompts.

**Parameters**:
- `sam3_model`: Model from loader
- `image`: Input image
- `text_prompt`:
  - Single: `"person"`
  - Multiple: `"person,shoe,car"`
  - Descriptive: `"red car"`, `"person wearing blue"`
- `keep_model_loaded`: Keep in VRAM (default: false)

**Outputs**:
- `mask`: Segmentation mask [1, H, W]
- `image`: Masked image [1, H, W, C]

---

## Quick Start

```
LoadImage → Sam3Segmentation → PreviewImage
              ↑
    (Down)Load SAM3 Model
```

**Text prompt examples**:
- `"person"` - All people
- `"person,shoe"` - People and shoes (merged)
- `"red car"` - Red cars only
- `"person wearing blue"` - People in blue clothes

---

## Model Setup

**Model path**: `/root/ComfyUI/models/sam3/sam3.pt`

**Auto-download**: If model doesn't exist, it downloads from Hugging Face automatically.

**Requirements**:
```bash
huggingface-cli login
# Accept terms at https://huggingface.co/facebook/sam3
```

---

## Documentation

- 📘 [English Documentation](README_SAM3.md)
- 📗 [中文文档](README_SAM3_CN.md)

---

## Examples

### Single Class
```python
text_prompt = "person"  # Segments all people
```

### Multiple Classes (Auto-merged)
```python
text_prompt = "person,car,dog"  # Segments people, cars, and dogs
```

### Descriptive
```python
text_prompt = "red car"  # Only red cars
text_prompt = "person wearing blue"  # People in blue
```

---

## Technical Details

- Built on SAM3 official implementation
- Strictly aligned with official inference pipeline
- Uses `torch.autocast` with bf16 for optimal performance
- Supports multi-prompt inference with automatic merging

---

## Requirements

- ComfyUI
- PyTorch with CUDA (for GPU acceleration)
- Hugging Face account (for model download)

---

## Credits

- [SAM3 Official](https://github.com/facebookresearch/sam3) - Meta AI Research
- [ComfyUI SAM2](https://github.com/kijai/ComfyUI-segment-anything-2) - SAM2 integration reference

---

## License

This project integrates:
- SAM3: Apache 2.0 License
- SAM2: Apache 2.0 License

---

## Links

- 📄 [SAM3 Paper](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)
- 🤗 [SAM3 Model](https://huggingface.co/facebook/sam3)
- 💻 [GitHub Repository](https://github.com/wzyfromhust/ComfyUI-segment-anything-3)
