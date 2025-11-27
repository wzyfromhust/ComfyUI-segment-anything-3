# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ComfyUI custom nodes for **SAM 3 (Segment Anything with Concepts)** from Meta AI. SAM 3 enables open-vocabulary text-based segmentation - segment objects using natural language prompts like "person", "red car", "shoe" without manual clicking.

The repository includes:
- `sam3_nodes/`: ComfyUI node implementations for SAM 3
- `sam3/`: Full SAM 3 research codebase (git submodule from facebookresearch/sam3)
- `ComfyUI-segment-anything-2/`: SAM 2 integration (git submodule, requires `git submodule update --init`)

## Commands

### Testing

```bash
# Test node imports (no model required)
python test_nodes_import.py

# Test SAM3 standalone functionality
python test_sam3_standalone.py
```

### SAM 3 Development (in sam3/ directory)

```bash
cd sam3
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
ufmt format .
```

### Initialize Submodules

```bash
git submodule update --init --recursive
```

## Architecture

### ComfyUI Nodes (sam3_nodes/)

**Entry Point**: `__init__.py` - Merges SAM2 and SAM3 node mappings for ComfyUI

**Nodes** (`sam3_nodes/nodes.py`):
- `DownloadAndLoadSAM3Model`: Loads SAM 3 from HuggingFace (auto-downloads if missing)
- `Sam3SegmentationByIndex`: Main node with mask_index for left-to-right instance selection
- `Sam3SegmentationByFace`: Selects mask with highest intersection to CROP_DATA bbox
- `Sam3SegmentationOriginal`: Simple version returning merged mask and black-background cutout

**Model Loading** (`sam3_nodes/load_model.py`):
- Thin wrapper around official `sam3.build_sam3_image_model()`
- Creates `Sam3Processor` for inference
- Manages autocast context for bf16/fp16 precision

### SAM 3 Model (sam3/)

**Model Builder** (`sam3/sam3/model_builder.py`):
- `build_sam3_image_model()`: Image segmentation with text/visual prompts
- `build_sam3_video_model()`: Video tracking with text prompts
- `build_tracker()`: Tracker module for temporal consistency

**Key Components**:
- `Sam3Image`: Core detector with ViT backbone, text encoder, transformer, segmentation head
- `Sam3Processor`: Handles image preprocessing and inference state
- `VETextEncoder`: BPE tokenizer + transformer for text prompts
- `SequenceGeometryEncoder`: Encodes points/boxes as sequences

### Data Flow

1. `DownloadAndLoadSAM3Model` loads model to GPU with precision settings
2. `Sam3Segmentation*` receives IMAGE tensor [B,H,W,C] float32 [0,1]
3. Converts to PIL, calls `processor.set_image()` then `processor.set_text_prompt()`
4. Inference state contains `masks` [N,1,H,W] bool and `scores` [N,] bfloat16
5. Merges/selects masks, returns ComfyUI MASK [1,H,W] float32 and visualization IMAGE

## Model Files

**SAM 3 Model**:
- Path: `/root/ComfyUI/models/sam3/sam3.pt`
- Auto-downloads from `facebook/sam3` on HuggingFace
- Requires: `huggingface-cli login` + accept terms at https://huggingface.co/facebook/sam3

**BPE Tokenizer**:
- Path: `sam3/assets/bpe_simple_vocab_16e6.txt.gz`

## Custom Node Development

When adding new nodes to `sam3_nodes/nodes.py`:
1. Define class with `INPUT_TYPES()`, `RETURN_TYPES`, `RETURN_NAMES`, `FUNCTION`, `CATEGORY`
2. Use `comfy.model_management` for device/memory management
3. Register in `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` dicts

Key patterns from existing nodes:
- Image format: `[B,H,W,C]` float32 [0,1] -> convert to PIL for SAM3
- Mask format: `[1,H,W]` float32 for ComfyUI compatibility
- Memory cleanup: Call `processor.reset_all_prompts()` and `mm.soft_empty_cache()` when `keep_model_loaded=False`
- Precision: bf16/fp16 require CUDA; raise error for CPU with non-fp32

## Important Notes

- **Submodules**: Run `git submodule update --init --recursive` after clone
- **HuggingFace Auth**: SAM 3 model requires accepting terms before download
- **Batch Size**: Nodes only support batch_size=1 currently
- **Text Prompts**: Single or comma-separated (e.g., "person,car,dog")
- **Mask Selection**: `mask_index` works left-to-right by mask center x-coordinate; only works with single prompt
- **CROP_DATA**: `Sam3SegmentationByFace` expects format `((w,h), (x1,y1,x2,y2))`
