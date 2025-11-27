"""
VITMatte Detail Utils for SAM3
基于 ComfyUI_LayerStyle 的 VITMatte 实现，用于边缘细化
"""

import os
import math
import copy
import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path

import folder_paths


# ============== Tensor/PIL 转换 ==============

def tensor2pil(tensor: torch.Tensor) -> Image.Image:
    """将 tensor 转换为 PIL Image"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.dim() == 3:
        if tensor.shape[0] in [1, 3, 4]:  # CHW format
            tensor = tensor.permute(1, 2, 0)
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(-1)

    np_image = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

    if np_image.shape[-1] == 1:
        return Image.fromarray(np_image.squeeze(-1), mode='L')
    elif np_image.shape[-1] == 3:
        return Image.fromarray(np_image, mode='RGB')
    elif np_image.shape[-1] == 4:
        return Image.fromarray(np_image, mode='RGBA')
    else:
        return Image.fromarray(np_image.squeeze(-1), mode='L')


def pil2tensor(image: Image.Image) -> torch.Tensor:
    """将 PIL Image 转换为 tensor"""
    np_image = np.array(image).astype(np.float32) / 255.0
    if np_image.ndim == 2:
        np_image = np_image[:, :, np.newaxis]
    return torch.from_numpy(np_image).unsqueeze(0)


# ============== VITMatte 模型 ==============

class VITMatteModel:
    """VITMatte 模型包装类"""
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor


def load_VITMatte_model(local_files_only: bool = True) -> VITMatteModel:
    """
    加载 VITMatte 模型

    Args:
        local_files_only: 是否只从本地加载（默认 True）

    Returns:
        VITMatteModel 实例
    """
    from transformers import VitMatteImageProcessor, VitMatteForImageMatting

    if local_files_only:
        model_path = Path(os.path.join(folder_paths.models_dir, "vitmatte"))
        if not model_path.exists():
            raise FileNotFoundError(
                f"VITMatte 模型未找到: {model_path}\n"
                f"请从 HuggingFace 下载 'hustvl/vitmatte-small-composition-1k' 到该目录"
            )
        model = VitMatteForImageMatting.from_pretrained(model_path, local_files_only=True)
        processor = VitMatteImageProcessor.from_pretrained(model_path, local_files_only=True)
    else:
        model_name = "hustvl/vitmatte-small-composition-1k"
        model = VitMatteForImageMatting.from_pretrained(model_name)
        processor = VitMatteImageProcessor.from_pretrained(model_name)

    return VITMatteModel(model, processor)


# ============== Trimap 生成 ==============

def generate_VITMatte_trimap(mask: torch.Tensor, erode_kernel_size: int, dilate_kernel_size: int) -> Image.Image:
    """
    生成 VITMatte 需要的 trimap

    Args:
        mask: 输入 mask tensor [1, H, W] 或 [H, W]
        erode_kernel_size: 腐蚀核大小（控制确定前景区域）
        dilate_kernel_size: 膨胀核大小（控制不确定区域宽度）

    Returns:
        trimap PIL Image (L mode)
    """
    def _generate_trimap(mask_np, erode_size=10, dilate_size=10):
        erode_kernel = np.ones((erode_size, erode_size), np.uint8)
        dilate_kernel = np.ones((dilate_size, dilate_size), np.uint8)

        # 腐蚀得到确定前景区域
        eroded = cv2.erode(mask_np, erode_kernel, iterations=5)
        # 膨胀得到确定背景外边界
        dilated = cv2.dilate(mask_np, dilate_kernel, iterations=5)

        # 构建 trimap: 0=背景, 128=未知区域, 255=前景
        trimap = np.zeros_like(mask_np)
        trimap[dilated == 255] = 128  # 未知区域
        trimap[eroded == 255] = 255   # 确定前景
        return trimap

    # 处理 tensor 维度
    if mask.dim() == 3:
        mask = mask.squeeze(0)

    # 转换为 numpy uint8
    mask_np = mask.cpu().detach().numpy().astype(np.uint8) * 255

    # 生成 trimap
    trimap = _generate_trimap(mask_np, erode_kernel_size, dilate_kernel_size).astype(np.float32)

    # 转换为 0, 0.5, 1 的浮点格式
    trimap[trimap == 128] = 0.5
    trimap[trimap == 255] = 1.0

    trimap_tensor = torch.from_numpy(trimap).unsqueeze(0)
    return tensor2pil(trimap_tensor).convert('L')


# ============== VITMatte 推理 ==============

def generate_VITMatte(
    image: Image.Image,
    trimap: Image.Image,
    device: str = "cuda",
    max_megapixels: float = 2.0
) -> Image.Image:
    """
    使用 VITMatte 生成精细 alpha matte

    Args:
        image: 输入 RGB 图像
        trimap: trimap 图像 (L mode)
        device: 计算设备
        max_megapixels: 最大处理像素数（百万），超过则降采样

    Returns:
        精细化的 mask (PIL Image, L mode)
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    if trimap.mode != 'L':
        trimap = trimap.convert('L')

    # 计算是否需要降采样
    max_pixels = max_megapixels * 1048576
    width, height = image.size
    original_size = (width, height)

    if width * height > max_pixels:
        ratio = width / height
        target_width = int(math.sqrt(ratio * max_pixels))
        target_height = int(target_width / ratio)
        image = image.resize((target_width, target_height), Image.BILINEAR)
        trimap = trimap.resize((target_width, target_height), Image.BILINEAR)
        print(f"VITMatte: 图像 {width}x{height} 过大，降采样至 {target_width}x{target_height}")

    # 设置设备
    if device == "cuda" and torch.cuda.is_available():
        torch_device = torch.device('cuda')
    else:
        torch_device = torch.device('cpu')
        if device == "cuda":
            print("VITMatte: CUDA 不可用，使用 CPU")

    # 加载模型（本地模式）
    vit_matte_model = load_VITMatte_model(local_files_only=True)
    vit_matte_model.model.to(torch_device)

    # 推理
    inputs = vit_matte_model.processor(images=image, trimaps=trimap, return_tensors="pt")
    with torch.no_grad():
        inputs = {k: v.to(torch_device) for k, v in inputs.items()}
        predictions = vit_matte_model.model(**inputs).alphas

    # 清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 后处理
    mask = tensor2pil(predictions).convert('L')
    # 移除 padding（VITMatte 会添加 32px 对齐的 padding）
    mask = mask.crop((0, 0, image.width, image.height))

    # 如果降采样了，恢复原尺寸
    if original_size != image.size:
        mask = mask.resize(original_size, Image.BILINEAR)

    return mask


# ============== 直方图重映射 ==============

def histogram_remap(image: torch.Tensor, black_point: float, white_point: float) -> torch.Tensor:
    """
    直方图重映射，增强 mask 对比度

    Args:
        image: 输入 tensor
        black_point: 黑点阈值，低于此值映射为 0
        white_point: 白点阈值，高于此值映射为 1

    Returns:
        重映射后的 tensor
    """
    bp = min(black_point, white_point - 0.001)
    scale = 1.0 / (white_point - bp)

    result = copy.deepcopy(image.cpu().numpy())
    result = np.clip((result - bp) * scale, 0.0, 1.0)

    return torch.from_numpy(result)


# ============== 完整的边缘细化处理 ==============

def process_mask_with_vitmatte(
    image: Image.Image,
    mask: torch.Tensor,
    detail_erode: int = 6,
    detail_dilate: int = 4,
    black_point: float = 0.15,
    white_point: float = 0.99,
    device: str = "cuda",
    max_megapixels: float = 2.0
) -> Image.Image:
    """
    使用 VITMatte 对 mask 进行边缘细化的完整流程

    Args:
        image: 原始 RGB 图像 (PIL)
        mask: SAM3 生成的粗糙 mask tensor [1, H, W]
        detail_erode: trimap 腐蚀核大小
        detail_dilate: trimap 膨胀核大小
        black_point: 直方图重映射黑点
        white_point: 直方图重映射白点
        device: 计算设备
        max_megapixels: VITMatte 最大处理像素数

    Returns:
        细化后的 mask (PIL Image, L mode)
    """
    # Step 1: 生成 trimap
    trimap = generate_VITMatte_trimap(mask, detail_erode, detail_dilate)

    # Step 2: VITMatte 推理
    refined_mask = generate_VITMatte(image, trimap, device=device, max_megapixels=max_megapixels)

    # Step 3: 直方图重映射
    refined_mask_tensor = pil2tensor(refined_mask)
    refined_mask_tensor = histogram_remap(refined_mask_tensor, black_point, white_point)
    refined_mask = tensor2pil(refined_mask_tensor).convert('L')

    return refined_mask
