"""
SAM3 ComfyUI Nodes
支持开放词汇文本分割
"""

import torch
import os
import numpy as np
from PIL import Image

import comfy.model_management as mm

# 预定义调色板 - 鲜明易区分的颜色 (RGB)
MASK_COLORS = [
    (255, 0, 0),      # 红
    (0, 255, 0),      # 绿
    (0, 0, 255),      # 蓝
    (255, 255, 0),    # 黄
    (255, 0, 255),    # 品红
    (0, 255, 255),    # 青
    (255, 128, 0),    # 橙
    (128, 0, 255),    # 紫
    (0, 255, 128),    # 青绿
    (255, 0, 128),    # 玫红
]
from comfy.utils import ProgressBar

from .load_model import load_sam3_model, unload_sam3_model

# 脚本目录
script_directory = os.path.dirname(os.path.abspath(__file__))


class DownloadAndLoadSAM3Model:
    """SAM3 模型加载器"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
                "confidence_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
        }

    RETURN_TYPES = ("SAM3MODEL",)
    RETURN_NAMES = ("sam3_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "SAM3"

    def loadmodel(self, device, precision, confidence_threshold):
        """加载SAM3模型"""

        # 1. 检查精度设置
        if precision != "fp32" and device == "cpu":
            raise ValueError("CPU只支持fp32精度，请选择fp32或切换到cuda")

        # 2. 设置路径
        model_path = "/root/ComfyUI/models/sam3/sam3.pt"
        bpe_path = os.path.join(
            os.path.dirname(script_directory),
            "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
        )

        print(f"Model path: {model_path}")
        print(f"BPE path: {bpe_path}")

        # 3. 检查文件
        if not os.path.exists(bpe_path):
            raise FileNotFoundError(f"BPE文件不存在: {bpe_path}")

        if not os.path.exists(model_path):
            print(f"⚠️  模型文件不存在: {model_path}")
            print("将从Hugging Face自动下载...")
            print("注意：需要先运行 `huggingface-cli login` 并接受facebook/sam3的使用条款")

        # 4. 加载模型
        model, processor, autocast_ctx = load_sam3_model(
            model_path=model_path,
            bpe_path=bpe_path,
            device=device,
            precision=precision,
            confidence_threshold=confidence_threshold
        )

        # 5. 返回模型字典
        sam3_model = {
            "model": model,
            "processor": processor,
            "device": device,
            "precision": precision,
            "confidence_threshold": confidence_threshold,
            "autocast_ctx": autocast_ctx
        }

        return (sam3_model,)


class Sam3Segmentation:
    """
    SAM3 分割节点
    支持文本提示进行开放词汇分割
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sam3_model": ("SAM3MODEL",),
                "image": ("IMAGE",),
                "text_prompt": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "自然语言描述，如'shoe', 'person', 'red car'（单个prompt时支持mask_index选择）"
                }),
                "mask_index": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 99,
                    "step": 1,
                    "tooltip": "按从左到右顺序选择第几个mask（0开始），-1表示不选择"
                }),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "IMAGE")
    RETURN_NAMES = ("mask_merged", "mask_selected", "image")
    FUNCTION = "segment"
    CATEGORY = "SAM3"

    def segment(self, image, sam3_model, text_prompt, mask_index, keep_model_loaded):
        """
        执行SAM3分割

        Args:
            image: ComfyUI图像张量 [B, H, W, C] float32 [0, 1]
            sam3_model: SAM3模型字典
            text_prompt: 文本提示
            keep_model_loaded: 是否保持模型加载

        Returns:
            mask: ComfyUI mask张量 [1, H, W] float32 (所有实例合并)
            image: ComfyUI图像张量 [1, H, W, C] float32 (mask区域的原图)
        """

        # 提取模型组件
        processor = sam3_model["processor"]
        device = sam3_model["device"]

        # ===== Step 1: 图像格式转换 =====
        # ComfyUI: [B, H, W, C] float32 [0, 1]
        # SAM3需要: PIL Image
        B, H, W, C = image.shape

        if B > 1:
            print(f"⚠️  警告：当前只支持batch_size=1，将只处理第一张图片")

        # 转换为PIL Image
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        width, height = pil_image.size

        print(f"Processing image: {width} x {height}")

        # ===== Step 2: 设置图像 =====
        inference_state = processor.set_image(pil_image)

        # ===== Step 3: 解析文本提示 =====
        if not text_prompt or not text_prompt.strip():
            raise ValueError("必须提供text_prompt！例如：'person', 'shoe', 'car'")

        # 支持逗号分隔的多个提示
        prompts = [p.strip() for p in text_prompt.split(',') if p.strip()]
        if not prompts:
            raise ValueError("必须提供有效的text_prompt！")

        print(f"Text prompts: {prompts} ({len(prompts)} prompts)")

        # ===== Step 4: 执行推理（支持多个提示） =====
        all_masks = []
        total_objects = 0

        for i, prompt in enumerate(prompts):
            processor.reset_all_prompts(inference_state)
            inference_state = processor.set_text_prompt(
                state=inference_state,
                prompt=prompt
            )

            # 提取当前提示的结果
            masks = inference_state["masks"]  # [N, 1, H, W] bool
            scores = inference_state["scores"]  # [N,] bfloat16

            num_objects = len(masks)
            total_objects += num_objects
            print(f"  [{i+1}/{len(prompts)}] '{prompt}': {num_objects} objects")

            if num_objects > 0:
                print(f"    Scores: {scores.cpu().float().numpy()}")
                all_masks.append(masks)

        print(f"✓ Total detected: {total_objects} objects")

        # ===== Step 5: 格式转换到ComfyUI =====
        # 合并所有提示的所有实例masks
        if all_masks:
            # 拼接所有masks: list of [N_i, 1, H, W] -> [sum(N_i), 1, H, W]
            combined_masks = torch.cat(all_masks, dim=0)
            # 合并为一张: [sum(N_i), 1, H, W] -> [1, H, W]
            masks_squeezed = combined_masks.squeeze(1)  # [sum(N_i), H, W]
            merged_mask = torch.any(masks_squeezed, dim=0, keepdim=True)  # [1, H, W]
            mask_merged = merged_mask.cpu().float()
        else:
            # 无检测结果，返回空mask
            mask_merged = torch.zeros((1, H, W), dtype=torch.float32)
            combined_masks = None

        print(f"✓ Output merged mask shape: {mask_merged.shape}")

        # ===== Step 6: 按 mask_index 选择单个 mask =====
        mask_selected = None
        selected_idx = None

        if mask_index >= 0 and combined_masks is not None and len(combined_masks) > 0:
            # 只支持单个 prompt
            if len(prompts) > 1:
                print(f"⚠️  警告：mask_index 只支持单个 prompt，当前有 {len(prompts)} 个 prompt，将忽略 mask_index")
            else:
                num_instances = len(combined_masks)

                # 计算每个 mask 的中心点 x 坐标
                center_x_list = []
                for idx in range(num_instances):
                    mask_2d = combined_masks[idx].squeeze(0)  # [H, W]
                    # 获取 mask 中所有为 True 的像素坐标
                    coords = torch.nonzero(mask_2d, as_tuple=True)
                    if len(coords[1]) > 0:
                        # coords[1] 是 x 坐标（列）
                        center_x = coords[1].float().mean().item()
                    else:
                        center_x = float('inf')  # 空 mask 放最后
                    center_x_list.append((idx, center_x))

                # 按中心点 x 坐标排序（从左到右）
                center_x_list.sort(key=lambda x: x[1])
                sorted_indices = [item[0] for item in center_x_list]

                print(f"  Mask order (left to right): {sorted_indices}")
                print(f"  Center X coords: {[f'{item[1]:.1f}' for item in center_x_list]}")

                # 选择指定 index，超出范围则选最后一个
                actual_index = min(mask_index, num_instances - 1)
                selected_idx = sorted_indices[actual_index]

                print(f"  Selected: index={mask_index} -> actual_index={actual_index} -> mask_idx={selected_idx}")

                # 提取选中的 mask
                mask_selected = combined_masks[selected_idx].squeeze(0).cpu().float().unsqueeze(0)  # [1, H, W]

        # 如果没有选中，返回空 mask
        if mask_selected is None:
            mask_selected = torch.zeros((1, H, W), dtype=torch.float32)

        print(f"✓ Output selected mask shape: {mask_selected.shape}")

        # ===== Step 7: 生成彩色可视化图像 =====
        vis_image = image[0].clone()  # [H, W, C]
        alpha = 0.5  # 透明度

        if combined_masks is not None and len(combined_masks) > 0:
            # 如果选择了特定 mask，只可视化该 mask
            if selected_idx is not None:
                instance_mask = combined_masks[selected_idx].squeeze(0).cpu().float()  # [H, W]
                color = MASK_COLORS[0]  # 使用第一个颜色（红色）
                color_tensor = torch.tensor(color, dtype=torch.float32) / 255.0
                mask_3d = instance_mask.unsqueeze(-1)  # [H, W, 1]
                vis_image = vis_image * (1 - mask_3d * alpha) + color_tensor * mask_3d * alpha
                print(f"  Visualizing selected mask (idx={selected_idx})")
            else:
                # 否则可视化所有 mask（每个不同颜色）
                num_instances = len(combined_masks)
                print(f"  Visualizing {num_instances} instances with different colors")

                for idx in range(num_instances):
                    instance_mask = combined_masks[idx].squeeze(0).cpu().float()  # [H, W]
                    color = MASK_COLORS[idx % len(MASK_COLORS)]
                    color_tensor = torch.tensor(color, dtype=torch.float32) / 255.0
                    mask_3d = instance_mask.unsqueeze(-1)  # [H, W, 1]
                    vis_image = vis_image * (1 - mask_3d * alpha) + color_tensor * mask_3d * alpha

        masked_image = vis_image.unsqueeze(0)  # [1, H, W, C]

        print(f"✓ Output image shape: {masked_image.shape}")

        # ===== Step 8: 内存管理 =====
        if not keep_model_loaded:
            processor.reset_all_prompts(inference_state)
            inference_state.clear()
            mm.soft_empty_cache()

        return (mask_merged, mask_selected, masked_image)


class Sam3SegmentationByFace:
    """
    SAM3 分割节点 - 基于人脸区域选择
    根据 AutoCropFaces 输出的矩形区域选择相交面积最大的 mask
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sam3_model": ("SAM3MODEL",),
                "image": ("IMAGE",),
                "text_prompt": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "单个文本提示，如'hair', 'person'"
                }),
                "crop_data": ("CROP_DATA",),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK", "MASK", "IMAGE")
    RETURN_NAMES = ("mask_merged", "mask_selected", "image")
    FUNCTION = "segment"
    CATEGORY = "SAM3"

    def segment(self, image, sam3_model, text_prompt, crop_data, keep_model_loaded):
        """
        执行SAM3分割，选择与人脸区域相交面积最大的mask

        Args:
            image: ComfyUI图像张量 [B, H, W, C]
            sam3_model: SAM3模型字典
            text_prompt: 单个文本提示
            crop_data: AutoCropFaces输出 ((w, h), (x1, y1, x2, y2))
            keep_model_loaded: 是否保持模型加载

        Returns:
            mask_merged: 所有实例合并的 mask
            mask_selected: 相交面积最大的单个 mask
            image: 可视化图像
        """

        # 提取模型组件
        processor = sam3_model["processor"]
        device = sam3_model["device"]

        # ===== Step 1: 解析 crop_data =====
        # crop_data 格式: ((w, h), (x1, y1, x2, y2)) 或 [[w, h], [x1, y1, x2, y2]]
        try:
            if isinstance(crop_data, (list, tuple)) and len(crop_data) == 2:
                size_info, bbox_info = crop_data
                if len(bbox_info) == 4:
                    face_x1, face_y1, face_x2, face_y2 = map(int, bbox_info)
                else:
                    raise ValueError(f"bbox_info 格式错误: {bbox_info}")
            else:
                raise ValueError(f"crop_data 格式错误: {crop_data}")
        except Exception as e:
            raise ValueError(f"无法解析 crop_data: {crop_data}, 错误: {e}")

        print(f"Face region: ({face_x1}, {face_y1}) - ({face_x2}, {face_y2})")

        # ===== Step 2: 图像格式转换 =====
        B, H, W, C = image.shape

        if B > 1:
            print(f"⚠️  警告：当前只支持batch_size=1，将只处理第一张图片")

        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        width, height = pil_image.size

        print(f"Processing image: {width} x {height}")

        # ===== Step 3: 设置图像 =====
        inference_state = processor.set_image(pil_image)

        # ===== Step 4: 解析文本提示 =====
        if not text_prompt or not text_prompt.strip():
            raise ValueError("必须提供text_prompt！")

        prompt = text_prompt.strip()
        print(f"Text prompt: '{prompt}'")

        # ===== Step 5: 执行推理 =====
        processor.reset_all_prompts(inference_state)
        inference_state = processor.set_text_prompt(
            state=inference_state,
            prompt=prompt
        )

        masks = inference_state["masks"]  # [N, 1, H, W] bool
        scores = inference_state["scores"]  # [N,] bfloat16

        num_objects = len(masks)
        print(f"✓ Detected: {num_objects} objects")

        if num_objects > 0:
            print(f"  Scores: {scores.cpu().float().numpy()}")

        # ===== Step 6: 格式转换到ComfyUI =====
        if num_objects > 0:
            combined_masks = masks
            masks_squeezed = combined_masks.squeeze(1)  # [N, H, W]
            merged_mask = torch.any(masks_squeezed, dim=0, keepdim=True)  # [1, H, W]
            mask_merged = merged_mask.cpu().float()
        else:
            mask_merged = torch.zeros((1, H, W), dtype=torch.float32)
            combined_masks = None

        print(f"✓ Output merged mask shape: {mask_merged.shape}")

        # ===== Step 7: 选择与人脸区域相交面积最大的 mask =====
        mask_selected = None
        selected_idx = None

        if combined_masks is not None and len(combined_masks) > 0:
            # 创建人脸区域的矩形 mask
            face_rect = torch.zeros((H, W), dtype=torch.bool, device=combined_masks.device)
            # 确保坐标在图像范围内
            fx1 = max(0, min(face_x1, W))
            fy1 = max(0, min(face_y1, H))
            fx2 = max(0, min(face_x2, W))
            fy2 = max(0, min(face_y2, H))
            face_rect[fy1:fy2, fx1:fx2] = True

            print(f"  Face rect (clamped): ({fx1}, {fy1}) - ({fx2}, {fy2})")

            # 计算每个 mask 与人脸矩形的相交面积
            intersection_areas = []
            for idx in range(len(combined_masks)):
                mask_2d = combined_masks[idx].squeeze(0)  # [H, W]
                intersection = mask_2d & face_rect
                area = intersection.sum().item()
                intersection_areas.append((idx, area))
                print(f"    Mask {idx}: intersection area = {area}")

            # 选择相交面积最大的
            intersection_areas.sort(key=lambda x: x[1], reverse=True)
            selected_idx = intersection_areas[0][0]
            max_area = intersection_areas[0][1]

            print(f"  Selected mask {selected_idx} with max intersection area = {max_area}")

            # 提取选中的 mask
            mask_selected = combined_masks[selected_idx].squeeze(0).cpu().float().unsqueeze(0)  # [1, H, W]

        # 如果没有选中，返回空 mask
        if mask_selected is None:
            mask_selected = torch.zeros((1, H, W), dtype=torch.float32)

        print(f"✓ Output selected mask shape: {mask_selected.shape}")

        # ===== Step 8: 生成彩色可视化图像 =====
        vis_image = image[0].clone()  # [H, W, C]
        alpha = 0.5

        if selected_idx is not None:
            instance_mask = combined_masks[selected_idx].squeeze(0).cpu().float()  # [H, W]
            color = MASK_COLORS[0]  # 红色
            color_tensor = torch.tensor(color, dtype=torch.float32) / 255.0
            mask_3d = instance_mask.unsqueeze(-1)  # [H, W, 1]
            vis_image = vis_image * (1 - mask_3d * alpha) + color_tensor * mask_3d * alpha
            print(f"  Visualizing selected mask (idx={selected_idx})")

        masked_image = vis_image.unsqueeze(0)  # [1, H, W, C]

        print(f"✓ Output image shape: {masked_image.shape}")

        # ===== Step 9: 内存管理 =====
        if not keep_model_loaded:
            processor.reset_all_prompts(inference_state)
            inference_state.clear()
            mm.soft_empty_cache()

        return (mask_merged, mask_selected, masked_image)


class Sam3SegmentationOriginal:
    """
    SAM3 原始分割节点
    输出合并的mask和mask区域的原图（黑底）
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sam3_model": ("SAM3MODEL",),
                "image": ("IMAGE",),
                "text_prompt": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "自然语言描述，如'shoe', 'person', 'red car'"
                }),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "image")
    FUNCTION = "segment"
    CATEGORY = "SAM3"

    def segment(self, image, sam3_model, text_prompt, keep_model_loaded):
        """
        执行SAM3分割（原始版本）

        Returns:
            mask: 所有实例合并的mask
            image: mask区域的原图（黑底）
        """

        processor = sam3_model["processor"]
        device = sam3_model["device"]

        B, H, W, C = image.shape

        if B > 1:
            print(f"⚠️  警告：当前只支持batch_size=1，将只处理第一张图片")

        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        width, height = pil_image.size

        print(f"Processing image: {width} x {height}")

        inference_state = processor.set_image(pil_image)

        if not text_prompt or not text_prompt.strip():
            raise ValueError("必须提供text_prompt！例如：'person', 'shoe', 'car'")

        prompts = [p.strip() for p in text_prompt.split(',') if p.strip()]
        if not prompts:
            raise ValueError("必须提供有效的text_prompt！")

        print(f"Text prompts: {prompts} ({len(prompts)} prompts)")

        all_masks = []
        total_objects = 0

        for i, prompt in enumerate(prompts):
            processor.reset_all_prompts(inference_state)
            inference_state = processor.set_text_prompt(
                state=inference_state,
                prompt=prompt
            )

            masks = inference_state["masks"]
            scores = inference_state["scores"]

            num_objects = len(masks)
            total_objects += num_objects
            print(f"  [{i+1}/{len(prompts)}] '{prompt}': {num_objects} objects")

            if num_objects > 0:
                print(f"    Scores: {scores.cpu().float().numpy()}")
                all_masks.append(masks)

        print(f"✓ Total detected: {total_objects} objects")

        if all_masks:
            combined_masks = torch.cat(all_masks, dim=0)
            masks_squeezed = combined_masks.squeeze(1)
            merged_mask = torch.any(masks_squeezed, dim=0, keepdim=True)
            masks_comfy = merged_mask.cpu().float()
        else:
            masks_comfy = torch.zeros((1, H, W), dtype=torch.float32)

        print(f"✓ Output mask shape: {masks_comfy.shape}")

        # 生成masked image（黑底）
        mask_expanded = masks_comfy.unsqueeze(-1)
        masked_image = image[0:1] * mask_expanded

        print(f"✓ Output image shape: {masked_image.shape}")

        if not keep_model_loaded:
            processor.reset_all_prompts(inference_state)
            inference_state.clear()
            mm.soft_empty_cache()

        return (masks_comfy, masked_image)


class Sam3SegmentationWithDetail:
    """
    SAM3 分割节点 + VITMatte 边缘细化
    输出精细化的mask和mask区域的原图（黑底）
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sam3_model": ("SAM3MODEL",),
                "image": ("IMAGE",),
                "text_prompt": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "自然语言描述，如'shoe', 'person', 'red car'"
                }),
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "detail_erode": ("INT", {"default": 6, "min": 1, "max": 255, "step": 1,
                    "tooltip": "Trimap 腐蚀核大小，控制确定前景区域"}),
                "detail_dilate": ("INT", {"default": 4, "min": 1, "max": 255, "step": 1,
                    "tooltip": "Trimap 膨胀核大小，控制不确定区域宽度"}),
                "black_point": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.98, "step": 0.01,
                    "display": "slider", "tooltip": "直方图重映射黑点，低于此值映射为0"}),
                "white_point": ("FLOAT", {"default": 0.99, "min": 0.02, "max": 0.99, "step": 0.01,
                    "display": "slider", "tooltip": "直方图重映射白点，高于此值映射为1"}),
                "max_megapixels": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1,
                    "tooltip": "VITMatte 最大处理像素数（百万），超过则降采样"}),
            },
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "image")
    FUNCTION = "segment"
    CATEGORY = "SAM3"

    def segment(self, image, sam3_model, text_prompt, keep_model_loaded,
                detail_erode, detail_dilate, black_point, white_point, max_megapixels):
        """
        执行SAM3分割 + VITMatte边缘细化

        Returns:
            mask: 精细化后的合并mask
            image: mask区域的原图（黑底）
        """
        from .detail_utils import process_mask_with_vitmatte

        processor = sam3_model["processor"]
        device = sam3_model["device"]

        B, H, W, C = image.shape

        if B > 1:
            print(f"⚠️  警告：当前只支持batch_size=1，将只处理第一张图片")

        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        width, height = pil_image.size

        print(f"Processing image: {width} x {height}")

        inference_state = processor.set_image(pil_image)

        if not text_prompt or not text_prompt.strip():
            raise ValueError("必须提供text_prompt！例如：'person', 'shoe', 'car'")

        prompts = [p.strip() for p in text_prompt.split(',') if p.strip()]
        if not prompts:
            raise ValueError("必须提供有效的text_prompt！")

        print(f"Text prompts: {prompts} ({len(prompts)} prompts)")

        all_masks = []
        total_objects = 0

        for i, prompt in enumerate(prompts):
            processor.reset_all_prompts(inference_state)
            inference_state = processor.set_text_prompt(
                state=inference_state,
                prompt=prompt
            )

            masks = inference_state["masks"]
            scores = inference_state["scores"]

            num_objects = len(masks)
            total_objects += num_objects
            print(f"  [{i+1}/{len(prompts)}] '{prompt}': {num_objects} objects")

            if num_objects > 0:
                print(f"    Scores: {scores.cpu().float().numpy()}")
                all_masks.append(masks)

        print(f"✓ Total detected: {total_objects} objects")

        if all_masks:
            combined_masks = torch.cat(all_masks, dim=0)
            masks_squeezed = combined_masks.squeeze(1)
            merged_mask = torch.any(masks_squeezed, dim=0, keepdim=True)
            masks_comfy = merged_mask.cpu().float()
        else:
            masks_comfy = torch.zeros((1, H, W), dtype=torch.float32)

        print(f"✓ SAM3 mask shape: {masks_comfy.shape}")

        # ===== VITMatte 边缘细化 =====
        print(f"Processing VITMatte detail refinement...")
        print(f"  detail_erode={detail_erode}, detail_dilate={detail_dilate}")
        print(f"  black_point={black_point}, white_point={white_point}")
        print(f"  max_megapixels={max_megapixels}")

        refined_mask_pil = process_mask_with_vitmatte(
            image=pil_image,
            mask=masks_comfy,
            detail_erode=detail_erode,
            detail_dilate=detail_dilate,
            black_point=black_point,
            white_point=white_point,
            device=device,
            max_megapixels=max_megapixels
        )

        # 转回 tensor
        refined_mask_np = np.array(refined_mask_pil).astype(np.float32) / 255.0
        masks_comfy = torch.from_numpy(refined_mask_np).unsqueeze(0)

        print(f"✓ Refined mask shape: {masks_comfy.shape}")

        # 生成masked image（黑底）
        mask_expanded = masks_comfy.unsqueeze(-1)
        masked_image = image[0:1] * mask_expanded

        print(f"✓ Output image shape: {masked_image.shape}")

        if not keep_model_loaded:
            processor.reset_all_prompts(inference_state)
            inference_state.clear()
            mm.soft_empty_cache()

        return (masks_comfy, masked_image)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadSAM3Model": DownloadAndLoadSAM3Model,
    "Sam3SegmentationByIndex": Sam3Segmentation,
    "Sam3SegmentationByFace": Sam3SegmentationByFace,
    "Sam3SegmentationOriginal": Sam3SegmentationOriginal,
    "Sam3SegmentationWithDetail": Sam3SegmentationWithDetail,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadSAM3Model": "(Down)Load SAM3 Model",
    "Sam3SegmentationByIndex": "SAM3 Segmentation By Index",
    "Sam3SegmentationByFace": "SAM3 Segmentation By Face",
    "Sam3SegmentationOriginal": "SAM3 Segmentation Original",
    "Sam3SegmentationWithDetail": "SAM3 Segmentation With Detail",
}
