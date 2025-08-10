import os
import cv2
import numpy as np
from pathlib import Path


def crop_images_to_patches(input_dir, output_dir, patch_size=64):
    """
    将BSD68数据集中的图像切割成指定大小的图像块

    Args:
        input_dir (str): 原始图像目录路径
        output_dir (str): 裁剪后图像保存目录路径
        patch_size (int): 裁剪图像块的大小 (默认: 64x64)
    """
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_files = [f for f in os.listdir(input_dir)
                   if os.path.splitext(f)[1].lower() in image_extensions]

    patch_count = 0

    for img_file in image_files:
        # 读取图像
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)

        if img is None:
            print(f"无法读取图像: {img_path}")
            continue

        h, w = img.shape[:2]

        # 计算可以切割的图像块数量
        n_patches_h = h // patch_size
        n_patches_w = w // patch_size

        # 切割图像
        for i in range(n_patches_h):
            for j in range(n_patches_w):
                # 计算裁剪区域
                start_h = i * patch_size
                end_h = start_h + patch_size
                start_w = j * patch_size
                end_w = start_w + patch_size

                # 裁剪图像块
                patch = img[start_h:end_h, start_w:end_w]

                # 保存图像块
                patch_filename = f"{os.path.splitext(img_file)[0]}_patch_{patch_count:04d}.png"
                patch_path = os.path.join(output_dir, patch_filename)
                cv2.imwrite(patch_path, patch)

                patch_count += 1

        print(f"已处理 {img_file}，生成 {n_patches_h * n_patches_w} 个图像块")

    print(f"总共生成 {patch_count} 个图像块，保存在 {output_dir}")


# 使用示例
if __name__ == "__main__":
    # 设置输入和输出路径
    input_directory = "datasets/BSD68_valid"  # 原始BSD68数据集路径
    output_directory = "datasets/BSD68_valid_crop"  # 裁剪后图像保存路径
    patch_size = 64  # 图像块大小

    # 执行切割操作
    crop_images_to_patches(input_directory, output_directory, patch_size)
