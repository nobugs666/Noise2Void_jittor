import os
import shutil
import random
from PIL import Image


def resize_image_to_480x320(input_path, output_path, method='crop'):
    """
    将图片调整为480x320尺寸

    Args:
        input_path (str): 输入图片路径
        output_path (str): 输出图片路径
        method (str): 处理方法 'resize' 或 'crop'
    """
    with Image.open(input_path) as img:
        if method == 'resize':
            # 方法1: 直接缩放到480x320
            resized_img = img.resize((480, 320), Image.Resampling.LANCZOS)
            resized_img.save(output_path)
        elif method == 'crop':
            # 方法2: 从中心裁剪到480x320
            width, height = img.size

            # 计算裁剪区域（从中心裁剪）
            left = (width - 480) // 2
            top = (height - 320) // 2
            right = left + 480
            bottom = top + 320

            # 裁剪并保存
            cropped_img = img.crop((left, top, right, bottom))
            cropped_img.save(output_path)


def split_bsd68_dataset(source_dir, valid_dir, test_dir, valid_count=34, crop_method='crop'):
    """
    将BSD68数据集划分为验证集和测试集，并调整图片尺寸为480x320

    Args:
        source_dir: 原始BSD68数据集目录
        valid_dir: 验证集目录
        test_dir: 测试集目录
        valid_count: 验证集图片数量
        crop_method: 裁剪方法 'crop'（裁剪）或 'resize'（缩放）
    """
    # 创建目录
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 获取所有图片文件
    image_files = [f for f in os.listdir(source_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif'))]

    # 随机打乱
    random.seed(123)  # 固定种子以保证可重复性
    random.shuffle(image_files)

    # 划分数据集
    valid_files = image_files[:valid_count]
    test_files = image_files[valid_count:]

    # 处理并复制验证集文件
    print("处理验证集...")
    for file in valid_files:
        input_path = os.path.join(source_dir, file)
        output_path = os.path.join(valid_dir, file)
        try:
            resize_image_to_480x320(input_path, output_path, crop_method)
        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")
            # 如果处理失败，直接复制原文件
            shutil.copy(input_path, output_path)

    # 处理并复制测试集文件
    print("处理测试集...")
    for file in test_files:
        input_path = os.path.join(source_dir, file)
        output_path = os.path.join(test_dir, file)
        try:
            resize_image_to_480x320(input_path, output_path, crop_method)
        except Exception as e:
            print(f"处理文件 {file} 时出错: {e}")
            # 如果处理失败，直接复制原文件
            shutil.copy(input_path, output_path)

    print(f"验证集: {len(valid_files)} 张图片")
    print(f"测试集: {len(test_files)} 张图片")
    print(f"使用处理方法: {'裁剪' if crop_method == 'crop' else '缩放'}")


# 使用示例
if __name__ == "__main__":
    # 使用裁剪方法（推荐）
    split_bsd68_dataset('datasets/BSD68', 'datasets/BSD68_valid', 'datasets/BSD68_test', 34, 'crop')

    # 或者使用缩放方法
    # split_bsd68_dataset('datasets/BSD68_crop', 'datasets/BSD68_valid', 'datasets/BSD68_test', 34, 'resize')
