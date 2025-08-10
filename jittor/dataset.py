import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from jittor.dataset import Dataset as JtDataset
import random
import glob
import io
import numpy as np
import PIL.Image as pil_image
import copy
import jittor as jt
import torch

# def data_aug(img1, img2, mode=0):
#     # data augmentation
#     if mode == 0:
#         return img1, img2
#     elif mode == 1:
#         return np.flipud(img1), np.flipud(img2)
#     elif mode == 2:
#         return np.rot90(img1), np.rot90(img2)
#     elif mode == 3:
#         return np.flipud(np.rot90(img1)), np.flipud(np.rot90(img2))
#     elif mode == 4:
#         return np.rot90(img1, k=2), np.rot90(img2, k=2)
#     elif mode == 5:
#         return np.flipud(np.rot90(img1, k=2)), np.flipud(np.rot90(img2, k=2))
#     elif mode == 6:
#         return np.rot90(img1, k=3), np.rot90(img2, k=3)
#     elif mode == 7:
#         return np.flipud(np.rot90(img1, k=3)), np.flipud(np.rot90(img2, k=3))

def data_aug(img, mode=0):
    # data augmentation
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def gen_patch(datasets_path, patch_size, stride):
    file_list = sorted(glob.glob(datasets_path + '/*')) # 图像列表
    data = [] # 返回制作好的图像块集合
    patches = [] # 每一张图像的图像块集合

    # 按stride和patch_size得到每张图像的块集合
    for i in range(len(file_list)):
        clean_image = pil_image.open(file_list[i]).convert('RGB')
        for j in range(0, clean_image.width - patch_size + 1, stride):
            for k in range(0, clean_image.height - patch_size + 1, stride):
                x = clean_image.crop((j, k, j + patch_size, k + patch_size))
                patches.append(x)
                for m in range(0, 1):
                    x_aug = data_aug(x, mode=np.random.randint(0, 8))
                    patches.append(x_aug)

    # 得到所有的图像块集合
    for patch in patches:
        data.append(patch)

    return data

# # REDNet初始数据读取
# class Dataset(object):
#     def __init__(self, images_dir, patch_size, gaussian_noise_level, downsampling_factor, jpeg_quality):
#         # self.image_files = sorted(glob.glob(images_dir + '/*'))
#         self.image_files = images_dir
#         self.patch_size = patch_size
#         self.gaussian_noise_level = gaussian_noise_level
#         self.downsampling_factor = downsampling_factor
#         self.jpeg_quality = jpeg_quality
#
#
#     def __getitem__(self, idx):
#         # clean_image = pil_image.open(self.image_files[idx]).convert('RGB')
#         clean_image = self.image_files[idx]
#         # randomly crop patch from training set
#         # crop_x = random.randint(0, clean_image.width - self.patch_size)
#         # crop_y = random.randint(0, clean_image.height - self.patch_size)
#         # clean_image = clean_image.crop((crop_x, crop_y, crop_x + self.patch_size, crop_y + self.patch_size))
#
#         noisy_image1 = clean_image.copy()
#         # gaussian_noise1 = np.zeros((clean_image.height, clean_image.width, 3), dtype=np.float32)
#         # gaussian_noise2 = np.zeros((clean_image.height, clean_image.width, 3), dtype=np.float32)
#
#         gaussian_noise1 = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.float32)
#
#         # additive gaussian noise
#         if self.gaussian_noise_level is not None:
#             if len(self.gaussian_noise_level) == 1:
#                 sigma = self.gaussian_noise_level[0]
#             else:
#                 sigma = random.randint(self.gaussian_noise_level[0], self.gaussian_noise_level[1])
#             gaussian_noise1 += np.random.normal(0.0, sigma, (self.patch_size, self.patch_size, 3)).astype(
#                 np.float32)
#
#         # # downsampling
#         # if self.downsampling_factor is not None:
#         #     if len(self.downsampling_factor) == 1:
#         #         downsampling_factor = self.downsampling_factor[0]
#         #     else:
#         #         downsampling_factor = random.randint(self.downsampling_factor[0], self.downsampling_factor[1])
#         #
#         #     noisy_image = noisy_image.resize((self.patch_size // downsampling_factor,
#         #                                       self.patch_size // downsampling_factor),
#         #                                      resample=pil_image.BICUBIC)
#         #     noisy_image = noisy_image.resize((self.patch_size, self.patch_size), resample=pil_image.BICUBIC)
#         #
#         # # additive jpeg noise
#         # if self.jpeg_quality is not None:
#         #     if len(self.jpeg_quality) == 1:
#         #         quality = self.jpeg_quality[0]
#         #     else:
#         #         quality = random.randint(self.jpeg_quality[0], self.jpeg_quality[1])
#         #     buffer = io.BytesIO()
#         #     noisy_image.save(buffer, format='jpeg', quality=quality)
#         #     noisy_image = pil_image.open(buffer)
#
#         noisy_image1 = np.array(noisy_image1).astype(np.float32)
#         clean_image = np.array(clean_image).astype(np.float32)
#         noisy_image1 += gaussian_noise1
#
#         # for m in range(0, 1):
#         #     noisy_image1, noisy_image2 = data_aug(noisy_image1, noisy_image2, mode=np.random.randint(0, 8))
#
#         input = np.transpose(noisy_image1, axes=[2, 0, 1])
#         label = np.transpose(clean_image, axes=[2, 0, 1])
#
#         # print(input.shape)
#         # print(label.shape)
#
#         # normalization
#         input /= 255.0
#         label /= 255.0
#
#         return input, label
#
#     def __len__(self):
#         return len(self.image_files)


# # TODO noise2noise，同一图像添加相同噪声分布
# class Dataset(object):
#     def __init__(self, images_dir, patch_size, gaussian_noise_level, downsampling_factor, jpeg_quality):
#         # self.image_files = sorted(glob.glob(images_dir + '/*'))
#         self.image_files = images_dir
#         self.patch_size = patch_size
#         self.gaussian_noise_level = gaussian_noise_level
#         self.downsampling_factor = downsampling_factor
#         self.jpeg_quality = jpeg_quality
#
#
#     def __getitem__(self, idx):
#         # clean_image = pil_image.open(self.image_files[idx]).convert('RGB')
#         clean_image = self.image_files[idx]
#         # randomly crop patch from training set
#         # crop_x = random.randint(0, clean_image.width - self.patch_size)
#         # crop_y = random.randint(0, clean_image.height - self.patch_size)
#         # clean_image = clean_image.crop((crop_x, crop_y, crop_x + self.patch_size, crop_y + self.patch_size))
#
#         noisy_image1 = clean_image.copy()
#         noisy_image2 = clean_image.copy()
#         # gaussian_noise1 = np.zeros((clean_image.height, clean_image.width, 3), dtype=np.float32)
#         # gaussian_noise2 = np.zeros((clean_image.height, clean_image.width, 3), dtype=np.float32)
#
#         gaussian_noise1 = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.float32)
#         gaussian_noise2 = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.float32)
#
#         # additive gaussian noise
#         if self.gaussian_noise_level is not None:
#             if len(self.gaussian_noise_level) == 1:
#                 sigma = self.gaussian_noise_level[0]
#             else:
#                 sigma = random.randint(self.gaussian_noise_level[0], self.gaussian_noise_level[1])
#             gaussian_noise1 += np.random.normal(0.0, sigma, (self.patch_size, self.patch_size, 3)).astype(
#                 np.float32)
#             gaussian_noise2 += np.random.normal(0.0, sigma, (self.patch_size, self.patch_size, 3)).astype(
#                 np.float32)
#
#         # # downsampling
#         # if self.downsampling_factor is not None:
#         #     if len(self.downsampling_factor) == 1:
#         #         downsampling_factor = self.downsampling_factor[0]
#         #     else:
#         #         downsampling_factor = random.randint(self.downsampling_factor[0], self.downsampling_factor[1])
#         #
#         #     noisy_image = noisy_image.resize((self.patch_size // downsampling_factor,
#         #                                       self.patch_size // downsampling_factor),
#         #                                      resample=pil_image.BICUBIC)
#         #     noisy_image = noisy_image.resize((self.patch_size, self.patch_size), resample=pil_image.BICUBIC)
#         #
#         # # additive jpeg noise
#         # if self.jpeg_quality is not None:
#         #     if len(self.jpeg_quality) == 1:
#         #         quality = self.jpeg_quality[0]
#         #     else:
#         #         quality = random.randint(self.jpeg_quality[0], self.jpeg_quality[1])
#         #     buffer = io.BytesIO()
#         #     noisy_image.save(buffer, format='jpeg', quality=quality)
#         #     noisy_image = pil_image.open(buffer)
#
#         noisy_image1 = np.array(noisy_image1).astype(np.float32)
#         noisy_image2 = np.array(noisy_image2).astype(np.float32)
#         noisy_image1 += gaussian_noise1
#         noisy_image2 += gaussian_noise2
#
#         # for m in range(0, 1):
#         #     noisy_image1, noisy_image2 = data_aug(noisy_image1, noisy_image2, mode=np.random.randint(0, 8))
#
#         input = np.transpose(noisy_image1, axes=[2, 0, 1])
#         label = np.transpose(noisy_image2, axes=[2, 0, 1])
#
#         # print(input.shape)
#         # print(label.shape)
#
#         # normalization
#         input /= 255.0
#         label /= 255.0
#
#         return input, label
#
#     def __len__(self):
#         return len(self.image_files)

# TODO noise2void
class Dataset(JtDataset):
    def __init__(self, images_dir, patch_size, gaussian_noise_level, downsampling_factor, jpeg_quality, is_gray=True):
        super().__init__()
        self.image_files = sorted(glob.glob(images_dir + '/*'))
        # self.image_files = images_dir
        self.patch_size = patch_size
        self.gaussian_noise_level = gaussian_noise_level
        self.downsampling_factor = downsampling_factor
        self.jpeg_quality = jpeg_quality
        self.is_gray = is_gray
        self.batch_size = None,
        self.shuffle = True,
        self.num_workers = None,
        self.pin_memory = True,
        self.drop_last = True


    @staticmethod
    def random_horizontal_flip(lr, hr):
        if random.random() < 0.5:
            lr = lr[:, ::-1, :].copy()
            hr = hr[:, ::-1, :].copy()
        return lr, hr

    @staticmethod
    def random_vertical_flip(lr, hr):
        if random.random() < 0.5:
            lr = lr[::-1, :, :].copy()
            hr = hr[::-1, :, :].copy()
        return lr, hr

    @staticmethod
    def random_rotate_90(lr, hr):
        if random.random() < 0.5:
            lr = np.rot90(lr, axes=(1, 0)).copy()
            hr = np.rot90(hr, axes=(1, 0)).copy()
        return lr, hr

    def __getitem__(self, idx):
        idx = idx % len(self.image_files)
        if self.is_gray:
            clean_image1 = pil_image.open(self.image_files[idx]).convert('L')
        else:
            clean_image1 = pil_image.open(self.image_files[idx]).convert('RGB')
        # clean_image1 = self.image_files[idx]

        # randomly crop patch from training set
        crop_x = random.randint(0, clean_image1.width - self.patch_size)
        crop_y = random.randint(0, clean_image1.height - self.patch_size)
        clean_image1 = clean_image1.crop((crop_x, crop_y, crop_x + self.patch_size, crop_y + self.patch_size))

        noisy_image1 = clean_image1.copy()
        # gaussian_noise1 = np.zeros((clean_image.height, clean_image.width, 3), dtype=np.float32)
        # gaussian_noise2 = np.zeros((clean_image.height, clean_image.width, 3), dtype=np.float32)

        # additive gaussian noise
        if self.gaussian_noise_level is not None:
            if len(self.gaussian_noise_level) == 1:
                sigma = self.gaussian_noise_level[0]
            else:
                sigma = random.randint(self.gaussian_noise_level[0], self.gaussian_noise_level[1])

            if self.is_gray:
                gaussian_noise1 = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
                gaussian_noise1 += np.random.normal(0.0, sigma, (self.patch_size, self.patch_size)).astype(
                    np.float32)
            else:
                gaussian_noise1 = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.float32)
                gaussian_noise1 += np.random.normal(0.0, sigma, (self.patch_size, self.patch_size, 3)).astype(
                    np.float32)

        # # downsampling
        # if self.downsampling_factor is not None:
        #     if len(self.downsampling_factor) == 1:
        #         downsampling_factor = self.downsampling_factor[0]
        #     else:
        #         downsampling_factor = random.randint(self.downsampling_factor[0], self.downsampling_factor[1])
        #
        #     noisy_image = noisy_image.resize((self.patch_size // downsampling_factor,
        #                                       self.patch_size // downsampling_factor),
        #                                      resample=pil_image.BICUBIC)
        #     noisy_image = noisy_image.resize((self.patch_size, self.patch_size), resample=pil_image.BICUBIC)
        #
        # # additive jpeg noise
        # if self.jpeg_quality is not None:
        #     if len(self.jpeg_quality) == 1:
        #         quality = self.jpeg_quality[0]
        #     else:
        #         quality = random.randint(self.jpeg_quality[0], self.jpeg_quality[1])
        #     buffer = io.BytesIO()
        #     noisy_image.save(buffer, format='jpeg', quality=quality)
        #     noisy_image = pil_image.open(buffer)

        noisy_image1 = np.array(noisy_image1).astype(np.float32)
        noisy_image1 += gaussian_noise1

        if self.is_gray:
            noisy_image1 = np.expand_dims(noisy_image1, axis=-1)
            source, target, blid_pos = self.center_pixel_mask(noisy_image1)
        else:
            source, target, blid_pos = self.center_pixel_mask(noisy_image1)


        source, target = self.random_horizontal_flip(source, target)
        source, target = self.random_vertical_flip(source, target)
        source, target = self.random_rotate_90(source, target)

        if self.is_gray:
            # source = np.expand_dims(source,axis=-1)
            # target = np.expand_dims(target,axis=-1)
            input = np.transpose(source, axes=[2, 0, 1])
            label = np.transpose(target, axes=[2, 0, 1])
        else:
            input = np.transpose(source, axes=[2, 0, 1])
            label = np.transpose(target, axes=[2, 0, 1])

        # normalization
        input /= 255.0
        label /= 255.0

        input = jt.array(input, dtype=jt.float32)
        label = jt.array(label, dtype=jt.float32)
        blid_pos = jt.array(blid_pos)
        # print(input.shape)
        # print(label.shape)
        # print(blid_pos.shape)

        return input, label, blid_pos

    def __len__(self):
        return len(self.image_files) * 20

    def center_pixel_mask(self, img):

        h, w, _ = img.shape  # 获取图像维度

        center_pixel_h = h // 2  # 中心像素坐标
        center_pixel_w = w // 2

        random_pixel_h = np.random.randint(0, h)  # 随机位置
        random_pixel_w = np.random.randint(0, w)

        # 如果随机位置是中心位置，则重新选取，直到不相同为止。
        while (center_pixel_h == random_pixel_h) and (center_pixel_w == random_pixel_w):
            random_pixel_h = np.random.randint(0, h)
            random_pixel_w = np.random.randint(0, w)

        target = copy.deepcopy(img)
        source = copy.deepcopy(img)

        # 用随机像素替换中心像素
        source[center_pixel_h, center_pixel_w, :] = img[random_pixel_h, random_pixel_w, :]

        # 返回源图像、目标图像和中心像素位置
        return source, target, [center_pixel_h, center_pixel_w]


# TODO REDNet原始验证集
class EvalDataset(JtDataset):
    def __init__(self, clean_image_dir, gaussian_noise_level, downsampling_factor, jpeg_quality, is_gray=False):
        super().__init__()  # 调用父类初始化
        self.clean_image_files = sorted(glob.glob(clean_image_dir + '/*'))
        self.gaussian_noise_level = gaussian_noise_level
        self.downsampling_factor = downsampling_factor
        self.jpeg_quality = jpeg_quality
        self.is_gray = is_gray
        self.batch_size = 1
    def __getitem__(self, idx):
        if self.is_gray:
            clean_image = pil_image.open(self.clean_image_files[idx]).convert('L')
        else:
            clean_image = pil_image.open(self.clean_image_files[idx]).convert('RGB')

        noisy_image1 = clean_image.copy()

        # additive gaussian noise
        if self.gaussian_noise_level is not None:
            if len(self.gaussian_noise_level) == 1:
                sigma = self.gaussian_noise_level[0]
            else:
                sigma = random.randint(self.gaussian_noise_level[0], self.gaussian_noise_level[1])
            if self.is_gray:
                gaussian_noise1 = np.zeros((clean_image.height, clean_image.width), dtype=np.float32)
                gaussian_noise1 += np.random.normal(0.0, sigma, (clean_image.height, clean_image.width)).astype(
                    np.float32)
            else:
                gaussian_noise1 = np.zeros((clean_image.height, clean_image.width, 3), dtype=np.float32)
                gaussian_noise1 += np.random.normal(0.0, sigma, (clean_image.height, clean_image.width, 3)).astype(
                    np.float32)

        # # downsampling
        # if self.downsampling_factor is not None:
        #     if len(self.downsampling_factor) == 1:
        #         downsampling_factor = self.downsampling_factor[0]
        #     else:
        #         downsampling_factor = random.randint(self.downsampling_factor[0], self.downsampling_factor[1])
        #
        #     noisy_image = noisy_image.resize((clean_image.height // downsampling_factor,
        #                                        clean_image.width // downsampling_factor),
        #                                      resample=pil_image.BICUBIC)
        #     noisy_image = noisy_image.resize((clean_image.height, clean_image.width), resample=pil_image.BICUBIC)
        #
        # # additive jpeg noise
        # if self.jpeg_quality is not None:
        #     if len(self.jpeg_quality) == 1:
        #         quality = self.jpeg_quality[0]
        #     else:
        #         quality = random.randint(self.jpeg_quality[0], self.jpeg_quality[1])
        #     buffer = io.BytesIO()
        #     noisy_image.save(buffer, format='jpeg', quality=quality)
        #     noisy_image = pil_image.open(buffer)
        #
        noisy_image1 = np.array(noisy_image1).astype(np.float32)
        clean_image = np.array(clean_image).astype(np.float32)
        noisy_image1 += gaussian_noise1

        if self.is_gray:
            noisy_image1 = np.expand_dims(noisy_image1, axis=-1)
            clean_image = np.expand_dims(clean_image, axis=-1)
            input = np.transpose(noisy_image1, axes=[2, 0, 1])
            label = np.transpose(clean_image, axes=[2, 0, 1])
        else:
            input = np.transpose(noisy_image1, axes=[2, 0, 1])
            label = np.transpose(clean_image, axes=[2, 0, 1])


        # normalization
        input /= 255.0
        label /= 255.0

        return input, label

    def __len__(self):
        return len(self.clean_image_files)

# # TODO noise2noise
# # 验证集，BSD68有原图像和加噪后的图像，直接读取
# class EvalDataset(object):
#     def __init__(self, clean_image_dir, gaussian_noise_level, downsampling_factor, jpeg_quality):
#         self.clean_image_files = sorted(glob.glob(clean_image_dir + '/*'))
#         self.gaussian_noise_level = gaussian_noise_level
#         self.downsampling_factor = downsampling_factor
#         self.jpeg_quality = jpeg_quality
#
#
#     def __getitem__(self, idx):
#         clean_image = pil_image.open(self.clean_image_files[idx]).convert('RGB')
#
#         noisy_image1 = clean_image.copy()
#         noisy_image2 = clean_image.copy()
#         gaussian_noise1 = np.zeros((clean_image.height, clean_image.width, 3), dtype=np.float32)
#         gaussian_noise2 = np.zeros((clean_image.height, clean_image.width, 3), dtype=np.float32)
#
#         # additive gaussian noise
#         if self.gaussian_noise_level is not None:
#             if len(self.gaussian_noise_level) == 1:
#                 sigma = self.gaussian_noise_level[0]
#             else:
#                 sigma = random.randint(self.gaussian_noise_level[0], self.gaussian_noise_level[1])
#             gaussian_noise1 += np.random.normal(0.0, sigma, (clean_image.height, clean_image.width, 3)).astype(
#                 np.float32)
#             gaussian_noise2 += np.random.normal(0.0, sigma, (clean_image.height, clean_image.width, 3)).astype(
#                 np.float32)
#
#         # downsampling
#         if self.downsampling_factor is not None:
#             if len(self.downsampling_factor) == 1:
#                 downsampling_factor = self.downsampling_factor[0]
#             else:
#                 downsampling_factor = random.randint(self.downsampling_factor[0], self.downsampling_factor[1])
#
#             noisy_image = noisy_image.resize((clean_image.height // downsampling_factor,
#                                                clean_image.width // downsampling_factor),
#                                              resample=pil_image.BICUBIC)
#             noisy_image = noisy_image.resize((clean_image.height, clean_image.width), resample=pil_image.BICUBIC)
#
#         # additive jpeg noise
#         if self.jpeg_quality is not None:
#             if len(self.jpeg_quality) == 1:
#                 quality = self.jpeg_quality[0]
#             else:
#                 quality = random.randint(self.jpeg_quality[0], self.jpeg_quality[1])
#             buffer = io.BytesIO()
#             noisy_image.save(buffer, format='jpeg', quality=quality)
#             noisy_image = pil_image.open(buffer)
#
#         noisy_image1 = np.array(noisy_image1).astype(np.float32)
#         noisy_image2 = np.array(noisy_image2).astype(np.float32)
#         noisy_image1 += gaussian_noise1
#         noisy_image2 += gaussian_noise2
#
#         input = np.transpose(noisy_image1, axes=[2, 0, 1])
#         label = np.transpose(noisy_image2, axes=[2, 0, 1])
#
#         # normalization
#         input /= 255.0
#         label /= 255.0
#
#         return input, label
#
#     def __len__(self):
#         return len(self.clean_image_files)