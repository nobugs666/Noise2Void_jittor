# ********************************************
# draw_evaluation.py
# 功能：根据训练过程保存的csv画曲线：Loss_G, PSNR, SSIM
# ********************************************
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

parser = argparse.ArgumentParser(description='Draw Evaluation')
parser.add_argument('--arch', type=str, default='N2V_Unet', help='N2V_Unet, RGB_REDNet30, Noise2Noise_REDNet30, REDNet10, REDNet20, REDNet30')
parser.add_argument('--csv_dir', default='weights', type=str, help='directory where csv files are saved')
parser.add_argument('--out_dir', default='Plt/', type=str, help='save plt directory')
parser.add_argument('--gaussian_noise_level', type=str, default='30')
opt = parser.parse_args()

if opt.gaussian_noise_level is not None:
    opt.gaussian_noise_level = list(map(lambda x: int(x), opt.gaussian_noise_level.split(',')))

# 读取 CSV 文件
csv_file_path = opt.csv_dir +'_srf_' + str(opt.gaussian_noise_level) + '_' + str(opt.arch) + '_train_results.csv'
data_frame = pd.read_csv(csv_file_path, index_col='Epoch')
data_frame.index = range(1, len(data_frame) + 1)

# TODO 三个指标画到一个图里
# 创建一个 Matplotlib 图形，并设置子图布局
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# 绘制 Loss_G 曲线
axs[0].plot(data_frame.index, data_frame['Loss'], label='{}_{}'.format(opt.arch, opt.gaussian_noise_level), color='blue')
#axs[0].set_title('Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend(loc='upper right')
axs[0].grid(ls='-')

# 绘制 PSNR 曲线
axs[1].plot(data_frame.index, data_frame['PSNR'], label='{}_{}'.format(opt.arch, opt.gaussian_noise_level), color='green')
#axs[1].set_title('PSNR')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('PSNR(dB)')
axs[1].legend(loc='lower right')
axs[1].grid(ls='-')

# 绘制 SSIM 曲线
axs[2].plot(data_frame.index, data_frame['SSIM'], label='{}_{}'.format(opt.arch, opt.gaussian_noise_level), color='red')
# axs[2].set_title('SSIM')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('SSIM')
axs[2].legend(loc='lower right')
axs[2].grid(ls='-')
# 调整子图之间的间距
plt.tight_layout()
plt.autoscale(enable=True, axis='y', tight=True)

# 保存图像
plt_save_path = opt.out_dir + 'evalution_plt_{}_{}'.format(opt.gaussian_noise_level, opt.arch)

if not os.path.exists(opt.out_dir):
    os.makedirs(opt.out_dir)

plt.savefig(plt_save_path, bbox_inches='tight', dpi=600)
# 显示图形
plt.show()


# # # TODO 单画PSNR
# plt.figure()
# plt.rcParams['xtick.direction'] = 'in'#将x周的刻度线方向设置向内
# plt.rcParams['ytick.direction'] = 'in'#将y轴的刻度方向设置向内
# plt.plot(data_frame.index, data_frame['PSNR'], label='DnCNN', color='green')
#
# plt.xlabel('Epochs')
# plt.ylabel('Average PSNR(dB)')
# plt.legend(loc='lower right')
# plt.grid(ls='-')
#
# ax = plt.gca()
# x_major_locator = MultipleLocator(5)
# ax.xaxis.set_major_locator(x_major_locator)
#
# y_major_locator = MultipleLocator(0.2)
# ax.yaxis.set_major_locator(y_major_locator)
#
# plt.savefig(opt.out_dir + 'PSNR_plt_{}_{}'.format(opt.gaussian_noise_level, opt.arch), bbox_inches='tight', dpi=600)
# # plt.show()