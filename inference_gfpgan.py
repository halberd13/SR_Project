# upload your own images
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import torch
import shutil
import argparse
import cv2
import glob
import numpy as np
from basicsr.utils import imwrite
from gfpgan import GFPGANer

model_path = "D:/github_ncu/GFPGAN/experiments/pretrained_models/GFPGANv1.pth"
dataset_input = 'G:/My Drive/Proposal/Dataset_RFMD/RWMFD_part_2_pro'

#make directory result
dataset_result = 'G:/My Drive/Proposal/Dataset_RFMD/RWMFD_part_2_pro_result'
if os.path.isdir(dataset_result):
    shutil.rmtree(dataset_result)
os.mkdir(dataset_result)

# image = mpimg.imread(image_path)
# imgplot = plt.imshow(image)
input_list = sorted(glob.glob(os.path.join(dataset_input, '*')))
output_list = sorted(glob.glob(os.path.join(dataset_result, '*')))


if not torch.cuda.is_available():  # CPU
    import warnings
    warnings.warn('The unoptimized RealESRGAN is very slow on CPU. We do not use it. '
                  'If you really want to use it, please modify the corresponding codes.')
    bg_upsampler = None
else:
    from realesrgan import RealESRGANer
    bg_upsampler = RealESRGANer(
        scale=2,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True)  # need to set False in CPU mode

restorer = GFPGANer(
        model_path=model_path,
        upscale=4,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler='realesrgan')

# upload images
# uploaded = files.upload()


# for filename in uploaded.keys():
#   dst_path = os.path.join(upload_folder, filename)
#   print(f'move {filename} to {dst_path}')
#   shutil.move(filename, dst_path)