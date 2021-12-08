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

model_path = "D:/github_ncu/GFPGAN/experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth"
dataset_input = 'G:/My Drive/Proposal/Dataset_RFMD/RWMFD_part_2_pro/00000'

#make directory result
dataset_result = 'G:/My Drive/Proposal/Dataset_RFMD/RWMFD_part_2_pro_result/00000'
if os.path.isdir(dataset_result):
    shutil.rmtree(dataset_result)
os.mkdir(dataset_result)

# image = mpimg.imread(image_path)
# imgplot = plt.imshow(image)
input_list = sorted(glob.glob(os.path.join(dataset_input, '*')))
output_list = sorted(glob.glob(os.path.join(dataset_result, '*')))

parser = argparse.ArgumentParser()
parser.add_argument('--upscale', type=int, default=2, help='The final upsampling scale of the image')
parser.add_argument('--arch', type=str, default='clean', help='The GFPGAN architecture. Option: clean | original')
parser.add_argument('--channel', type=int, default=2, help='Channel multiplier for large networks of StyleGAN2')
parser.add_argument('--model_path', type=str, default='experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth')
parser.add_argument('--bg_upsampler', type=str, default='realesrgan', help='background upsampler')
parser.add_argument(
    '--bg_tile', type=int, default=400, help='Tile size for background sampler, 0 for no tile during testing')
parser.add_argument('--test_path', type=str, default='inputs/whole_imgs', help='Input folder')
parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces')
parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face')
parser.add_argument('--aligned', action='store_true', help='Input are aligned faces')
parser.add_argument('--paste_back', action='store_false', help='Paste the restored faces back to images')
parser.add_argument('--save_root', type=str, default='results', help='Path to save root')
parser.add_argument(
    '--ext',
    type=str,
    default='auto',
    help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
args = parser.parse_args()

args = parser.parse_args()
if args.test_path.endswith('/'):
    args.test_path = args.test_path[:-1]
os.makedirs(args.save_root, exist_ok=True)



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

img_list = sorted(glob.glob(os.path.join(dataset_input, '*')))
for img_path in img_list:
    # read image
    img_name = os.path.basename(img_path)
    print(f'Processing {img_name} ...')
    basename, ext = os.path.splitext(img_name)
    input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # restore faces and background if necessary
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        input_img, has_aligned="store_true", only_center_face="store_false", paste_back="store_true")

    # save faces
    for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
        # save cropped face
        save_crop_path = os.path.join(args.save_root, 'cropped_faces', f'{basename}_{idx:02d}.png')
        imwrite(cropped_face, save_crop_path)
        # save restored face
        if args.suffix is not None:
            save_face_name = f'{basename}_{idx:02d}_{args.suffix}.png'
        else:
            save_face_name = f'{basename}_{idx:02d}.png'
        save_restore_path = os.path.join(args.save_root, 'restored_faces', save_face_name)
        imwrite(restored_face, save_restore_path)
        # save comparison image
        cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
        imwrite(cmp_img, os.path.join(args.save_root, 'cmp', f'{basename}_{idx:02d}.png'))

    # save restored img
    if restored_img is not None:
        if args.ext == 'auto':
            extension = ext[1:]
        else:
            extension = args.ext

        if args.suffix is not None:
            save_restore_path = os.path.join(args.save_root, 'restored_imgs',
                                                f'{basename}_{args.suffix}.{extension}')
        else:
            save_restore_path = os.path.join(args.save_root, 'restored_imgs', f'{basename}.{extension}')
        imwrite(restored_img, save_restore_path)

print(f'Results are in the [{args.save_root}] folder.')

# upload images
# uploaded = files.upload()


# for filename in uploaded.keys():
#   dst_path = os.path.join(upload_folder, filename)
#   print(f'move {filename} to {dst_path}')
#   shutil.move(filename, dst_path)