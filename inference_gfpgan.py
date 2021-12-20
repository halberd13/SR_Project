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
import tensorflow_hub as hub
import tensorflow as tf
from super_resolution.model.srgan import generator, discriminator
from PIL import Image

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# pre_generator = generator()
# gan_generator = generator()

# pre_generator.load_weights('D:\github_ncu\SR_Project\super_resolution\weights\srgan\pre_generator.h5')
# gan_generator.load_weights('D:\github_ncu\SR_Project\super_resolution\weights\srgan\gan_generator.h5')



dataset_path = 'D:\github_ncu\SR_Project\dataset\RWMFD'

model_path_gfpgan = "D:/github_ncu/GFPGAN/experiments/pretrained_models/GFPGANCleanv1-NoCE-C2.pth"
model_path_reasgan = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth'
model_path_esrgan = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
input_folder_list = []
output_folder_list = []

def load_image(path):
    return np.array(Image.open(path))

def resolve_single(model, lr):
    return resolve(model, tf.expand_dims(lr, axis=0))[0]


def resolve(model, lr_batch):
    lr_batch = tf.cast(lr_batch, tf.float32)
    sr_batch = model(lr_batch)
    sr_batch = tf.clip_by_value(sr_batch, 0, 255)
    sr_batch = tf.round(sr_batch)
    sr_batch = tf.cast(sr_batch, tf.uint8)
    return sr_batch

def rct_folder_result():
    check_folder = 0
    for dfolder in os.listdir(dataset_path):
        if dfolder.endswith("result"):
            check_folder+=1
    for dfolder in os.listdir(dataset_path):
        dpath=""
        if check_folder > 0 :
            if dfolder.endswith("result"):
                dpath = os.path.join(dataset_path, dfolder)
                shutil.rmtree(dpath)
                print(f"deleting folders result...{dfolder}")
                cpath = os.path.join(dataset_path, dfolder)
                os.mkdir(cpath)
                print(f"create folders result...{dfolder}")
                output_folder_list.append(dfolder)
            else:
                input_folder_list.append(dfolder)
        else:
            cpath = os.path.join(dataset_path, dfolder+"_result")
            os.mkdir(cpath)
            print(f"create folders result...{dfolder}_result")    
            input_folder_list.append(dfolder)
            output_folder_list.append(dfolder+"_result")

def preprocess_image(image_path):
    """ Loads image from path and preprocesses to make it model ready
        Args:
        image_path: Path to the image file
    """
    hr_image = tf.image.decode_image(tf.io.read_file(image_path))
    # If PNG, remove the alpha channel. The model only supports
    # images with 3 color channels.
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[...,:-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)

def save_image(image, filename):
    """
    Saves unscaled Tensor Images.
    Args:
        image: 3D image tensor. [height, width, channels]
        filename: Name of the file to save.
    """
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
        image.save("D:\github_ncu\SR_Project\dataset\RWMFD\srgan\%s.jpg" % filename)
        print("Saved as %s.jpg" % filename)

# # rebuild the folder result to make it clean
rct_folder_result()

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

if not torch.cuda.is_available():  # CPU
    import warnings
    warnings.warn('The unoptimized RealESRGAN is very slow on CPU. We do not use it. '
                'If you really want to use it, please modify the corresponding codes.')
    bg_upsampler = None
else:
    from realesrgan import RealESRGANer
    bg_upsampler = RealESRGANer(
        scale=2,
        model_path=model_path_reasgan,
        tile=400,
        tile_pad=5,
        pre_pad=0,
        half=True)  # need to set False in CPU mode

restorer = GFPGANer(
            model_path=model_path_gfpgan,
            upscale=4,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler='realesrgan')


# model = hub.load(model_path_esrgan)
# with tf.device('/gpu:0'):
for inp_folders in input_folder_list:
    
    args.save_root = os.path.join(dataset_path, inp_folders)     

    img_list = sorted(glob.glob(os.path.join(args.save_root, '*')))
    for img_path in img_list:
        
        img_name = os.path.basename(img_path)
        print(f'Processing directory: {args.save_root} in image: {img_name} ...')
        basename, ext = os.path.splitext(img_name)
        #handling for only extension jpg ,png 
        if not ext.lower() in (".jpg",".png"):
            continue
        input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        # lr = load_image(img_path)
        # pre_sr = resolve_single(pre_generator, lr)
        # gan_sr = resolve_single(gan_generator, lr)
        # save_image(tf.squeeze(gan_sr), basename+"_srgan")
        
        # hr_image = preprocess_image(img_path)
        # fake_image = model(hr_image)
        # fake_image = tf.squeeze(fake_image)
        # save_image(tf.squeeze(fake_image), basename+"_srgan")
        
        # restore faces and background if necessary
        cropped_faces, restored_faces, restored_img = restorer.enhance(
            input_img, has_aligned=True, only_center_face=True, paste_back=False)

        # save faces
        for idx, (cropped_face, restored_face) in enumerate(zip(cropped_faces, restored_faces)):
            save_result = args.save_root+"_result"
            # save comparison image
            cmp_img = np.concatenate((cropped_face, restored_face), axis=1)
            imwrite(cmp_img, os.path.join(save_result, 'cmp', f'{basename}_{idx:02d}.png'))

print(f'Results are in the [{args.save_root}] folder.')
