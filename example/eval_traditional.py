import json
import sys
import os
import cv2
import subprocess
from skimage.metrics import structural_similarity as ssim
import numpy as np
from tqdm import tqdm
sys.path.append(os.getcwd())

import argparse


qualities_jepg = [10,15,22,31,42]
qualities_bpg = [34,32,30,27,25,23,21,19]
qualities_vvc = [45,43,40,38,36,34,32,30]
class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_ssim(image1, image2):
    """
    计算两张 3 通道 RGB 图像的 SSIM 值
    
    参数:
    - image1: 形状为 (3, h, w) 的 NumPy 数组，代表第一张图像
    - image2: 形状为 (3, h, w) 的 NumPy 数组，代表第二张图像
    
    返回:
    - 两张图像的平均 SSIM 值
    """
    assert image1.shape == image2.shape, "两张图像必须具有相同的形状"
    ssim_total = 0.0
    for i in range(3):
        ssim_value, _ = ssim(image1[i], image2[i], full=True,data_range = 255)
        ssim_total += ssim_value
    return ssim_total / 3

def calculate_bpp_psnr_jepg(image,quality):
    output_file_path = "tem.jp2"
    compression_params = [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, quality] 
    result = cv2.imwrite(output_file_path, image, compression_params)
    compressed_image = cv2.imread(output_file_path)
    real_bpp = (os.path.getsize(output_file_path) * 8)/(compressed_image.shape[0]*compressed_image.shape[1])
    mse = np.mean((image.astype(np.float32) - compressed_image.astype(np.float32)) ** 2)
    psnr = 10 * np.log10(255 * 255 / np.mean((image.astype(np.float32) - compressed_image.astype(np.float32)) ** 2))
    ssim = calculate_ssim(image.transpose(2, 0, 1),compressed_image.transpose(2, 0, 1))
    os.remove(output_file_path)
    return real_bpp,mse,psnr,ssim

def calculate_bpp_psnr_bpg(image,quality):
    width = image.shape[1]
    height = image.shape[0]
    tem_output_image = "bpg_tem.png"
    temp_hevc_path = 'temp.hevc'
    tem_png_path = 'bpg_re_temp.png'
    if os.path.exists(tem_output_image):
        os.remove(tem_output_image)
    if os.path.exists(temp_hevc_path):
        os.remove(temp_hevc_path)
    if os.path.exists(tem_png_path):
        os.remove(tem_png_path)
    cv2.imwrite(tem_output_image, image)
    cmd_convert_to_yuv = [
        'ffmpeg',
        '-i', tem_output_image,
        '-c:v', 'libx265',
        '-crf', str(quality),
        temp_hevc_path
    ]
    with subprocess.Popen(cmd_convert_to_yuv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as process:
        process.communicate()  
    cmd = [
        'ffmpeg',
        '-i', temp_hevc_path,
        '-f', 'image2',
        '-pix_fmt', 'rgba',
        tem_png_path
    ]
    with subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as process:
        process.communicate()  
    image = cv2.imread(tem_output_image)
    compressed_image = cv2.imread(tem_png_path)
    real_bpp = (os.path.getsize(temp_hevc_path) * 8)/(compressed_image.shape[0]*compressed_image.shape[1])
    mse = np.mean((image.astype(np.float32) - compressed_image.astype(np.float32)) ** 2)
    ssim = calculate_ssim(image.transpose(2, 0, 1),compressed_image.transpose(2, 0, 1))
    psnr = 10 * np.log10(255 * 255 / np.mean((image.astype(np.float32) - compressed_image.astype(np.float32)) ** 2))
    if os.path.exists(tem_output_image):
        os.remove(tem_output_image)
    if os.path.exists(temp_hevc_path):
        os.remove(temp_hevc_path)
    if os.path.exists(tem_png_path):
        os.remove(tem_png_path)
    return real_bpp,mse,psnr,ssim

def calculate_bpp_psnr_vvc(image,quality):
    width = image.shape[1]
    height = image.shape[0]
    tem_output_image = "vvc_tem.png"
    tem_output_yuv = "vvc_temp.yuv"
    tem_output_266 = "vvc_temp.266"
    tem_output_re_yuv = "vvc_re_temp.yuv"
    tem_output_re_image = "vvc_re_tem.png"
    if os.path.exists(tem_output_image):
        os.remove(tem_output_image)
    if os.path.exists(tem_output_yuv):
        os.remove(tem_output_yuv)
    if os.path.exists(tem_output_266):
        os.remove(tem_output_266)
    if os.path.exists(tem_output_re_yuv):
        os.remove(tem_output_re_yuv)
    if os.path.exists(tem_output_re_image):
        os.remove(tem_output_re_image)
    cv2.imwrite(tem_output_image, image)
    # ffmpeg -i tem.jp2 -pix_fmt yuv420p temp.yuv
    command = [
        'ffmpeg',
        '-i', tem_output_image,   
        '-pix_fmt', 'yuv420p10le',  
        tem_output_yuv            
    ]
    with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as process:
        process.communicate()  

    # vvencapp -i temp.yuv -s 512x512 --fps 50/1 -o str.266
    command = [
        'vvencapp',
        '-i', tem_output_yuv,  
        '-s', f'{width}x{height}', 
        '--format', 'yuv420_10',  
        '--preset', 'faster', 
        '-q', str(quality),
        '-o', tem_output_266  
    ]
    with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as process:
        process.communicate()  

    # vvdecapp -b str.266 -o re_temp.yuv 
    command = [
        'vvdecapp',
        '-b', tem_output_266,  
        '-o', tem_output_re_yuv  
    ]
    with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as process:
        process.communicate()  # 等待命令执行完成

    command = [
        'ffmpeg',
        '-s',  f'{width}x{height}',  
        '-pix_fmt', 'yuv420p10le',  
        '-i', tem_output_re_yuv,  
        '-frames:v', '1',  
        tem_output_re_image  
    ]
    with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as process:
        process.communicate()  # 等待命令执行完成

    image = cv2.imread(tem_output_image)
    compressed_image = cv2.imread(tem_output_re_image)
    real_bpp = (os.path.getsize(tem_output_266) * 12)/(compressed_image.shape[0]*compressed_image.shape[1])
    mse = np.mean((image.astype(np.float32) - compressed_image.astype(np.float32)) ** 2)
    ssim = calculate_ssim(image.transpose(2, 0, 1),compressed_image.transpose(2, 0, 1))
    psnr = 10 * np.log10(255 * 255 / np.mean((image.astype(np.float32) - compressed_image.astype(np.float32)) ** 2))
    if os.path.exists(tem_output_image):
        os.remove(tem_output_image)
    if os.path.exists(tem_output_yuv):
        os.remove(tem_output_yuv)
    if os.path.exists(tem_output_266):
        os.remove(tem_output_266)
    if os.path.exists(tem_output_re_yuv):
        os.remove(tem_output_re_yuv)
    if os.path.exists(tem_output_re_image):
        os.remove(tem_output_re_image)
    return real_bpp,mse,psnr,ssim

def operate_jepg(images):
    bpps = []
    mses = []
    psnrs = []
    msssims = []
    image_number = images.shape[0]

    for quality in qualities_jepg:
        reconstruction_loss = AverageMeter()
        bpp_loss = AverageMeter()
        psnr = AverageMeter()
        msssim = AverageMeter()
        for i in tqdm(range(int(image_number)), desc=f'quality = {quality}'):
            real_bpp, mse, psnr_, msssim_ = calculate_bpp_psnr_jepg(
                image = images[i,:,:,:].transpose(1, 2, 0),     # (h,w,3)
                quality = quality 
            )
            reconstruction_loss.update(mse)
            bpp_loss.update(real_bpp)
            psnr.update(psnr_)
            msssim.update(msssim_)
        mses.append(reconstruction_loss.avg.item())
        bpps.append(bpp_loss.avg)
        psnrs.append(psnr.avg.item())
        msssims.append(msssim.avg.item())
    return {
        "bpp": bpps,
        "mse": mses,
        "psnr": psnrs,
        "ms-ssim":msssims
    }

def operate_bpg(images):
    bpps = []
    mses = []
    psnrs = []
    msssims = []
    image_number = images.shape[0]

    for quality in qualities_bpg:
        reconstruction_loss = AverageMeter()
        bpp_loss = AverageMeter()
        psnr = AverageMeter()
        msssim = AverageMeter()
        for i in tqdm(range(int(image_number)), desc=f'quality = {quality}'):
            real_bpp, mse, psnr_, msssim_ = calculate_bpp_psnr_bpg(
                image = images[i,:,:,:].transpose(1, 2, 0),
                quality = quality
            )
            reconstruction_loss.update(mse)
            bpp_loss.update(real_bpp)
            psnr.update(psnr_)
            msssim.update(msssim_)
        mses.append(reconstruction_loss.avg.item())
        bpps.append(bpp_loss.avg)
        psnrs.append(psnr.avg.item())
        msssims.append(msssim.avg.item())
    return {
        "bpp": bpps,
        "mse": mses,
        "psnr": psnrs,
        "ms-ssim":msssims
    }

def operate_vvc(images):
    bpps = []
    mses = []
    psnrs = []
    msssims = []
    image_number = images.shape[0]

    for quality in qualities_vvc:
        reconstruction_loss = AverageMeter()
        bpp_loss = AverageMeter()
        psnr = AverageMeter()
        msssim = AverageMeter()
        for i in tqdm(range(int(image_number)), desc=f'quality = {quality}'):
            real_bpp, mse, psnr_, msssim_ = calculate_bpp_psnr_vvc(
                image = images[i,:,:,:].transpose(1, 2, 0),
                quality = quality
            )
            reconstruction_loss.update(mse)
            bpp_loss.update(real_bpp)
            psnr.update(psnr_)
            msssim.update(msssim_)
        mses.append(reconstruction_loss.avg.item())
        bpps.append(bpp_loss.avg)
        psnrs.append(psnr.avg.item())
        msssims.append(msssim.avg.item())
    return {
        "bpp": bpps,
        "mse": mses,
        "psnr": psnrs,
        "ms-ssim":msssims
    }   

def main(args):
    """ get data """
    images = np.memmap(args.data_path, dtype=np.uint8, mode='r', shape=(args.image_number,) + (args.image_channel, args.image_height, args.image_width), offset=0)

    """ operator """
    if args.type == "jepg":
        ret = operate_jepg(images)
    elif args.type == "bpg":
        ret = operate_bpg(images)
    elif args.type == "vvc":
        ret = operate_vvc(images)

    print(ret)
    "save result"
    with open(args.result_path, 'w') as file:
        json.dump(ret, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type",type = str, default = "vvc")
    parser.add_argument('--data_path',type=str, default= "data/camvid.npy")
    parser.add_argument('--image_number',type=int, default= 600)
    parser.add_argument('--image_channel',type=int, default= 3)
    parser.add_argument('--image_height',type=int, default= 512)
    parser.add_argument('--image_width',type=int, default= 512)
    parser.add_argument('--result_path',type=str, default="result/bpp_psnr/base/camvid/vvc-new.json")
    args = parser.parse_args()
    main(args)

# python example/get_bpp_psnr_json_traditional.py --type bpg --data_path data/soda.npy --image_number 600 --image_channel 3 --image_height 512 --image_width 512 --result_path soda_bpg.json
