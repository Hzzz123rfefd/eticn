import argparse
import sys
import os
sys.path.append(os.getcwd())
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from compressai.utils import *
from compressai import models

def read_image(image_path):
    image = cv2.imread(image_path)
    # h, w, c = image.shape
    resized_image = cv2.resize(image, (512, 512))
    # resized_image = image[(int)((h/2) - (self.target_height/2)):(int)((h/2) + (self.target_height/2)),(int)((w/2) - (self.target_width/2)):(int)((w/2) + (self.target_width/2)),:]
    if len(resized_image.shape) == 2:  
        resized_image = np.expand_dims(image, axis=0)
    return resized_image

def main(args):
    plt.rcParams.update({
    'font.family': ['Times New Roman', 'SimSun'],
    'axes.unicode_minus': False,                 
    'axes.labelweight': 'bold',
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12
    })
    os.makedirs(args.save_dir, exist_ok = True)
    config = load_config(args.model_config_path)
    """ get model"""
    net = models[config["model_type"]](**config["model"]).to(config["model"]["device"])

    net.load_pretrained(
        save_model_dir = args.model_path
    )
    
    image = torch.tensor(read_image(args.image_path), dtype=torch.float32) / 255.0
    image = image.permute(2, 0, 1).unsqueeze(0).to(net.device)

    y, _ = net.image_transform_encoder(image)
    y_hat = net.gaussian_conditional.quantize(
        y, "noise"
    )
    original_psnr = calculate_psnr(image[0].cpu() * 255, net.image_transform_decoder(y_hat)[0].cpu() * 255)
    psnrs = []
    for i in range(32):
        y_tmp = y_hat.clone()
        y_tmp[:, i, :, :] = 0.0 
        x_hat = net.image_transform_decoder(y_tmp)
        psnrs.append(calculate_psnr(image[0].cpu() * 255, x_hat[0].cpu() * 255))
        
    plt.figure()
    channels = np.arange(32)
    plt.bar(channels, psnrs)
    plt.axhline(
        y=original_psnr,
        color='red',
        linestyle='--',
        linewidth=2,
        label="原始PSNR"
    )
    # plt.xlabel("Channel Index")
    plt.xlabel("通道索引")
    plt.ylabel("PSNR(dB)")
    # plt.title("Channel Influence (PSNR)")
    plt.title("通道影响 (PSNR)")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.save_dir, "result1.png"), dpi=300, bbox_inches='tight')
    plt.show()
    
    psnrs = []
    for scale in range(5):
        y_tmp = y_hat.clone()
        y_tmp[:, 0, :, :] =  y_tmp[:, 0, :, :] * (0.1 + 0.2 *  scale)
        x_hat = net.image_transform_decoder(y_tmp)
        psnrs.append(calculate_psnr(image[0].cpu() * 255, x_hat[0].cpu() * 255))
    plt.figure()
    scales = [0.1, 0.3, 0.5, 0.7, 0.9]
    plt.bar(scales, psnrs, width=0.1)
    plt.axhline(
        y=original_psnr,
        color='red',
        linestyle='--',
        linewidth=2,
        label="原始PSNR"
    )
    # plt.xlabel("Scale Factor")
    plt.xlabel("缩放因子")
    plt.ylabel("PSNR(dB)")
    # plt.title("Scale Influence (PSNR)")
    plt.title("缩放影响 (PSNR)")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.save_dir, "result2.png"), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", type=str, default = "config/eticn/eticn.yml")
    parser.add_argument("--image_path", type=str, default = "datasets/camvid/train/0016E5_04590.png")
    parser.add_argument("--model_path", type=str, default = "saved_model/eticn/0.0932/")
    parser.add_argument("--save_dir", type=str, default = "result/2-fig22")
    args = parser.parse_args()
    main(args)