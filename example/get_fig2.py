import argparse
import sys
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
sys.path.append(os.getcwd())

from compressai import models
from compressai.utils import load_config

def read_image(image_path):
    image = cv2.imread(image_path)
    # h, w, c = image.shape
    resized_image = cv2.resize(image, (512, 512))
    # resized_image = image[(int)((h/2) - (self.target_height/2)):(int)((h/2) + (self.target_height/2)),(int)((w/2) - (self.target_width/2)):(int)((w/2) + (self.target_width/2)),:]
    if len(resized_image.shape) == 2:  
        resized_image = np.expand_dims(image, axis=0)
    return resized_image

def save_image(array: np.ndarray, filename: str, colormap=cv2.COLORMAP_JET):
    if np.issubdtype(array.dtype, np.floating):
        min_val = np.min(array)
        max_val = np.max(array)
        if max_val == min_val:
            array = np.zeros_like(array)  
        else:
            array = (array - min_val) / (max_val - min_val) * 255
        array = array.astype(np.uint8)
    
    elif array.dtype != np.uint8:
        raise TypeError("数组类型必须是 float32 或 uint8。")

    # array = cv2.applyColorMap(array, colormap)
    success = cv2.imwrite(filename, array)
    if not success:
        raise IOError(f"保存图像失败：{filename}")

    print(f"保存成功: {filename}")

def get_linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)

def get_alpha_schedule(beta):
    alpha = 1.0 - beta
    alpha_bar = np.cumprod(alpha)
    return alpha, alpha_bar

def diffusion_forward_process(image, timesteps, save_steps, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image = image.astype(np.float32) / 255.0  # normalize to [0,1]

    betas = get_linear_beta_schedule(timesteps)
    alphas, alpha_bars = get_alpha_schedule(betas)

    for t in range(timesteps):
        noise = np.random.normal(0, 1, image.shape).astype(np.float32)
        sqrt_alpha_bar = np.sqrt(alpha_bars[t])
        sqrt_one_minus_alpha_bar = np.sqrt(1 - alpha_bars[t])
        x_t = sqrt_alpha_bar * image + sqrt_one_minus_alpha_bar * noise
        x_t_clipped = np.clip(x_t * 255.0, 0, 255).astype(np.uint8)

        if t in save_steps:
            save_name = f"step_{t:04d}.png"
            cv2.imwrite(os.path.join(output_dir, save_name), x_t_clipped)
            print(f"Saved: {save_name}")

def main(args):
    os.makedirs(args.save_dir, exist_ok = True)
    config = load_config(args.model_config_path)
    """ get model"""
    net = models[config["model_type"]](**config["model"]).to(config["model"]["device"])

    net.load_pretrained(
        save_model_dir = args.model_path
    )
    
    image = torch.tensor(read_image(args.image_path), dtype=torch.float32) / 255.0
    image = image.permute(2, 0, 1).unsqueeze(0).to(net.device)
    c = 0
    y = net.g_a(image) * 2
    save_image(y[0, c, :, :].detach().cpu().numpy(), os.path.join(args.save_dir, "y_hat.jpg"));
    for i in range(7):
        scale, rescale, s = net.get_scale(i, False)
        y_hat = net.gaussian_conditional.quantize(y * scale, "dequantize" )* rescale
        y_hat = y_hat[0, c, :, :].detach().cpu().numpy()
        save_image(y_hat, os.path.join(args.save_dir, "y_hat" + str(i) + ".jpg"));

    timesteps = 1000
    save_steps = [1, 100, 200, 300, 400, 500, 600, 998, 999]
    image = cv2.imread(args.image_path)
    diffusion_forward_process(image, timesteps, save_steps, args.save_dir)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", type=str, default = "config/imagenetc/viccqvr3.yml")
    parser.add_argument("--image_path", type=str, default = "datasets/imagenet/imgs/ILSVRC2012_val_00000028.JPEG")
    parser.add_argument("--model_path", type=str, default = "saved_model/vbr/imagenet/viccqvr3/")
    parser.add_argument("--save_dir", type=str, default = "result/2-fig2")
    args = parser.parse_args()
    main(args)
    
