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

def plot_heatmap(save_path: str, data: np.ndarray):
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    if data.shape != (32, 32):
        raise ValueError("输入数组必须是 shape=(32,32) 或 (1,32,32)")

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 17

    plt.figure(figsize=(6, 6))
    im = plt.imshow(data, cmap='seismic', vmin=-3, vmax=3)

    cbar = plt.colorbar(im)

    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()

    plt.savefig(save_path, dpi=1200)
    plt.close()


def main(args):
    config = load_config(args.model_config_path)
    """ get model"""
    net = models[config["model_type"]](**config["model"]).to(config["model"]["device"])

    net.load_pretrained(
        save_model_dir = args.model_path
    )

    image = torch.tensor(read_image(args.image_path), dtype=torch.float32) / 255.0
    image = image.permute(2, 0, 1).unsqueeze(0).to(net.device)

    scale, rescale, s = net.get_scale(4, False)

    y = net.g_a(image)
    z = net.h_a(y)
    z_hat, z_likelihoods = net.entropy_bottleneck(z)
    scales_hat = net.h_s(z_hat)
    y_hat = net.gaussian_conditional.quantize(y * scale, "dequantize" )
    y_hat = y_hat * rescale

    t = torch.ones((1)) * (net.levels - 4)
    t = t.to(net.device)
    predict_noisy = net.predict_model(y_hat, t)
    y_hat2 = y_hat - predict_noisy * rescale

    c = 20
    y = y[0, c, :, :].detach().cpu().numpy()
    y_hat = y_hat[0, c, :, :].detach().cpu().numpy()
    y_hat2 = y_hat2[0, c, :, :].detach().cpu().numpy()
    y_hat2 = y_hat2 * 0.9 + y * 0.1

    mse = np.mean((y - y_hat) ** 2)
    print("MSE1:", mse)
    mse = np.mean((y - y_hat2) ** 2)
    print("MSE2:", mse)

    os.makedirs(args.save_dir, exist_ok = True)
    plot_heatmap(os.path.join(args.save_dir, "y.png"),  y)
    plot_heatmap(os.path.join(args.save_dir, "y_hat.png"), y_hat)
    plot_heatmap(os.path.join(args.save_dir, "y_hat2.png"), y_hat2)
    plot_heatmap(os.path.join(args.save_dir, "y_hat_diff.png"), np.abs(y - y_hat))
    plot_heatmap(os.path.join(args.save_dir, "y_hat2_diff.png"), np.abs(y - y_hat2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", type=str, default = "config/imagenetc/viccqvr3.yml")
    parser.add_argument("--image_path", type=str, default = "datasets/imagenet/imgs/ILSVRC2012_val_00000028.JPEG")
    parser.add_argument("--model_path", type=str, default = "saved_model/vbr/imagenet/viccqvr3/")
    parser.add_argument("--save_dir", type=str, default = "result/fig3")
    args = parser.parse_args()
    main(args)


