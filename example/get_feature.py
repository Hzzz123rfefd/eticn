import json
import sys
import os
import cv2
import numpy as np
import torch
sys.path.append(os.getcwd())

import argparse
from torch.utils.data import DataLoader
from compressai import models, datasets
from compressai.utils import load_config

def read_image(image_path):
    image = cv2.imread(image_path)
    # h, w, c = image.shape
    resized_image = cv2.resize(image, (512, 512))
    # resized_image = image[(int)((h/2) - (self.target_height/2)):(int)((h/2) + (self.target_height/2)),(int)((w/2) - (self.target_width/2)):(int)((w/2) + (self.target_width/2)),:]
    if len(resized_image.shape) == 2:  
        resized_image = np.expand_dims(image, axis=0)
    return resized_image

def save_gray_image(image_path, img):
    min_val = img.min()
    max_val = img.max()
    if max_val - min_val < 1e-5:
        norm_img = np.zeros_like(img, dtype=np.uint8) 
    else:
        norm_img = (img - min_val) / (max_val - min_val) * 66
        norm_img = norm_img.astype(np.uint8)
    color_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
    cv2.imwrite(image_path, color_img)


config = load_config("config/imagenetc/viccqvr3.yml")
""" get model"""
net = models[config["model_type"]](**config["model"]).to(config["model"]["device"])

net.load_pretrained(
    save_model_dir = "saved_model/vbr/imagenet/viccqvr3/"
)

image = torch.tensor(read_image("datasets/imagenet/imgs/ILSVRC2012_val_00000028.JPEG"), dtype=torch.float32) / 255.0
image = image.permute(2, 0, 1).unsqueeze(0)

image = image.to(net.device)
b, _, _, _ = image.shape
scale, rescale, s = net.get_scale(4, False)

y = net.g_a(image)
z = net.h_a(y)
z_hat, z_likelihoods = net.entropy_bottleneck(z)
scales_hat = net.h_s(z_hat)
y_hat, noisy = net.gaussian_conditional.quantize(y * scale, "noise" , return_noisy = True)
y_hat = y_hat * rescale

t = torch.ones((1)) * (net.levels - 4)
t = t.to(net.device)
predict_noisy = net.predict_model(y_hat, t)
y_hat2 = y_hat - predict_noisy * rescale

c = 20
y = y[0, c, :, :].detach().cpu().numpy()
y_hat = y_hat[0, c, :, :].detach().cpu().numpy()
y_hat2 = y_hat2[0, c, :, :].detach().cpu().numpy()

mse = np.mean((y - y_hat) ** 2)
print("MSE1:", mse)
mse = np.mean((y - y_hat2) ** 2)
print("MSE2:", mse)

save_gray_image("y.png", y)
save_gray_image("y_hat.png", y_hat)
save_gray_image("y_hat2.png", y_hat2)
save_gray_image("y_hat_diff.png", np.abs(y - y_hat))
save_gray_image("y_hat2_diff.png", np.abs(y - y_hat2))




