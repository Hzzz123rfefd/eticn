import sys
import os

import torch
sys.path.append(os.getcwd())
import argparse
from compressai import models
from compressai.utils import load_config
import torchprofile



def main(args):
    config = load_config(args.model_config_path)
    """ get model"""
    net = models[config["model_type"]](**config["model"]).to(config["model"]["device"])
    
    total_params = sum(p.numel() for p in net.parameters()) / 1000000
    print(f"Total parameters: {total_params:,}M")
    
    inputs = {
        "image": torch.randn(1, 3, 512, 512),  # 第一个输入
    }
    flops = torchprofile.profile_macs(net, inputs)
    print("FLOPs:", flops / 1000000000, "G")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", type=str, default = "config/camvid/vic.yml")
    args = parser.parse_args()
    main(args)
    
