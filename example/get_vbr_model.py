import argparse
import torch
import sys
import os
sys.path.append(os.getcwd())
from compressai import models
from compressai.utils import load_config


def main(args):
    vbr_config = load_config(args.vbr_model_config_path)
    fbr_config = load_config(args.fbr_model_config_path)
    os.makedirs(name = vbr_config['logging']['save_dir'], exist_ok = True)
    checkpoint = torch.load(fbr_config['logging']['save_dir'] + "/0.18/model.pth", map_location="cuda")
    model = models[vbr_config["model_type"]](**vbr_config["model"])
    model.load_state_dict(checkpoint, strict=False)
    model.save_pretrained(vbr_config['logging']['save_dir'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vbr_model_config_path", type=str, default = "config/imagenetc/stfqvrf.yml")
    parser.add_argument("--fbr_model_config_path", type=str, default = "config/imagenetc/stf.yml")
    args = parser.parse_args()
    main(args)



