import numpy as np
import torch
from torch.utils.data import DataLoader
from compressai import models,datasets
from compressai.utils import load_config
import argparse
import os

def main(args):
    for root, _, files in os.walk(args.model_config_dir):
        for file in files:
            if file.endswith(".yml"):
                model_config_path = os.path.join(root, file)
                config = load_config(model_config_path)

                """ get net struction"""
                net = models[config["model_type"]](**config["model"]).to("cuda")
                
                parameters = np.load(config["model"]["university_pretrain_path"]) 
                parameters = torch.from_numpy(parameters)
                net.universal_context.from_pretrain(parameters, requires_grad = False)
                full_model = torch.load(config["model"]["finetune_model_dir"] + "/model.pth")
                other_model = {
                    k: v for k, v in full_model.items()
                    if not k.startswith("universal_context")
                }
                net.load_state_dict(other_model, strict=False)
                os.makedirs(os.path.join(config["logging"]["save_dir"], str(config["model"]["lamda"])), exist_ok=True)
                net.save_pretrained(save_model_dir = config["logging"]["save_dir"], lamda = config["model"]["lamda"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_dir", type=str, default = "config/ablation/eticn_256_32/")
    args = parser.parse_args()
    main(args)