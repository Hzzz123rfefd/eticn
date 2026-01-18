import numpy as np
import torch
from torch.utils.data import DataLoader
from compressai import models,datasets
from compressai.utils import load_config
import argparse


def main(args):
    config = load_config(args.model_config_path)

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
    net.save_pretrained(save_model_dir = config["logging"]["save_dir"], lamda = config["model"]["lamda"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", type=str, default = "config/ablation/eticn_512_32/eticn_013.yml")
    args = parser.parse_args()
    main(args)