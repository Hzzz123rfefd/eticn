import sys
import os

import numpy as np
import torch
from tqdm import tqdm
sys.path.append(os.getcwd())

import argparse
from torch.utils.data import DataLoader
from compressai import models, datasets
from compressai.utils import load_config


def main(args):
    config = load_config(args.model_config_path)

    """ get model"""
    net = models[config["model_type"]](**config["model"])
    
    """get data loader"""
    """get data loader"""
    dataset = datasets[config["dataset_type"]](
        target_width = config["dataset"]["target_width"],
        target_height = config["dataset"]["target_height"],
        valid_data_path = args.data_path,
        data_type = "valid"
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size = config["traininng"]["batch_size"], 
        shuffle = False,
        collate_fn = dataset.collate_fn
    )

    lamdas = [0.0002, 0.0004, 0.0009, 0.0016, 0.0036, 0.0081]
    data = []
    
    for lamda in lamdas:    
        tem_data = []
        net.load_pretrained(
            save_model_dir = args.model_path,
            lamda = lamda
        )
        
        pbar = tqdm(dataloader,desc = f"lamda = {lamda}")
        with torch.no_grad():
            for batch_id, inputs in enumerate(pbar):
                output = net.forward(inputs)
                tem_data.append(output["latent"].cpu())
                
        data.append(torch.cat(tem_data, dim=0))
    
    data = torch.stack(data, dim=1)
    np.save(args.result_path, data.numpy())
                
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", type=str, default = "config/eticn.yml")
    parser.add_argument("--data_path", type=str, default = "camvid_train/val.jsonl")
    parser.add_argument("--model_path", type=str, default = "saved_model/")
    parser.add_argument("--result_path", type=str, default = "valid.npy")
    args = parser.parse_args()
    main(args)
