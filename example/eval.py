import sys
import os
sys.path.append(os.getcwd())
import argparse
from torch.utils.data import DataLoader
from compressai import models, datasets
from compressai.utils import load_config


def main(args):
    config = load_config(args.model_config_path)

    """ get model"""
    net = models[config["model_type"]](**config["model"])
    
    net.load_pretrained(
        save_model_dir = args.model_path,
        lamda = args.lamda
    )
    
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
    
    net.eval_model(val_dataloader = dataloader)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", type=str, default = "config/eticn.yml")
    parser.add_argument("--data_path", type=str, default = "camvid_train/val.jsonl")
    parser.add_argument("--model_path", type=str, default = "saved_model/eticn/")
    parser.add_argument("--lamda", type=str, default = 0.0001)
    args = parser.parse_args()
    main(args)
