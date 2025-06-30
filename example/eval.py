import json
import sys
import os
sys.path.append(os.getcwd())
import argparse
from torch.utils.data import DataLoader
from compressai import models, datasets
from compressai.utils import load_config


def main(args):
    config = load_config(args.model_config_path)

    lamdas = [0.18]
    bpps = []
    psnrs=[]
    """ get model"""
    net = models[config["model_type"]](**config["model"]).to(config["model"]["device"])
    
    for lamda in lamdas:
        net.load_pretrained(
            save_model_dir = args.model_path,
            lamda = lamda
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
    
        ret = net.eval_model(val_dataloader = dataloader)
        psnrs.append(float(ret["PSNR"]))
        bpps.append(float(ret["bpp"]))
        data = {
            "PSNR":psnrs,
            "bpp":bpps
        }
    
    with open(args.save_path, "w") as f:
        json.dump(data, f, indent=4)
    
        
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", type=str, default = "config/camvid/vaic.yml")
    parser.add_argument("--data_path", type=str, default = "camvid_train/val.jsonl")
    parser.add_argument("--model_path", type=str, default = "saved_model/fbr/vaic-camvid/")
    parser.add_argument("--save_path", type=str, default = "result/stf-imagenet.json")
    args = parser.parse_args()
    main(args)
