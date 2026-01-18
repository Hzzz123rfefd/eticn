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
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    """ get model"""
    net = models[config["model_type"]](**config["model"]).to(config["model"]["device"])
    net.eval()
    net.load_pretrained(
        save_model_dir = args.model_path,
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
    with open(args.save_path, "w") as f:
        json.dump(ret, f, indent=4)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_config_path", type=str, default = "config/eticn/eticn_qeevrf_stage3.yml")
    parser.add_argument("--data_path", type=str, default = "datasets/camvid/camvid_train/val.jsonl")
    parser.add_argument("--model_path", type=str, default = "saved_model/eticn/eticn_qeevrf/stage3")
    parser.add_argument("--save_path", type=str, default = "result/R-D/vbr/ablation/QEEVRF.json")
    args = parser.parse_args()
    main(args)
    
# python  example/eval_vbr.py --model_config_path config/camvid/eticn/eticn_qeevrf_stage3.yml --data_path datasets/camvid/camvid_train/val.jsonl --model_path saved_model/vbr/camvid/eticn_qeevrf/stage3 --save_path result/R-D/vbr/camvid/[Ours].json