import argparse
import json
import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

def get_label(label_path):
    image = cv2.imread(label_path)
    return image

def get_train_data(data_dir, label_dir, output_path, output_label_dir):
    with open(output_path, "w", encoding="utf-8") as jsonl_file:
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(".jpg"):
                    image_path = os.path.join(root, file)
                    label_filename = f"{os.path.splitext(file)[0]}_L.png"
                    label_path = label_dir + label_filename
                    label = get_label(label_path)
                    cv2.imwrite(output_label_dir + label_filename, label)
                    record = {"image_path": image_path, "label_path": output_label_dir + label_filename}
                    jsonl_file.write(json.dumps(record) + "\n")

def main(args):
    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(name = args.output_dir, exist_ok = True)
    os.makedirs(name = args.output_dir + "label/", exist_ok = True)
    get_train_data(
        data_dir = "datasets/soda/train/",
        label_dir = "datasets/soda/train_labels/",
        output_path = args.output_dir + "train.jsonl",
        output_label_dir = args.output_dir + "label/",
    )
    
    get_train_data(
        data_dir = "datasets/soda/test/",
        label_dir = "datasets/soda/test_labels/",
        output_path = args.output_dir + "test.jsonl",
        output_label_dir = args.output_dir + "label/",
    )
        
    get_train_data(
        data_dir = "datasets/soda/val/",
        label_dir = "datasets/soda/val_labels/",
        output_path = args.output_dir + "val.jsonl",
        output_label_dir = args.output_dir + "label/",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",type=str,default = "soda_train/")
    args = parser.parse_args()
    main(args)
