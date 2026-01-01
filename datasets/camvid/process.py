import argparse
import json
import os
import shutil
import cv2
import numpy as np

colors = {
    'person': [0, 64, 64],   
    'car': [128, 0, 64],      
    'bicycle': [192, 128, 0],  
    'truck': [192, 128, 192],  
    'train': [128, 64, 192]   
}

def get_label(label_path):
    image = cv2.imread(label_path)
    label = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
    for color_name, rgb_value in colors.items():
        mask = cv2.inRange(image, np.array(rgb_value) - np.array([10, 10, 10]), np.array(rgb_value) + np.array([10, 10, 10]))
        label[mask == 255] = 0
    return label
        
def get_train_data(data_dir, label_dir, output_path, output_label_dir):
    with open(output_path, "w", encoding="utf-8") as jsonl_file:
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.lower().endswith(".png"):
                    image_path = os.path.join(root, file)
                    label_filename = f"{os.path.splitext(file)[0]}_L.png"
                    label_path = label_dir + label_filename
                    label = get_label(label_path)
                    cv2.imwrite(output_label_dir + label_filename, label)
                    record = {"image_path": image_path, "label_path": output_label_dir + label_filename}
                    jsonl_file.write(json.dumps(record) + "\n")

def main(args):
    shutil.rmtree(args.output_dir)
    os.makedirs(name = args.output_dir, exist_ok = True)
    os.makedirs(name = args.output_dir + "label/", exist_ok = True)
    get_train_data(
        data_dir = "datasets/camvid/train/",
        label_dir = "datasets/camvid/train_labels/",
        output_path = args.output_dir + "train.jsonl",
        output_label_dir = args.output_dir + "label/",
    )
    
    get_train_data(
        data_dir = "datasets/camvid/test/",
        label_dir = "datasets/camvid/test_labels/",
        output_path = args.output_dir + "test.jsonl",
        output_label_dir = args.output_dir + "label/",
    )
        
    get_train_data(
        data_dir = "datasets/camvid/val/",
        label_dir = "datasets/camvid/val_labels/",
        output_path = args.output_dir + "val.jsonl",
        output_label_dir = args.output_dir + "label/",
    )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",type=str,default = "camvid_train/")
    args = parser.parse_args()
    main(args)
