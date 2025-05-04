import cv2
import datasets
import numpy as np
import torch
from torch.utils.data import Dataset
from compressai.utils import *
from third_party.DL_Pipeline.src.dataset import DatasetBase

class DatasetForETICN(DatasetBase):
    def __init__(
        self, 
        target_width: int,
        target_height: int,
        train_data_path:str = None,
        test_data_path:str = None,
        valid_data_path:str = None,
        data_type:str = "train"
    ):
        super().__init__(train_data_path, test_data_path, valid_data_path, data_type)
        self.target_width = target_width
        self.target_height = target_height
     
    def __getitem__(self, idx):
        output = {}
        image_path = self.dataset[idx]['image_path']
        label_path =(self.dataset[idx]['label_path'])
        image = torch.tensor(self.read_image(image_path), dtype=torch.float32) / 255.0
        label = self.read_image(label_path)
        label = np.mean(label, axis=-1)
        label = torch.tensor(label, dtype=torch.float32) / 255.0
        output["image"] = image.permute(2, 0, 1)
        output["label"] = label
        return output

    def read_image(self, image_path):
        image = cv2.imread(image_path)
        h, w, c = image.shape
        resized_image = image[(int)((h/2) - (self.target_height/2)):(int)((h/2) + (self.target_height/2)),(int)((w/2) - (self.target_width/2)):(int)((w/2) + (self.target_width/2)),:]
        if len(resized_image.shape) == 2:  
            resized_image = np.expand_dims(image, axis=0)
        return resized_image



# class DatasetForETICN(Dataset):
#     def __init__(
#         self, 
#         target_width: int,
#         target_height: int,
#         train_data_path:str = None,
#         test_data_path:str = None,
#         valid_data_path:str = None,
#         data_type:str = "train"
#     ):
#         self.target_width = target_width
#         self.target_height = target_height
#         if data_type == "train":
#             self.data_path = train_data_path
#         elif data_type == "test":
#             self.data_path = test_data_path
#         elif data_type == "valid":
#             self.data_path = valid_data_path
#         self.dataset = datasets.load_dataset('json', data_files = self.data_path, split = "train")
#         self.total_len = len(self.dataset)
     
#     def __len__(self):
#         return self.total_len
    
#     def __getitem__(self, idx):
#         output = {}
#         image_path = self.dataset[idx]['image_path']
#         label_path =(self.dataset[idx]['label_path'])
#         image = torch.tensor(self.read_image(image_path), dtype=torch.float32) / 255.0
#         label = self.read_image(label_path)
#         label = np.mean(label, axis=-1)
#         label = torch.tensor(label, dtype=torch.float32) / 255.0
#         output["image"] = image.permute(2, 0, 1)
#         output["label"] = label
#         return output

#     def read_image(self, image_path):
#         image = cv2.imread(image_path)
#         h, w, c = image.shape
#         resized_image = image[(int)((h/2) - (self.target_height/2)):(int)((h/2) + (self.target_height/2)),(int)((w/2) - (self.target_width/2)):(int)((w/2) + (self.target_width/2)),:]
#         # resized_image = cv2.resize(image, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)
#         if len(resized_image.shape) == 2:  
#             resized_image = np.expand_dims(image, axis=0)
#         return resized_image

#     def collate_fn(self, batch):
#         return recursive_collate_fn(batch)
    