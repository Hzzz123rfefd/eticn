import cv2
import datasets
import numpy as np
import torch
from torch.utils.data import Dataset
from compressai.utils import *

class DatasetForETICN(Dataset):
    def __init__(
        self, 
        target_width: int,
        target_height: int,
        train_data_path:str = None,
        test_data_path:str = None,
        valid_data_path:str = None,
        data_type:str = "train"
    ):
        self.target_width = target_width
        self.target_height = target_height
        if data_type == "train":
            self.data_path = train_data_path
        elif data_type == "test":
            self.data_path = test_data_path
        elif data_type == "valid":
            self.data_path = valid_data_path
        self.dataset = datasets.load_dataset('json', data_files = self.data_path, split = "train")
        self.total_len = len(self.dataset)
     
    def __len__(self):
        return self.total_len
    
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
        # resized_image = cv2.resize(image, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)
        if len(resized_image.shape) == 2:  
            resized_image = np.expand_dims(image, axis=0)
        return resized_image

    def collate_fn(self, batch):
        return recursive_collate_fn(batch)
    

# class DatasetForImageReader(Dataset):
#     def __init__(
#         self, 
#         train_image_folder:str = None,
#         test_image_folder:str = None,
#         valid_image_folder:str = None,
#         train_image_path:str = None,
#         test_image_path:str = None,
#         valid_image_path:str = None,
#         data_type:str = "train"
#     ):
#         if train_image_folder != None:
#             if data_type == "train":
#                 pass
#         else:
#             if data_type == "train":
#                 self.dataset = np.load(train_image_path)
#             elif data_type == "test":
#                 self.dataset = np.load(test_image_path)
#             else:
#                 self.dataset = np.load(valid_image_path)
        
#         self.dataset = self.dataset / 255.0
#         self.total_samples = len(self.dataset)

#     def __len__(self):
#         return self.total_samples
    
#     def __getitem__(self, idx):
#         output = {}
#         output["images"] = torch.tensor(self.dataset[idx], dtype=torch.float32)
#         return output
    
#     def collate_fn(self,batch):
#         return recursive_collate_fn(batch)

# class DatasetForETICN(DatasetForImageReader):
#     def __init__(
#         self, 
#         train_image_folder:str = None,
#         test_image_folder:str = None,
#         valid_image_folder:str = None,
#         train_image_path:str = None,
#         test_image_path:str = None,
#         valid_image_path:str = None,
#         train_label_folder:str = None,
#         test_label_folder:str = None,
#         valid_label_folder:str = None,
#         train_label_path:str = None,
#         test_label_path:str = None,
#         valid_label_path:str = None,
#         data_type:str = "train"
#     ):
#         super().__init__(train_image_folder, test_image_folder, valid_image_folder, train_image_path,test_image_path, valid_image_path, data_type)
#         if train_label_folder != None:
#             if data_type == "train":
#                 pass
#         else:
#             if data_type == "train":
#                 self.label = np.load(train_label_path)
#             elif data_type == "test":
#                 self.label = np.load(test_label_path)
#             else:
#                 self.label = np.load(valid_label_path)
#         self.label = self.label / 255.0

#     def __getitem__(self, idx):
#         output = {}
#         output["images"] = torch.tensor(self.dataset[idx], dtype=torch.float32)
#         output["label"] = torch.tensor(self.label[idx], dtype=torch.float32)
#         return output