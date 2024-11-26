import numpy as np
import torch
from torch.utils.data import Dataset
from compressai.utils import *


class DatasetForImageReader(Dataset):
    def __init__(
        self, 
        train_image_folder:str = None,
        test_image_folder:str = None,
        valid_image_folder:str = None,
        train_image_path:str = None,
        test_image_path:str = None,
        valid_image_path:str = None,
        data_type:str = "train"
    ):
        if train_image_folder != None:
            if data_type == "train":
                pass
        else:
            if data_type == "train":
                self.dataset = np.load(train_image_path)
            elif data_type == "test":
                self.dataset = np.load(test_image_path)
            else:
                self.dataset = np.load(valid_image_path)
        
        self.dataset = self.dataset / 255.0
        self.total_samples = len(self.dataset)

    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        output = {}
        output["images"] = torch.tensor(self.dataset[idx], dtype=torch.float32)
        return output
    
    def collate_fn(self,batch):
        return recursive_collate_fn(batch)

class DatasetForETICN(DatasetForImageReader):
    def __init__(
        self, 
        train_image_folder:str = None,
        test_image_folder:str = None,
        valid_image_folder:str = None,
        train_image_path:str = None,
        test_image_path:str = None,
        valid_image_path:str = None,
        train_label_folder:str = None,
        test_label_folder:str = None,
        valid_label_folder:str = None,
        train_label_path:str = None,
        test_label_path:str = None,
        valid_label_path:str = None,
        data_type:str = "train"
    ):
        super().__init__(train_image_folder, test_image_folder, valid_image_folder, train_image_path,test_image_path, valid_image_path, data_type)
        if train_label_folder != None:
            if data_type == "train":
                pass
        else:
            if data_type == "train":
                self.label = np.load(train_label_path)
            elif data_type == "test":
                self.label = np.load(test_label_path)
            else:
                self.label = np.load(valid_label_path)
        self.label = self.label / 255.0

    def __getitem__(self, idx):
        output = {}
        output["images"] = torch.tensor(self.dataset[idx], dtype=torch.float32)
        output["label"] = torch.tensor(self.label[idx], dtype=torch.float32)
        return output