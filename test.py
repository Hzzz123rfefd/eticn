import torch
from torch.utils.data import DataLoader
from compressai import models,datasets
from compressai.utils import load_config

import argparse
config = load_config("config/eticnqvrf.yml")
model = models[config["model_type"]](**config["model"])
checkpoint = torch.load('saved_model/0.0001/model.pth', map_location="cuda")
model.load_state_dict(checkpoint, strict=False)
model.save_pretrained("saved_model/eticnqvrf")